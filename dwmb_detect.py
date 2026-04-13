"""YOLOv5 PyTorch tiled inference for DWMB evaluation (no ONNX, no ``ultralytics.YOLO``).

Uses vendored ``models.experimental.attempt_load`` and a local letterbox + decode path so evaluation can
run on ``best.pt`` without loading the Ultralytics high-level API.

Run standalone (from ``dwmb_experiments/backends/yolov5``):

  python dwmb_detect.py --weights path/to/best.pt --source path/to/img.jpg --top-n 50
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import torch

from dwmb_postprocess import decode_yolov5_predictions

if TYPE_CHECKING:
    from numpy.typing import NDArray

Y5_ROOT = Path(__file__).resolve().parent


def _axis_starts(length: int, tile: int, overlap: float) -> list[int]:
    overlap = max(0.0, min(float(overlap), 0.999))
    if length <= 0 or tile <= 0:
        return []
    if length <= tile:
        return [0]
    step = max(1, int(round(tile * (1.0 - overlap))))
    starts: list[int] = []
    pos = 0
    while pos + tile < length:
        starts.append(pos)
        pos += step
    starts.append(length - tile)
    return sorted(set(starts))


def _iter_tile_boxes(iw: int, ih: int, tile: int, overlap: float) -> list[tuple[int, int, int, int]]:
    xs = _axis_starts(iw, tile, overlap)
    ys = _axis_starts(ih, tile, overlap)
    out: list[tuple[int, int, int, int]] = []
    for y0 in ys:
        for x0 in xs:
            x1 = min(x0 + tile, iw)
            y1 = min(y0 + tile, ih)
            out.append((x0, y0, x1, y1))
    return out


def letterbox(
    im: "NDArray[np.uint8]",
    new_shape: tuple[int, int] | int = (640, 640),
    color: tuple[int, int, int] = (114, 114, 114),
    auto: bool = True,
    scale_fill: bool = False,
    scaleup: bool = True,
    stride: int = 32,
) -> tuple[Any, tuple[float, float], tuple[float, float]]:
    """Resize/pad image to ``new_shape``; return image, gain ratio (w,h), (pad_w, pad_h) per side."""
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    ratio = (r, r)
    new_unpad = (round(shape[1] * r), round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scale_fill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = (new_shape[1] / shape[1], new_shape[0] / shape[0])
    dw /= 2.0
    dh /= 2.0
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)


def clip_boxes(boxes: torch.Tensor, shape: tuple[int, int]) -> None:
    """Clip xyxy in-place to ``shape`` (height, width)."""
    h, w = shape[0], shape[1]
    boxes[..., 0].clamp_(0, w)
    boxes[..., 1].clamp_(0, h)
    boxes[..., 2].clamp_(0, w)
    boxes[..., 3].clamp_(0, h)


def scale_boxes(img1_shape: tuple[int, int], boxes: torch.Tensor, img0_shape: tuple[int, int]) -> torch.Tensor:
    """Rescale ``boxes`` (xyxy) from letterboxed ``img1_shape`` to original ``img0_shape`` (h, w)."""
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad_w = (img1_shape[1] - img0_shape[1] * gain) / 2
    pad_h = (img1_shape[0] - img0_shape[0] * gain) / 2
    boxes = boxes.clone()
    boxes[..., [0, 2]] -= pad_w
    boxes[..., [1, 3]] -= pad_h
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


_YOLOV5_MODEL_CACHE: dict[str, tuple[Any, torch.device, int, bool]] = {}


def _select_torch_device(device: str | None) -> torch.device:
    d = (device or "").strip().lower()
    if d in ("", "auto"):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if d == "cpu":
        return torch.device("cpu")
    if d == "mps":
        return torch.device("mps")
    if d.isdigit():
        return torch.device(f"cuda:{int(d)}") if torch.cuda.is_available() else torch.device("cpu")
    if d.startswith("cuda"):
        return torch.device((device or "").strip()) if torch.cuda.is_available() else torch.device("cpu")
    return torch.device(device or "cpu")


def load_yolov5(weights: Path, device: torch.device) -> tuple[Any, int, bool]:
    """Load fused eval model; returns (model, stride, use_half)."""
    key = f"{weights.resolve()}|{device}"
    if key in _YOLOV5_MODEL_CACHE:
        return _YOLOV5_MODEL_CACHE[key]

    old = os.getcwd()
    os.chdir(str(Y5_ROOT))
    try:
        from models.experimental import attempt_load

        model = attempt_load(str(weights), device=device, fuse=True)
        model.eval()
        stride = int(model.stride.max()) if hasattr(model, "stride") else 32
        half = device.type == "cuda"
        if half:
            model.half()
        _YOLOV5_MODEL_CACHE[key] = (model, stride, half)
        return model, stride, half
    finally:
        os.chdir(old)


def _forward_letterboxed_bgr(
    model: Any,
    im_bgr: "NDArray[np.uint8]",
    device: torch.device,
    imgsz: int,
    stride: int,
    half: bool,
) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int]]:
    """Letterbox BGR uint8 → tensor (1,3,H,W), forward. Returns pred[0], letterbox (h,w), crop (h,w)."""
    h0, w0 = im_bgr.shape[:2]
    img, _, _ = letterbox(im_bgr, (imgsz, imgsz), auto=True, stride=stride)
    lb_h, lb_w = img.shape[0], img.shape[1]
    im = np.ascontiguousarray(img.transpose((2, 0, 1)))
    t = torch.from_numpy(im).to(device)
    t = t.half() if half else t.float()
    t /= 255.0
    if t.ndim == 3:
        t = t.unsqueeze(0)
    with torch.no_grad():
        out = model(t, augment=False)
        pred = out[0] if isinstance(out, (list, tuple)) else out
    return pred, (lb_h, lb_w), (h0, w0)


def infer_image_dets_bgr(
    model: Any,
    im_bgr: "NDArray[np.uint8]",
    device: torch.device,
    *,
    imgsz: int,
    stride: int,
    half: bool,
    conf_thres: float,
    nms_iou: float,
    top_n_by_p_obj: int,
    max_det: int,
) -> list[tuple[tuple[float, float, float, float], float, int]]:
    """One letterboxed forward + decode; boxes in **original** ``im_bgr`` pixel space (xyxy)."""
    pred, lb_hw, crop_hw = _forward_letterboxed_bgr(model, im_bgr, device, imgsz, stride, half)
    xyxy, conf, cls_idx = decode_yolov5_predictions(
        pred,
        conf_thres=conf_thres,
        top_n_by_p_obj=top_n_by_p_obj,
        nms_iou=nms_iou,
        max_det=max_det,
    )
    if xyxy.numel() == 0:
        return []
    xyxy = scale_boxes(lb_hw, xyxy, crop_hw)
    out: list[tuple[tuple[float, float, float, float], float, int]] = []
    for i in range(xyxy.shape[0]):
        x1, y1, x2, y2 = (float(v) for v in xyxy[i].tolist())
        out.append(((x1, y1, x2, y2), float(conf[i].item()), int(cls_idx[i].item())))
    return out


def infer_tiled_to_global_xyxy(
    weights: Path,
    image_path: Path,
    *,
    conf: float,
    imgsz: int,
    device: str | None,
    tile_overlap: float,
    nms_iou: float,
    full_image_size: tuple[int, int],
    top_n_by_p_obj: int = 4096,
    max_det: int = 300,
) -> tuple[list[tuple[tuple[float, float, float, float], float, int]], float, int]:
    """Sliding-window inference on a full image; boxes in **full image** pixel coords (xyxy)."""
    from PIL import Image, ImageOps

    dev = _select_torch_device(device)
    model, stride, half = load_yolov5(weights, dev)
    iw, ih = full_image_size

    tiles = _iter_tile_boxes(iw, ih, imgsz, tile_overlap)
    raw: list[tuple[tuple[float, float, float, float], float, int]] = []
    t0 = time.perf_counter()
    with Image.open(image_path) as im_full:
        im_full = ImageOps.exif_transpose(im_full.convert("RGB"))
        for (x0, y0, x1, y1) in tiles:
            crop = im_full.crop((x0, y0, x1, y1))
            arr = np.asarray(crop, dtype=np.uint8)
            im_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            dets = infer_image_dets_bgr(
                model,
                im_bgr,
                dev,
                imgsz=imgsz,
                stride=stride,
                half=half,
                conf_thres=conf,
                nms_iou=nms_iou,
                top_n_by_p_obj=top_n_by_p_obj,
                max_det=max_det,
            )
            for (xyxy, sc, ci) in dets:
                px1, py1, px2, py2 = xyxy
                px1 += x0
                py1 += y0
                px2 += x0
                py2 += y0
                px1 = max(0.0, min(px1, float(iw)))
                px2 = max(0.0, min(px2, float(iw)))
                py1 = max(0.0, min(py1, float(ih)))
                py2 = max(0.0, min(py2, float(ih)))
                if px2 <= px1 or py2 <= py1:
                    continue
                raw.append(((px1, py1, px2, py2), sc, ci))
    elapsed = time.perf_counter() - t0
    return raw, elapsed, len(tiles)


def main() -> int:
    p = argparse.ArgumentParser(description="YOLOv5 .pt detect: top-k by p_obj + NMS (DWMB helper).")
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--top-n", type=int, default=4096, help="Max candidates ranked by p_obj before NMS")
    p.add_argument("--max-det", type=int, default=300)
    args = p.parse_args()

    dev = _select_torch_device(args.device)
    model, stride, half = load_yolov5(args.weights.resolve(), dev)
    im_bgr = cv2.imread(str(args.source))
    if im_bgr is None:
        print(f"failed to read {args.source}", file=sys.stderr)
        return 1
    dets = infer_image_dets_bgr(
        model,
        im_bgr,
        dev,
        imgsz=args.imgsz,
        stride=stride,
        half=half,
        conf_thres=args.conf,
        nms_iou=args.iou,
        top_n_by_p_obj=args.top_n,
        max_det=args.max_det,
    )
    # Sort print order by score descending
    dets_sorted = sorted(dets, key=lambda t: t[1], reverse=True)
    for i, (xyxy, sc, ci) in enumerate(dets_sorted):
        print(f"{i:4d}  cls={ci}  conf={sc:.4f}  xyxy={xyxy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
