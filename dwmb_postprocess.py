"""Decode YOLOv5 Detect raw output: rank by objectness (p_obj), top-k, NMS.

Pure PyTorch + torchvision — no ONNX, no ``ultralytics.YOLO``, no ``utils.general`` (avoids ultralytics import chain).
"""

from __future__ import annotations

import torch
import torchvision

# xywh: center x, center y, width, height (pixel coords in letterboxed image)


def xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    y = x.clone()
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def decode_yolov5_predictions(
    pred: torch.Tensor,
    *,
    conf_thres: float,
    top_n_by_p_obj: int,
    nms_iou: float,
    max_det: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Turn YOLOv5 inference tensor into boxes (letterboxed image space).

    ``pred`` is ``(1, N, 5 + nc)`` or ``(N, 5 + nc)`` after the Detect head (xywh + p_obj + class logits
    already sigmoid-applied in the head).

    Candidates are sorted by **objectness** ``p_obj`` (index 4), then the top ``top_n_by_p_obj`` rows
    are kept before class-aware NMS. Per-detection **score** used for NMS and downstream matching is
    ``p_obj * p_cls`` (best class), same as YOLOv5 ``non_max_suppression``.

    Returns:
        xyxy (M, 4), score (M,), cls (M,) long
    """
    if pred.dim() == 3:
        pred = pred[0]
    if pred.numel() == 0 or pred.shape[0] == 0:
        z = pred.new_zeros((0, 4))
        return z, pred.new_zeros(0), pred.new_zeros(0, dtype=torch.long)

    nc = int(pred.shape[1]) - 5
    if nc < 1:
        raise ValueError(f"expected 5+nc outputs per anchor, got {pred.shape[1]}")

    xywh = pred[:, :4]
    p_obj = pred[:, 4]
    cls_scores = pred[:, 5 : 5 + nc]
    cls_prob, cls_idx = cls_scores.max(dim=1)
    conf = p_obj * cls_prob

    keep = conf >= conf_thres
    if not keep.any():
        z = pred.new_zeros((0, 4))
        return z, pred.new_zeros(0), pred.new_zeros(0, dtype=torch.long)

    xywh = xywh[keep]
    p_obj = p_obj[keep]
    conf = conf[keep]
    cls_idx = cls_idx[keep]

    # Sort by objectness (p_obj), keep top_n_by_p_obj before NMS
    k = min(int(top_n_by_p_obj), p_obj.shape[0])
    order = torch.argsort(p_obj, descending=True)[:k]
    xywh = xywh[order]
    p_obj = p_obj[order]
    conf = conf[order]
    cls_idx = cls_idx[order]

    boxes = xywh2xyxy(xywh)
    # Class-aware NMS (same idea as batched_nms with class indices)
    keep_idx = torchvision.ops.batched_nms(boxes, conf, cls_idx, float(nms_iou))
    keep_idx = keep_idx[: int(max_det)]
    return boxes[keep_idx], conf[keep_idx], cls_idx[keep_idx]
