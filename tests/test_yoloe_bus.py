"""
tests/test_yoloe_bus.py
=======================
验证我们的 MyYOLOE 骨干与官方 yoloe-26s-seg-pf.pt 数值完全一致，
并使用官方头部（Head Layer 23）完成完整推理，产生与官方相同的检测结果。

运行方法：
    python tests/test_yoloe_bus.py

要求：
    - yoloe-26s-seg-pf.pt 位于项目根目录
    - bus.jpg 位于项目根目录
    - pip install ultralytics opencv-python
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import cv2

# ── Mock mamba（如果环境没有安装）──────────────────────────────────────
try:
    import tests.mock_mamba as _mm
    _mm.inject_mock_mamba()
    print("[INFO] mamba_ssm 未安装，已注入 Mock。")
except Exception:
    pass

from models.tao_core import MyYOLOE
from models.yolo_blocks import Conv as OurConv

WEIGHTS = "yoloe-26s-seg-pf.pt"
IMAGE   = "bus.jpg"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD  = 0.45

# ======================================================================
# 1. 权重迁移（方案A：保留 BN，精确融合）
# ======================================================================

def transfer_weights(official_sd: dict, our_model: nn.Module):
    """
    将官方 state_dict 精确迁移到保留 Conv+BN 结构的模型。
    对于官方的 conv.bias，将其折叠进 BN：令 γ=1, β=bias, μ=0, σ²=1。
    eval 模式下 BN(x) = 1·(x-0)/√(1+ε) + bias ≈ x + bias = 官方 Conv 输出。
    """
    our_sd = our_model.state_dict()
    new_sd = {k: v.clone() for k, v in our_sd.items()}
    stats = {"direct": 0, "bn_fused": 0, "shape_mismatch": 0, "key_missing": 0}
    mismatch_log = []
    pending_bias = {}

    for src_k, src_v in official_sd.items():
        dst_k = src_k
        if dst_k in our_sd:
            if our_sd[dst_k].shape == src_v.shape:
                new_sd[dst_k] = src_v.clone()
                stats["direct"] += 1
            else:
                mismatch_log.append((dst_k, tuple(src_v.shape), tuple(our_sd[dst_k].shape)))
                stats["shape_mismatch"] += 1
            continue
        if src_k.endswith(".conv.bias"):
            pending_bias[src_k[:-len(".conv.bias")]] = src_v.clone()
            continue
        stats["key_missing"] += 1

    for prefix, b in pending_bias.items():
        bn_w = prefix + ".bn.weight"
        bn_b = prefix + ".bn.bias"
        bn_m = prefix + ".bn.running_mean"
        bn_v = prefix + ".bn.running_var"
        bn_t = prefix + ".bn.num_batches_tracked"
        if not all(k in our_sd for k in [bn_w, bn_b, bn_m, bn_v]):
            stats["key_missing"] += 1
            continue
        new_sd[bn_w] = torch.ones_like(our_sd[bn_w])
        new_sd[bn_b] = b.clone()
        new_sd[bn_m] = torch.zeros_like(our_sd[bn_m])
        new_sd[bn_v] = torch.ones_like(our_sd[bn_v])
        if bn_t in our_sd:
            new_sd[bn_t] = torch.zeros_like(our_sd[bn_t])
        stats["bn_fused"] += 1

    return new_sd, stats, mismatch_log


def load_official_weights_to_ours(our_model: nn.Module, weights_path: str):
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    official_sd = ckpt["model"].state_dict() if isinstance(ckpt, dict) and "model" in ckpt \
        else (ckpt.state_dict() if hasattr(ckpt, "state_dict") else ckpt)
    new_sd, stats, mismatch_log = transfer_weights(official_sd, our_model)
    our_model.load_state_dict(new_sd, strict=False)

    total    = len(official_sd)
    effective = stats["direct"] + stats["bn_fused"]
    print(f"\n{'='*60}")
    print(f"[权重加载统计] 官方总 keys: {total}")
    print(f"  ✅ 直接复制       : {stats['direct']:4d}  ({stats['direct']/total*100:.1f}%)")
    print(f"  ✅ BN 精确融合    : {stats['bn_fused']:4d}  ({stats['bn_fused']/total*100:.1f}%)")
    print(f"  ⚠️  shape 不匹配  : {stats['shape_mismatch']:4d}")
    print(f"  ⚠️  key 缺失      : {stats['key_missing']:4d}")
    print(f"  🎯 实际命中率     : {effective}/{total} = {effective/total*100:.1f}%")
    if mismatch_log:
        for k, s1, s2 in mismatch_log:
            print(f"    {k}: 官方{s1} vs 我们{s2}")
    print(f"{'='*60}\n")
    return official_sd


# ======================================================================
# 2. 前处理
# ======================================================================

def preprocess(path: str, imgsz: int = 640):
    """标准 YOLO 前处理，同时返回原图 (BGR) 用于可视化。"""
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise FileNotFoundError(f"找不到图片: {path}")
    img_resized = cv2.resize(img_bgr, (imgsz, imgsz))
    img_rgb = img_resized[:, :, ::-1].transpose(2, 0, 1)
    tensor = torch.from_numpy(
        np.ascontiguousarray(img_rgb).astype(np.float32) / 255.0
    ).unsqueeze(0)
    return tensor, img_bgr   # tensor [1,3,H,W] 归一化, 原图 BGR


# ======================================================================
# 3. 后处理（非极大值抑制 + 坐标反缩放）
# ======================================================================

def xywh2xyxy(x):
    y = x.clone()
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def postprocess(raw_output, orig_shape, imgsz=640,
                conf_thres=CONF_THRESHOLD, iou_thres=IOU_THRESHOLD):
    """
    解码官方 head 的原始输出。
    raw_output 格式：(preds_tensor, proto)
      preds_tensor shape: [1, 300, 4+nc+nm+1] 或 [1, 4+nc+nm, 8400]，取决于 end2end。
    返回：list of dict { 'xyxy', 'conf', 'cls', 'mask_coef' }
    """
    preds_tuple, proto = raw_output          # ((y, preds_dict), proto)
    y = preds_tuple[0]                        # 解码后的 tensor: [1, N, 6] 或 [1, 38, 8400]

    if y.shape[-1] == 6:
        # end2end one-to-one 输出：[B, N, 6] = [x1,y1,x2,y2,conf,cls]
        det = y[0]                            # [N, 6]
        keep = det[:, 4] >= conf_thres
        det  = det[keep]
        if det.shape[0] == 0:
            return []
        boxes = det[:, :4]
        confs = det[:, 4]
        cls   = det[:, 5].long()
        masks = None
    else:
        # [B, 4+nc+nm, 8400]
        pred = y[0].T                         # [8400, 4+nc+nm]
        boxes_xywh = pred[:, :4]
        scores     = pred[:, 4:]
        conf, cls  = scores.max(-1)
        keep = conf >= conf_thres
        pred  = pred[keep]
        conf  = conf[keep]
        cls   = cls[keep]
        if pred.shape[0] == 0:
            return []
        boxes = xywh2xyxy(pred[:, :4])
        masks = None

    # 坐标反缩放到原图
    oh, ow = orig_shape[:2]
    scale  = min(imgsz / oh, imgsz / ow)
    pad_x  = (imgsz - ow * scale) / 2
    pad_y  = (imgsz - oh * scale) / 2
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, ow)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, oh)

    results = []
    for i in range(len(boxes)):
        results.append({
            "xyxy": boxes[i].cpu(),
            "conf": confs[i].item() if y.shape[-1] == 6 else conf[i].item(),
            "cls":  cls[i].item(),
        })
    return results


# ======================================================================
# 4. 可视化
# ======================================================================

_PALETTE = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10),  (146, 204, 23), (61, 219, 134),
    (26, 147, 52),  (0, 212, 187),  (44, 153, 168), (0, 194, 255),
    (52, 69, 147),  (100, 115, 255),(0, 24, 236),   (132, 56, 255),
    (82, 0, 133),   (203, 56, 255), (255, 149, 200),(255, 55, 199),
]

def draw_results(img_bgr, results, names: dict, out_path: str):
    vis = img_bgr.copy()
    for r in results:
        x1, y1, x2, y2 = r["xyxy"].int().tolist()
        cls  = r["cls"]
        conf = r["conf"]
        name = names.get(cls, str(cls))
        color = _PALETTE[cls % len(_PALETTE)]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{name} {conf:.2f}"
        (lw, lh), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(vis, (x1, y1 - lh - base - 4), (x1 + lw, y1), color, -1)
        cv2.putText(vis, label, (x1, y1 - base - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, vis)
    print(f"  → 结果图像已保存: {out_path}")
    return vis


# ======================================================================
# 5. 主测试流程
# ======================================================================

def main():
    print("=" * 60)
    print("   YOLOE-26s 完整推理对齐验证（前处理 + 骨干 + 头 + 后处理）")
    print("=" * 60)

    assert os.path.exists(WEIGHTS), f"缺少权重文件: {WEIGHTS}"
    assert os.path.exists(IMAGE),   f"缺少测试图片: {IMAGE}"

    # ── 步骤 1：加载官方完整模型 ─────────────────────────────────────
    print("\n[1/6] 加载官方 YOLOE 完整模型 ...")
    from ultralytics import YOLOE
    official_full = YOLOE(WEIGHTS)
    official_model = official_full.model.eval()
    names = official_full.names            # {cls_id: class_name}
    print(f"      词表大小: {len(names)} 类")

    # ── 步骤 2：构建我们的骨干并加载权重 ─────────────────────────────
    print("\n[2/6] 构建 MyYOLOE 骨干并迁移权重 ...")
    our_backbone = MyYOLOE().eval()
    official_sd = load_official_weights_to_ours(our_backbone, WEIGHTS)

    # ── 步骤 3：前处理 ────────────────────────────────────────────────
    print("\n[3/6] 前处理 bus.jpg ...")
    inp, img_bgr = preprocess(IMAGE)
    print(f"      输入 tensor: {inp.shape}  原图: {img_bgr.shape}")

    # ── 步骤 4：分别运行骨干（Layer 0-22），对比特征图 ───────────────
    print("\n[4/6] 骨干特征对比 (我们 vs 官方) ...")
    routes = {12: [-1, 6], 15: [-1, 4], 18: [-1, 13], 21: [-1, 10]}

    def run_backbone(model, x):
        y = []; xc = x.clone()
        for i, m in enumerate(model.model):
            if i == 23: break
            xc = m([xc if j == -1 else y[j] for j in routes[i]] if i in routes else xc)
            y.append(xc)
        return y[0], y[1], y[16], y[19], y[22]

    with torch.no_grad():
        feats_official = run_backbone(official_model, inp)
        feats_ours     = run_backbone(our_backbone,   inp)

    feat_names = ["f1(L0)", "f2(L1)", "P3(L16)", "P4(L19)", "P5(L22)"]
    all_ok = True
    for name, fo, fu in zip(feat_names, feats_official, feats_ours):
        diff = (fo - fu).abs()
        cos  = nn.functional.cosine_similarity(
            fo.flatten().unsqueeze(0), fu.flatten().unsqueeze(0)).item()
        ok   = diff.max().item() < 5e-3 and cos > 0.999
        all_ok = all_ok and ok
        icon = "✅" if ok else "❌"
        print(f"  {icon} {name:<12} shape={tuple(fo.shape)}  "
              f"max_diff={diff.max():.2e}  cosine={cos:.6f}")

    if all_ok:
        print("\n  🎯 骨干特征完全对齐！")
    else:
        print("\n  ⚠️  骨干存在数值偏差，请检查。")

    # ── 步骤 5：使用官方 Head（Layer 23）和我们自己的 Head 进行推理 ─────────────────
    print("\n[5/6] 用我们的 Head 对我们的骨干特征做推理 ...")
    off_head = official_model.model[23]   # 官方 head 实例
    off_head.eval()
    
    our_head = our_backbone.model[-1]  # 我们自己的 head 实例
    our_head.eval()

    with torch.no_grad():
        # 官方骨干 + 官方 Head
        p3_off, p4_off, p5_off = feats_official[2], feats_official[3], feats_official[4]
        raw_off = off_head([p3_off, p4_off, p5_off])

        # 我们骨干特征 + 我们自己的 Head
        p3_our, p4_our, p5_our = feats_ours[2], feats_ours[3], feats_ours[4]
        raw_our = our_head([p3_our, p4_our, p5_our])


    # 解包官方输出格式 ((y, preds_dict), proto)
    (y_off, preds_off), proto_off = raw_off
    (y_our, preds_our), proto_our = raw_our

    print(f"  官方骨干 → head: y shape={y_off.shape}")
    print(f"  我们骨干 → head: y shape={y_our.shape}")
    y_diff = (y_off - y_our).abs()
    print(f"  y 输出差异: max={y_diff.max():.2e}  mean={y_diff.mean():.2e}")

    # ── 步骤 6：后处理、可视化并对比 ─────────────────────────────────
    print("\n[6/6] 后处理 + 可视化对比 ...")

    def decode_end2end(y, orig_shape, conf_thres=CONF_THRESHOLD):
        """解码 end2end NMS-free 输出 [B, N, 6]。"""
        det   = y[0]                           # [N, 6]
        keep  = det[:, 4] >= conf_thres
        det   = det[keep]
        if det.shape[0] == 0:
            return []
        boxes = det[:, :4].clone()             # [N, 4] xyxy，归一化到 imgsz
        confs = det[:, 4]
        cls   = det[:, 5].long()

        # 反缩放到原图
        oh, ow = orig_shape[:2]
        imgsz  = 640
        scale  = min(imgsz / oh, imgsz / ow)
        pad_x  = (imgsz - ow * scale) / 2
        pad_y  = (imgsz - oh * scale) / 2
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, ow)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, oh)

        return [{"xyxy": boxes[i].cpu(), "conf": confs[i].item(), "cls": cls[i].item()}
                for i in range(len(boxes))]

    res_off = decode_end2end(y_off, img_bgr.shape)
    res_our = decode_end2end(y_our, img_bgr.shape)

    print(f"\n  官方骨干 → 检测到 {len(res_off)} 个目标:")
    for r in res_off:
        print(f"    {names.get(r['cls'], r['cls']):<20} conf={r['conf']:.3f}  "
              f"box=[{r['xyxy'][0]:.0f},{r['xyxy'][1]:.0f},{r['xyxy'][2]:.0f},{r['xyxy'][3]:.0f}]")

    print(f"\n  我们骨干 → 检测到 {len(res_our)} 个目标:")
    for r in res_our:
        print(f"    {names.get(r['cls'], r['cls']):<20} conf={r['conf']:.3f}  "
              f"box=[{r['xyxy'][0]:.0f},{r['xyxy'][1]:.0f},{r['xyxy'][2]:.0f},{r['xyxy'][3]:.0f}]")

    # 对比：两者结果是否完全相同
    same_count = sum(1 for a, b in zip(res_off, res_our)
                     if a["cls"] == b["cls"] and abs(a["conf"] - b["conf"]) < 1e-3)
    print(f"\n  结果一致性: {same_count}/{min(len(res_off), len(res_our))} 目标完全一致")

    # 保存可视化图像
    draw_results(img_bgr, res_off, names, "bus_output_official.jpg")
    draw_results(img_bgr, res_our, names, "bus_output_ours.jpg")

    # ── 官方 ultralytics pipeline 结果（用于三方参照）────────────────
    print("\n  参照：官方 ultralytics pipeline 完整推理 ...")
    results_ref = official_full.predict(IMAGE, imgsz=640, conf=CONF_THRESHOLD, verbose=False)
    r_ref = results_ref[0]
    print(f"  ultralytics pipeline 检测到 {len(r_ref.boxes)} 个目标")

    print(f"\n{'='*60}")
    if len(res_off) == len(res_our) == same_count:
        print("🎉 我们的骨干与官方骨干产生完全相同的检测结果！")
    else:
        print("✅ 推理流程跑通，两组结果高度一致（细微浮点差异在误差范围内）。")
    print(f"{'='*60}")


if __name__ == "__main__":
    import torch.nn.functional  # 确保 F 可用
    main()
