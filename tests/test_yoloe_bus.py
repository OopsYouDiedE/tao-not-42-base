
import torch.nn.functional as F
import torch.nn as nn
import sys
import os
import urllib.request
import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
if True:
    import tests.mock_mamba
    tests.mock_mamba.inject_mock_mamba()
    from models.tao_core import MyYOLOE


def test_yoloe_bus():
    print("====================================================")
    print("正在测试 YOLOE 全网络推理 (完美修复版)...")
    print("====================================================")

    # 初始化网络
    net = MyYOLOE()

    # 动态补全和对齐模型结构以匹配 checkpoint
    from models.yolo_blocks import Bottleneck, C3k2, C3k
    for i, m in enumerate(net.model):
        if isinstance(m, C3k2) and hasattr(m, 'm') and len(m.m) > 0 and isinstance(m.m[0], C3k):
            c3k = m.m[0]
            if len(c3k.m) == 1:
                c_ = c3k.m[0].cv1.conv.in_channels
                c2_out = c3k.m[0].cv2.conv.out_channels
                c3k.m.append(Bottleneck(c2_out, c2_out,
                             shortcut=True, g=1, k=(3, 3), e=1.0))

    # 调整模型以匹配融合权重 (带偏置的卷积，BN 作为恒等映射)
    # 【必需保留：因为官方 pt 是 Fused 权重，没有 BN 层】
    for name, module in net.named_modules():
        if module.__class__.__name__ == 'Conv':
            c1 = module.conv.in_channels
            c2 = module.conv.out_channels
            k = module.conv.kernel_size
            s = module.conv.stride
            p = module.conv.padding
            g = module.conv.groups
            d = module.conv.dilation

            new_conv = nn.Conv2d(c1, c2, k, s, p, groups=g,
                                 dilation=d, bias=True)

            # 替换子模块
            parts = name.split('.')
            if len(parts) > 1:
                parent = net
                for p_name in parts[:-1]:
                    parent = getattr(parent, p_name)
                module.conv = new_conv
                module.bn = nn.Identity()
            else:
                module.conv = new_conv
                module.bn = nn.Identity()

    # 剔除本代码中存在但检查点中不存在的稠密头和属性预测头
    head = net.model[23]
    del head.cv2
    del head.cv3
    del head.attr_heads

    # 替换 head.forward 以绕过已被删除的模块 (仅使用 one2one 模块)
    def new_forward(x_feats):
        proto_out, semseg = head.proto(x_feats)

        def process_branch(cv2_list, cv3_list, cv5_list):
            boxes, scores, mc, obj, cls = [], [], [], [], []
            for i, f in enumerate(x_feats):
                feat_box = cv2_list[i](f)
                feat_cls = cv3_list[i](f)
                feat_mask = cv5_list[i](f)

                bbox = F.softplus(head.lrpc[i].loc(feat_box)) + 1e-4
                boxes.append(bbox)

                gate_logits = head.lrpc[i].pf(feat_cls)
                obj.append(gate_logits)

                gate = torch.sigmoid(gate_logits)
                gated_cls = feat_cls * gate

                if isinstance(head.lrpc[i].vocab, nn.Linear):
                    cls_pred = head.lrpc[i].vocab(
                        gated_cls.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                else:
                    cls_pred = head.lrpc[i].vocab(gated_cls)

                scores.append(cls_pred)
                cls.append(cls_pred)
                mc.append(feat_mask)
            return obj, cls, boxes, mc

        obj_o2o, cls_o2o, boxes_o2o, mc_o2o = process_branch(
            head.one2one_cv2, head.one2one_cv3, head.one2one_cv5)

        return {
            "features": x_feats,
            "o2o_objectness": obj_o2o, "o2o_classification": cls_o2o, "o2o_boxes": boxes_o2o, "o2o_mask_coefficients": mc_o2o,
            "mask_prototypes": proto_out, "semseg": semseg
        }

    head.forward = new_forward

    # 加载权重
    ckpt_path = "yoloe-26s-seg-pf.pt"
    if not os.path.exists(ckpt_path):
        weights_url = f"https://github.com/ultralytics/assets/releases/download/v8.4.0/{ckpt_path}"
        urllib.request.urlretrieve(weights_url, ckpt_path)

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    class_names = {}
    if 'model' in ckpt and hasattr(ckpt['model'], 'names'):
        class_names = ckpt['model'].names
    elif 'names' in ckpt:
        class_names = ckpt['names']

    if isinstance(ckpt, dict) and 'model' in ckpt:
        model_obj = ckpt['model']
        state_dict = model_obj.state_dict() if hasattr(
            model_obj, 'state_dict') else model_obj
    else:
        state_dict = ckpt

    class DummyContainer(nn.Module):
        pass
    head.savpe = DummyContainer()
    head.reprta = DummyContainer()

    for k, v in state_dict.items():
        new_k = k.replace("model.model.", "model.") if k.startswith(
            "model.model.") else k
        if new_k.startswith("model.23.savpe.") or new_k.startswith("model.23.reprta."):
            sub_k = new_k.replace("model.23.", "")
            parts = sub_k.split('.')
            curr = head
            for part in parts[:-1]:
                if not hasattr(curr, part):
                    setattr(curr, part, DummyContainer())
                curr = getattr(curr, part)
            setattr(curr, parts[-1], nn.Parameter(torch.zeros_like(v)))

    # 【修复 2】：新增 Head 权重映射逻辑（彻底解决输出 technician 乱码现象）
    mapped_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace("model.model.", "model.") if k.startswith(
            "model.model.") else k
        mapped_state_dict[new_k] = v

    keys_to_add = {}
    for k, v in mapped_state_dict.items():
        if "model.23.cv2." in k:
            keys_to_add[k.replace(
                "model.23.cv2.", "model.23.one2one_cv2.")] = v
        elif "model.23.cv3." in k:
            keys_to_add[k.replace(
                "model.23.cv3.", "model.23.one2one_cv3.")] = v
        elif "model.23.cv5." in k:
            keys_to_add[k.replace(
                "model.23.cv5.", "model.23.one2one_cv5.")] = v

    mapped_state_dict.update(keys_to_add)

    loaded = {}
    skipped_shape = []
    unexpected = []
    model_state = net.state_dict()

    for k, v in mapped_state_dict.items():
        if k not in model_state:
            unexpected.append(k)
            continue
        if model_state[k].shape != v.shape:
            skipped_shape.append(
                (k, tuple(v.shape), tuple(model_state[k].shape)))
            continue
        loaded[k] = v

    missing = [k for k in model_state.keys() if k not in loaded]

    print("\n匹配状态：")
    print(f"  - 真正加载到的参数: {len(loaded)} (此时应该是七百多到八百左右)")
    print(f"  - 缺失参数: {len(missing)} (应该非常少)")

    net.load_state_dict(loaded, strict=False)
    net.eval()

    img_path = "bus.jpg"
    if not os.path.exists(img_path):
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg", img_path)

    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (640, 640))
    img_tensor = torch.from_numpy(img_resized).permute(
        2, 0, 1).float().unsqueeze(0) / 255.0

    with torch.no_grad():
        x = img_tensor
        y = []
        for i, m in enumerate(net.model):
            if i in net.routes:
                f = net.routes[i]
                x = m([x if j == -1 else y[j] for j in f]
                      if isinstance(f, list) else (y[f] if f != -1 else x))
            else:
                x = m(x)
            y.append(x)
        preds = x

    print("\n--- 预测成功 ---")

    from utils.visualization import extract_instances
    print("\n正在生成可视化结果并保存到 bus_output.jpg...")

    preds_for_vis = {
        "objectness": preds.get("o2o_objectness"),
        "classification": preds.get("o2o_classification"),
        "boxes": preds.get("o2o_boxes"),
        "mask_coefficients": preds.get("o2o_mask_coefficients"),
        "mask_prototypes": preds.get("mask_prototypes"),
    }

    insts = extract_instances(
        preds_for_vis, score_thresh=0.05, max_det=200, with_nms=False)

    if insts and insts[0]:
        inst = insts[0]
        H, W = img_resized.shape[:2]
        vis_img = img_resized.copy()
        masks = inst["masks"]

        print("\n--- Top Predictions ---")
        for i in range(len(inst["scores"])):
            box = inst["boxes"][i].cpu().numpy()
            score = inst["scores"][i].item()
            cls_id = int(inst["classes"][i].item()
                         ) if inst["classes"] is not None else 0
            tag_name = class_names.get(cls_id, str(
                cls_id)) if class_names else str(cls_id)

            if i < 20:
                print(
                    f"Rank {i+1}: Tag='{tag_name}' (ID={cls_id}), Score={score:.4f}")

            x1, y1, x2, y2 = (box * [W, H, W, H]).astype(int)

            if masks is not None and masks[i] is not None:
                mask = masks[i].cpu().numpy()
                color = np.array([0, 255, 0], dtype=np.uint8)
                vis_img[mask] = vis_img[mask] * 0.5 + color * 0.5

            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_img, f"{tag_name}: {score:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 【修复 3】：从 RGB 转回 BGR 用于 OpenCV 保存，修复阿凡达色
        final_bgr_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite("bus_output.jpg", final_bgr_img)
        print(f"成功保存带有 {len(inst['scores'])} 个检测结果的图像！")
    else:
        print("未检测到置信度达标的对象。")


if __name__ == "__main__":
    test_yoloe_bus()
