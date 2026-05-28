import os
import time
import contextlib
import urllib.request

import torch
import torch.nn as nn

try:
    import wandb
except ImportError:
    wandb = None

from models.tao_core import *
from utils.losses import *
from utils.visualization import *

# 7. TAO 训练器 (OOP Refactoring)
# =====================================================================

class TAOTrainer:
    def __init__(self, args, model, buffer, prefetcher):
        self.args = args
        self.device = torch.device(args.device)
        self.model = model.to(self.device)
        self.buffer = buffer
        self.prefetcher = prefetcher
        self.wandb = wandb if (wandb is not None and getattr(wandb, "run", None) is not None) else None

        if self.args.yolo_weights:
            self._load_yolo_weights()
            if self.args.freeze:
                for param in self.model.segmenter.parameters():
                    param.requires_grad = False

        # [修复] 在初始化期间冻结词汇表，以保护零样本分类
        if hasattr(self.model.segmenter.model[-1], "lrpc"):
            for layer in self.model.segmenter.model[-1].lrpc:
                if hasattr(layer, "vocab"):
                    layer.vocab.requires_grad_(False)

        self.optimizer = torch.optim.AdamW(filter(
            lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        self.scaler = torch.amp.GradScaler(
            self.device.type) if self.device.type == "cuda" else None
        self.global_step, self.start_time, self.best_loss, self.epochs_no_improve = 0, time.time(), float("inf"), 0
        self.mode = "supervised"

    def _load_yolo_weights(self):
        if not os.path.exists(self.args.yolo_weights):
            print(f"正在从 Ultralytics 下载 {self.args.yolo_weights}...")
            urllib.request.urlretrieve(
                f"https://github.com/ultralytics/assets/releases/download/v8.4.0/{self.args.yolo_weights}", self.args.yolo_weights)
            print("下载完成。")

        for name, module in self.model.named_modules():
            if module.__class__.__name__ == 'Conv':
                c1, c2 = module.conv.in_channels, module.conv.out_channels
                k, s = module.conv.kernel_size, module.conv.stride
                p, g, d = module.conv.padding, module.conv.groups, module.conv.dilation

                new_conv = nn.Conv2d(
                    c1, c2, k, s, p, groups=g, dilation=d, bias=True)
                new_conv.to(module.conv.weight.device)
                module.conv = new_conv
                module.bn = nn.Identity()
            elif module.__class__.__name__ == 'PSABlock':
                if hasattr(module, 'add_norm1'):
                    module.add_norm1 = nn.Identity()
                if hasattr(module, 'add_norm2'):
                    module.add_norm2 = nn.Identity()

        ckpt = torch.load(self.args.yolo_weights,
                          map_location="cpu", weights_only=False)
        sd = ckpt["model"].state_dict() if isinstance(ckpt, dict) and "model" in ckpt else (
            ckpt.state_dict() if hasattr(ckpt, "state_dict") else ckpt)

        tgt = self.model.state_dict()

        def map_key(k):
            return k.replace("model.model.", "segmenter.model.").replace("model.", "segmenter.model.") if k.startswith("model.") else k

        loaded_keys = {k for k, v in sd.items() if map_key(
            k) in tgt and tgt[map_key(k)].shape == v.shape}
        print(f"[YOLO] 成功加载了 {len(loaded_keys)}/{len(sd)} 个键")
        tgt.update({map_key(k): v for k, v in sd.items() if k in loaded_keys})
        self.model.load_state_dict(tgt)

    def _setup_finetune(self):
        for param in self.model.segmenter.parameters():
            param.requires_grad = False

        trainable_modules = [
            self.model.geom_decoder, self.model.pose_head, self.model.st_block,
            self.model.st_block_p4, self.model.st_block_p5,
            self.model.feature_predictor, self.model.state_update_gate_head
        ]
        for m in trainable_modules:
            for p in m.parameters():
                p.requires_grad = True

        if hasattr(self.model.segmenter.model[-1], "obj_proj"):
            self.model.segmenter.model[-1].obj_proj.requires_grad_(True)
        if hasattr(self.model.segmenter.model[-1], "one2one_obj_proj"):
            self.model.segmenter.model[-1].one2one_obj_proj.requires_grad_(
                True)
        if hasattr(self.model.segmenter.model[-1], "class_prompts"):
            self.model.segmenter.model[-1].class_prompts.requires_grad_(True)

        # [FIX] 强制冻结 LRPCLayer 的 vocab（类别词典权重）
        # 保护 4585 维语义特征空间不被坍缩
        if hasattr(self.model.segmenter.model[-1], "lrpc"):
            for layer in self.model.segmenter.model[-1].lrpc:
                layer.vocab.requires_grad_(False)

        self.optimizer = torch.optim.AdamW(filter(
            lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr * 0.1)

    def train(self):
        self.model.train()
        for epoch in range(1, self.args.epochs + 1):
            if self.args.finetune_after_epoch and epoch > self.args.finetune_after_epoch and self.mode == "supervised":
                self.mode = "self_supervised"
                self._setup_finetune()

            epoch_loss = self._train_epoch(epoch)
            print(
                f"\n✅ 第 {epoch} 轮结束 | 平均损失: {epoch_loss:.4f} | 模式: {self.mode}")
            torch.save(self.model.state_dict(), self.args.checkpoint.replace(
                ".pth", f"_epoch_{epoch}.pth"))

            if epoch_loss < self.best_loss:
                self.best_loss, self.epochs_no_improve = epoch_loss, 0
                torch.save(self.model.state_dict(),
                           self.args.checkpoint.replace(".pth", "_best.pth"))
                print(f"🌟 已保存最佳模型 (损失: {self.best_loss:.4f})")
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.args.early_stop_patience:
                    print(f"\n🛑 已触发早停！")
                    break

    def _train_epoch(self, epoch):
        loss_sum = torch.tensor(0.0, device=self.device)
        for _ in range(self.args.steps_per_epoch):
            batch = self.prefetcher.next()
            if batch is None:
                continue

            loss_sum = loss_sum + self._train_chunk(batch)

            if self.global_step == 500 and self.mode == "supervised" and hasattr(self.model.segmenter.model[-1], "class_prompts"):
                self.model.segmenter.model[-1].class_prompts.requires_grad = True

            if self.mode == "supervised" and self.global_step in [self.args.unfreeze_step_1, self.args.unfreeze_step_2]:
                target_range = range(
                    20, 23) if self.global_step == self.args.unfreeze_step_1 else range(16, 20)
                for n, p in self.model.segmenter.named_parameters():
                    if any(f"model.{i}." in n for i in target_range):
                        p.requires_grad = True

        return loss_sum.item() / self.args.steps_per_epoch

    def _extract_target_chunk(self, batch, c_start, c_end, max_t):
        T = c_end - c_start
        B = batch["video"].shape[0]
        tgt = {}
        for k, v in batch.items():
            if k in ("video", "flow"):
                continue

            if k in ("camera_focal_length", "camera_sensor_width"):
                # [B] -> [B*T]
                tgt[k] = v.view(B, 1).expand(B, T).reshape(B * T)
            elif k == "is_dynamic":
                tgt[k] = v.unsqueeze(
                    1).expand(-1, T, -1).flatten(0, 1) if v is not None else None
            elif isinstance(v, list):
                tgt[k] = [x[:, c_start:c_end].flatten(0, 1) for x in v]
            else:
                tgt[k] = v[:, c_start:c_end].flatten(
                    0, 1) if v is not None else None

        flow = batch.get("flow")
        if flow is not None:
            flow_tgt = torch.zeros_like(flow[:, c_start:c_end])
            for i, step in enumerate(range(c_start, c_end)):
                flow_tgt[:, i] = flow[:, step] if step + \
                    1 < max_t else torch.zeros_like(flow[:, 0])
            tgt["flow_target"] = flow_tgt.flatten(0, 1)

        tgt["cam_pos_t"] = batch["cam_pos"][:, c_start:c_end].flatten(0, 1)
        tgt["cam_quat_t"] = batch["cam_quat"][:, c_start:c_end].flatten(0, 1)

        cam_pos_next = torch.zeros_like(batch["cam_pos"][:, c_start:c_end])
        cam_quat_next = torch.zeros_like(batch["cam_quat"][:, c_start:c_end])
        has_next = torch.zeros(B, T, device=self.device, dtype=torch.bool)

        for i, step in enumerate(range(c_start, c_end)):
            next_idx = step + 1 if step + 1 < max_t else step
            cam_pos_next[:, i] = batch["cam_pos"][:, next_idx]
            cam_quat_next[:, i] = batch["cam_quat"][:, next_idx]
            has_next[:, i] = step + 1 < max_t

        tgt["cam_pos_next"] = cam_pos_next.flatten(0, 1)
        tgt["cam_quat_next"] = cam_quat_next.flatten(0, 1)
        tgt["has_next"] = has_next.flatten(0, 1)

        if "cls_dense" in tgt:
            for i, step in enumerate(range(c_start, c_end)):
                if self.global_step < 1000 or step < 2:
                    if isinstance(tgt["cls_dense"], list):
                        for x in tgt["cls_dense"]:
                            x.view(B, T, *x.shape[1:])[:, i] = -100
                    else:
                        tgt["cls_dense"].view(
                            B, T, *tgt["cls_dense"].shape[1:])[:, i] = -100

        return tgt

    def _train_chunk(self, batch):
        v_seq, t_max = batch["video"], batch["video"].shape[1]
        total_loss_tensor = torch.tensor(0.0, device=self.device)

        loss_acc = {k: 0.0 for k in [
            "Obj", "Box", "Mask", "Depth", "Photo", "Ego", "Flow", "Anom", "Gate", "Cls",
            "Attr", "Track", "FlowEPEpx", "DepthAbsRel", "DepthRMSElog", "DepthDelta1"
        ]}
        total_frames = 0

        for c_start in range(0, t_max, self.args.seq_len):
            c_end = min(c_start + self.args.seq_len, t_max)
            c_vids = v_seq[:, c_start:c_end]
            T_chunk = c_end - c_start
            total_frames += T_chunk
            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=self.device.type, enabled=(self.scaler is not None)):
                allow_backbone_grad = self.mode == "supervised" and self.global_step >= self.args.unfreeze_step_1
                with contextlib.nullcontext() if allow_backbone_grad else torch.no_grad():
                    extracted = self.model.extract_features(
                        c_vids.reshape(-1, *c_vids.shape[2:]))
                    feats = [f.view(v_seq.shape[0], T_chunk, *f.shape[1:])
                             for f in extracted]

                dt = torch.full(
                    (v_seq.shape[0], T_chunk), 1.0 / 24.0, device=self.device)
                tgts = self._extract_target_chunk(batch, c_start, c_end, t_max)
                preds = self.model.forward_physics(
                    *feats, dt, self.global_step, get_loss_weights, c_vids.shape[-2:], tgts=tgts)

                img_next = torch.zeros_like(c_vids)
                for i, step in enumerate(range(c_start, c_end)):
                    img_next[:, i] = v_seq[:, min(step+1, t_max-1)]

                loss, l_dict, w_img = compute_physics_loss(preds, tgts, c_vids.flatten(
                    0, 1), img_next.flatten(0, 1), self.mode, self.global_step)

            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()

            total_loss_tensor = total_loss_tensor + loss.detach()
            for k in loss_acc:
                loss_acc[k] += l_dict.get(k, 0.0) * T_chunk

            if (self.global_step + 1) % self.args.vis_interval == 0:
                def slice_second_frame(v):
                    if v is None:
                        return None
                    if isinstance(v, list):
                        res = []
                        for x in v:
                            if x.dim() == 0:
                                res.append(x)
                            elif x.shape[0] == v_seq.shape[0] * T_chunk:
                                res.append(
                                    x[(v_seq.shape[0] - 1) * T_chunk + 1: (v_seq.shape[0] - 1) * T_chunk + 2])
                            else:
                                res.append(x[-v_seq.shape[0]:])
                        return res
                    if v.dim() == 0:
                        return v
                    if (
                        v.dim() >= 4
                        and v.shape[0] == v_seq.shape[0]
                        and v.shape[1] == T_chunk
                    ):
                        frame_id = min(1, T_chunk - 1)
                        return v[-1:, frame_id]
                    if v.shape[0] == v_seq.shape[0] * T_chunk:
                        return v[(v_seq.shape[0] - 1) * T_chunk + 1: (v_seq.shape[0] - 1) * T_chunk + 2]
                    return v[-v_seq.shape[0]:]

                fp = save_visualization(
                    c_vids[-1:, 1],
                    {k: slice_second_frame(v) for k, v in tgts.items()},
                    {k: slice_second_frame(v) for k, v in preds.items()},
                    self.global_step + 1,
                    slice_second_frame(w_img) if w_img is not None else None
                )
                if self.wandb and fp:
                    self.wandb.log({"Vis": self.wandb.Image(fp)}, step=self.global_step)

            self.global_step += 1
            if self.global_step % 10 == 0:
                print(f"[{time.time()-self.start_time:.1f}s] 步数 {self.global_step} | 总计:{loss.item():.4f} | " + " ".join(
                    [f"{k}:{loss_acc[k]/total_frames:.2f}" for k in ["Obj", "Box", "Mask", "Depth", "Ego", "Flow", "Anom", "Attr", "Track", "FlowEPEpx", "DepthAbsRel"]]))
                if self.wandb:
                    log_dict = {
                        f"Loss/{k}": loss_acc[k]/total_frames for k in loss_acc}
                    log_dict.update(
                        {"Loss/Total": loss.item(), "Step": self.global_step})
                    self.wandb.log(log_dict, step=self.global_step)

        return total_loss_tensor

# =====================================================================
