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
    def __init__(self, args, model, prefetcher):
        self.args = args
        self.device = torch.device(args.device)
        self.model = model.to(self.device)
        self.prefetcher = prefetcher
        self.wandb = wandb if (wandb is not None and getattr(wandb, "run", None) is not None) else None
        self.loss_ema = {}

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
        """智能加载 YOLOE 预训练权重。

        策略：
        - 官方 Conv 使用 bias=True 且无 BN，我们使用 bias=False + BN。
        - 对于 conv.weight：直接复制（shape 完全相同）。
        - 对于 conv.bias：将其加到对应的 bn.bias 中，等效吸收偏移量，
          保持 BN 结构不变，对训练稳定性友好。
        - 对于 lrpc.2.vocab（我们改为 Conv2d，官方也是 Conv2d）：直接复制。
        - 对于 proto.semseg.2（官方 80 类，我们 80 类）：直接复制。
        - 跳过 shape 不匹配的 key，不破坏任何结构。
        """
        is_training = self.model.training
        self.model.eval()

        if not os.path.exists(self.args.yolo_weights):
            print(f"正在从 Ultralytics 下载 {self.args.yolo_weights}...")
            urllib.request.urlretrieve(
                f"https://github.com/ultralytics/assets/releases/download/v8.4.0/{self.args.yolo_weights}",
                self.args.yolo_weights)
            print("下载完成。")

        ckpt = torch.load(self.args.yolo_weights, map_location="cpu", weights_only=False)
        sd_src = ckpt["model"].state_dict() if isinstance(ckpt, dict) and "model" in ckpt else (
            ckpt.state_dict() if hasattr(ckpt, "state_dict") else ckpt)

        tgt = self.model.state_dict()

        # 官方 key 前缀 "model." → 我们的 "segmenter.model."
        def remap(k):
            if k.startswith("model."):
                return "segmenter." + k
            return k

        direct_loaded = 0      # 直接 shape 匹配并复制
        bias_absorbed = 0      # bias 被吸收到 bn.bias
        shape_skipped = 0      # shape 不匹配，跳过
        key_skipped = 0        # key 在我们模型中不存在

        new_sd = dict(tgt)  # 从我们的 state_dict 出发（保留未命中参数的随机初始化）

        for src_k, src_v in sd_src.items():
            dst_k = remap(src_k)

            if dst_k not in tgt:
                # 官方有，我们没有对应 key
                # 检查是否是 conv.bias，对应我们的 conv.weight 同级 bn.bias
                if src_k.endswith(".conv.bias"):
                    # 尝试将 bias 吸收到对应的 bn.bias
                    bn_key = remap(src_k.replace(".conv.bias", ".bn.bias"))
                    if bn_key in tgt and tgt[bn_key].shape == src_v.shape:
                        new_sd[bn_key] = tgt[bn_key] + src_v.to(tgt[bn_key].device)  # 加法吸收偏移
                        bias_absorbed += 1
                    else:
                        key_skipped += 1
                else:
                    key_skipped += 1
                continue

            if tgt[dst_k].shape != src_v.shape:
                shape_skipped += 1
                continue

            new_sd[dst_k] = src_v
            direct_loaded += 1

        self.model.load_state_dict(new_sd, strict=False)

        total_src = len(sd_src)
        print(f"\n{'='*55}")
        print(f"[YOLO 权重加载统计]")
        print(f"  官方总参数块数 : {total_src}")
        print(f"  ✅ 直接复制    : {direct_loaded} ({direct_loaded/total_src*100:.1f}%)")
        print(f"  ✅ bias→BN 吸收: {bias_absorbed} ({bias_absorbed/total_src*100:.1f}%)")
        print(f"  ⚠️ shape 不匹配: {shape_skipped}")
        print(f"  ⚠️ key 不存在  : {key_skipped}")
        effective = direct_loaded + bias_absorbed
        print(f"  🎯 实际命中率  : {effective}/{total_src} = {effective/total_src*100:.1f}%")
        print(f"{'='*55}\n")
        self.model.train(is_training)

    def _setup_finetune(self):
        for param in self.model.segmenter.parameters():
            param.requires_grad = False

        trainable_modules = [
            self.model.geom_decoder, self.model.se3_physics_head, self.model.st_block,
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
        from tqdm.auto import tqdm
        loss_sum = torch.tensor(0.0, device=self.device)
        pbar = tqdm(
            range(self.args.steps_per_epoch),
            desc=f"Epoch {epoch}/{self.args.epochs}",
            leave=True
        )
        self.pbar = pbar
        for _ in pbar:
            batch = self.prefetcher.next()
            if batch is None:
                continue

            loss_sum = loss_sum + self._train_chunk(batch)

            # 彻底取消了所有硬设置 step / 渐进式参数解冻的调度机制，参数状态保持纯净静态控制
            pass

        self.pbar = None
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

        return tgt

    def _train_chunk(self, batch):
        v_seq, t_max = batch["video"], batch["video"].shape[1]
        total_loss_tensor = torch.tensor(0.0, device=self.device)

        loss_acc = {k: 0.0 for k in [
            "Obj", "Box", "Mask", "Depth", "Photo", "Ego", "Flow", "Anom", "Gate", "Cls",
            "Attr", "Track", "FlowEPEpx", "DepthAbsRel", "DepthRMSElog", "DepthDelta1"
        ]}
        total_frames = 0
        
        # 跨 Chunk 追踪状态持久化记忆
        prev_queries = None

        for c_start in range(0, t_max, self.args.seq_len):
            if self.global_step == 0:
                print(f"\n[🚀 第一次前向传播] 正在触发 cuDNN 底层算子 Benchmark 穷举搜索，寻找最优卷积算法...", flush=True)
                cudnn_start = time.time()

            c_end = min(c_start + self.args.seq_len, t_max)
            c_vids = v_seq[:, c_start:c_end]
            T_chunk = c_end - c_start
            total_frames += T_chunk
            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=self.device.type, enabled=(self.scaler is not None)):
                # 彻底去除了基于 global_step 的骨干梯度限制，根据是否显式冻结静态决定
                allow_backbone_grad = self.mode == "supervised" and not self.args.freeze
                with contextlib.nullcontext() if allow_backbone_grad else torch.no_grad():
                    extracted = self.model.extract_features(
                        c_vids.reshape(-1, *c_vids.shape[2:]))
                    feats = [f.view(v_seq.shape[0], T_chunk, *f.shape[1:])
                             for f in extracted]

                dt = torch.full(
                    (v_seq.shape[0], T_chunk), 1.0 / 24.0, device=self.device)
                tgts = self._extract_target_chunk(batch, c_start, c_end, t_max)

                # 彻底去除了基于 global_step 的分类标记覆盖掩码逻辑，由分类损失权重（已固定设为 0.0）统一过滤，保持干净的端到端几何表示
                pass

                K, K_inv = None, None
                if "camera_focal_length" in tgts:
                    from utils.geometry import generate_intrinsics
                    K, K_inv = generate_intrinsics(
                        c_vids.shape[-2], c_vids.shape[-1], self.device,
                        focal_length=tgts.get("camera_focal_length"),
                        sensor_width=tgts.get("camera_sensor_width"),
                        dtype=torch.float32
                    )

                # 将上一个 Chunk 遗留的查询状态以及内参送入前向传播
                preds = self.model.forward_physics(
                    *feats, dt, self.global_step, get_loss_weights, c_vids.shape[-2:], tgts=tgts,
                    K=K, K_inv=K_inv, prev_queries=prev_queries)
                    
                prev_queries = preds.get("next_queries", None)

                img_next = self._build_next_frames(v_seq, c_start, c_end, t_max)

                # 步骤 4：显式向 compute_physics_loss 传入实例 EMA 状态
                loss, l_dict, w_img = compute_physics_loss(preds, tgts, c_vids.flatten(
                    0, 1), img_next.flatten(0, 1), self.mode, self.global_step, ema_state=self.loss_ema)

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

            if self.global_step == 0:
                print(f"[✅ Benchmark 完成] 耗时: {time.time() - cudnn_start:.1f}s。网络已进入极速计算模式。\n", flush=True)

            # 步骤 2：使用提取出的外部可视化辅助方法
            self._maybe_visualize(c_vids, tgts, preds, w_img, T_chunk)

            self.global_step += 1
            if self.global_step % 10 == 0:
                metric_str = f"Loss:{loss.item():.4f} | " + " ".join(
                    [f"{k}:{loss_acc[k]/total_frames:.2f}" for k in ["Obj", "Box", "Mask", "Depth", "Ego", "Flow", "Anom", "Attr", "Track", "FlowEPEpx", "DepthAbsRel"]]
                )
                if hasattr(self, "pbar") and self.pbar is not None:
                    self.pbar.set_postfix_str(metric_str)
                else:
                    print(f"[{time.time()-self.start_time:.1f}s] 步数 {self.global_step} | {metric_str}")
                if self.wandb:
                    log_dict = {
                        f"Loss/{k}": loss_acc[k]/total_frames for k in loss_acc}
                    log_dict.update(
                        {"Loss/Total": loss.item(), "Step": self.global_step})
                    self.wandb.log(log_dict, step=self.global_step)

        return total_loss_tensor

    def _slice_frame(self, v, B, T_chunk):
        if v is None:
            return None
        if isinstance(v, dict):
            return {k: self._slice_frame(val, B, T_chunk) for k, val in v.items()}
        if isinstance(v, list):
            res = []
            for x in v:
                if x.dim() == 0:
                    res.append(x)
                elif x.shape[0] == B * T_chunk:
                    res.append(
                        x[(B - 1) * T_chunk + 1: (B - 1) * T_chunk + 2])
                else:
                    res.append(x[-B:])
            return res
        if v.dim() == 0:
            return v
        if (
            v.dim() >= 4
            and v.shape[0] == B
            and v.shape[1] == T_chunk
        ):
            frame_id = min(1, T_chunk - 1)
            return v[-1:, frame_id]
        if v.shape[0] == B * T_chunk:
            return v[(B - 1) * T_chunk + 1: (B - 1) * T_chunk + 2]
        return v[-B:]

    def _build_next_frames(self, v_seq, c_start, c_end, t_max):
        c_vids = v_seq[:, c_start:c_end]
        img_next = torch.zeros_like(c_vids)
        for i, step in enumerate(range(c_start, c_end)):
            img_next[:, i] = v_seq[:, min(step+1, t_max-1)]
        return img_next

    def _maybe_visualize(self, c_vids, tgts, preds, w_img, T_chunk):
        if (self.global_step + 1) % self.args.vis_interval == 0:
            B = c_vids.shape[0]
            fp = save_visualization(
                c_vids[-1:, 1],
                {k: self._slice_frame(v, B, T_chunk) for k, v in tgts.items()},
                {k: self._slice_frame(v, B, T_chunk) for k, v in preds.items()},
                self.global_step + 1,
                self._slice_frame(w_img, B, T_chunk) if w_img is not None else None
            )
            if self.wandb and fp:
                self.wandb.log({"Vis": self.wandb.Image(fp)}, step=self.global_step)
