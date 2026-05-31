import torch
import torch.nn as nn
import torch.nn.functional as F

from models.yolo_blocks import *
from models.custom_heads import *
from models.yoloe_head import *

# =====================================================================

class YOLOEBackbone(nn.Module):
    """自定义 YOLOE 网络，完全对齐官方 yoloe-26s 结构。"""
    def __init__(self):
        super().__init__()
        # 定义模型层列表 (完全参照官方 YAML 配置与 s 缩放比)
        self.model = nn.ModuleList([
            Conv(3, 32, 3, 2),  # 0
            Conv(32, 64, 3, 2),  # 1
            C3k2(64, 128, n=1, shortcut=True, c3k=False, e=0.25),  # 2
            Conv(128, 128, 3, 2),  # 3
            C3k2(128, 256, n=1, shortcut=True, c3k=False, e=0.25),  # 4
            Conv(256, 256, 3, 2),  # 5
            C3k2(256, 256, n=1, shortcut=True, c3k=True, e=0.5),  # 6 (修复: shortcut=True)
            Conv(256, 512, 3, 2),  # 7
            C3k2(512, 512, n=1, shortcut=True, c3k=True, e=0.5),  # 8 (修复: shortcut=True)
            SPPF(512, 512, k=5, add=True),  # 9：官方有 add=True 残差连接
            C2PSA(512, 512, n=1, e=0.5),  # 10
            nn.Upsample(scale_factor=2.0, mode='nearest'),  # 11
            Concat(1),  # 12
            C3k2(768, 256, n=1, shortcut=True, c3k=True, e=0.5),  # 13 (修复: shortcut=True)
            nn.Upsample(scale_factor=2.0, mode='nearest'),  # 14
            Concat(1),  # 15
            C3k2(512, 128, n=1, shortcut=True, c3k=True, e=0.5),  # 16 (P3特征)
            Conv(128, 128, 3, 2),  # 17
            Concat(1),  # 18
            C3k2(384, 256, n=1, shortcut=True, c3k=True, e=0.5),  # 19 (P4特征)
            Conv(256, 256, 3, 2),  # 20
            Concat(1),  # 21
            C3k2(768, 512, n=1, shortcut=True, attn=True, e=0.5),  # 22 (P5特征，修复: attn=True)
            YOLOESegment26(nc=4585, nm=32, npr=128, embed=512, ch=(128, 256, 512))  # 23 (对齐 4585 类)
        ])

        # 定义路由连接
        self.routes = {12: [-1, 6], 15: [-1, 4], 18: [-1, 13], 21: [-1, 10]}

    def forward(self, x):
        """前向传播，返回多尺度特征图。"""
        y = []
        for i, m in enumerate(self.model):
            if i == 23:
                break
            if i in self.routes:
                f = self.routes[i]
                x = m([x if j == -1 else y[j] for j in f])
            else:
                x = m(x)
            y.append(x)
        return y[0], y[1], y[16], y[19], y[22]

class TAONot42VisionModel(nn.Module):
    """TAO-Not-42 视觉模型，集成了检测、分割、多帧几何解译、Splatting 物理组装和追踪功能。"""
    def __init__(self):
        super().__init__()
        self.segmenter = YOLOEBackbone() 
        
        self.geom_decoder = UnifiedGeometryDecoder(128, 64, 32)
        self.se3_physics_head = SE3PhysicsHead(ch_list=[128, 256, 512], prototype_ch=32)
        self.rigid_projector = RigidFlowProjector()
        self.obj_rigid_projector = ObjectRigidFlowProjector()
        
        self.st_block = SpatioTemporalMambaBlock(128)
        self.st_block_p4 = SpatioTemporalMambaBlock(256)
        self.st_block_p5 = SpatioTemporalMambaBlock(512)
        
        self.feature_predictor = FeaturePredictorHead(128)
        self.track_module = TrackQueryModule(feat_channels=128, num_queries=32, num_heads=4, nc=4585, nm=32)

        self.f1_temporal = nn.Conv3d(32, 32, kernel_size=(3, 1, 1), padding=(1, 0, 0), groups=32)
        self.f2_temporal = nn.Conv3d(64, 64, kernel_size=(3, 1, 1), padding=(1, 0, 0), groups=64)

    def extract_features(self, peripheral):
        return self.segmenter(peripheral)

    def forward_physics(self, f1, f2, p3_fused, p4, p5, dt, step, get_loss_weights_fn=None, original_shape=None, tgts=None, K=None, K_inv=None, prev_queries=None, gt_pose=None, box_prompts=None):
        B, T = f1.shape[:2]
        h, w = original_shape if original_shape else (f1.shape[3] * 2, f1.shape[4] * 2)
        t0 = torch.rand(B, 1, device=f1.device) * 1000.0
        t_abs = t0 + torch.cumsum(dt, dim=1)

        # 1. 时序特征混合
        next_st, spatiotemporal_p3, spatiotemporal_p4, spatiotemporal_p5 = self._run_spatiotemporal_mixing(
            p3_fused, p4, p5, t_abs
        )

        lw = get_loss_weights_fn(step) if get_loss_weights_fn else {"flow": 1, "box": 1, "mask": 1, "anom": 1}

        # 2. 运行 YOLOE 分割预测头
        seg_preds = self.segmenter.model[-1]([
            spatiotemporal_p3.flatten(0, 1),
            spatiotemporal_p4.flatten(0, 1),
            spatiotemporal_p5.flatten(0, 1)
        ], compute_cls=False)
        seg_dict = seg_preds if isinstance(seg_preds, dict) else {}

        # 3. 严格遵循物理方程的几何与运动解码 (深度、位姿、Object Splatting 拼装最终光流)
        depth_pred, flow_pred, ego_pose_dict = self._run_geometry_decoding(
            f1, f2, spatiotemporal_p3, spatiotemporal_p4, spatiotemporal_p5, B, T, h, w, K, K_inv, gt_pose
        )
        
        ego_action_vec = torch.cat([ego_pose_dict["t"], ego_pose_dict["rot6d"]], dim=1)

        # 4. 异常检测与自监督计算 (结合不确定性方差)
        feat_err = self._run_anomaly_detection(next_st, ego_action_vec, lw["anom"], B, T, f1.device)

        # 5. 跨 Chunk 持久化时序追踪
        track_out, next_queries = self._run_tracking(spatiotemporal_p3, prev_queries, box_prompts)

        # 6. 将 anomaly_map 插值到与 depth, flow 同样的高分辨率 (h, w) 以对齐空间结构
        anomaly_map_resized = F.interpolate(
            feat_err.flatten(0, 1).unsqueeze(1),
            size=(h, w),
            mode="bilinear",
            align_corners=False
        ).squeeze(1)

        return {
            **seg_dict, 
            "depth": depth_pred, 
            "log_depth": torch.log(depth_pred + 1e-4), 
            "ego_pose": ego_pose_dict,
            "flow": flow_pred,
            "features": spatiotemporal_p3.flatten(0, 1), 
            "anomaly_map": anomaly_map_resized,
            "feature_error": feat_err.mean(),
            "track_boxes":   track_out["track_boxes"],
            "track_classes": track_out["track_classes"],
            "track_alive":   track_out["track_alive"],
            "track_masks":   track_out["track_masks"],
            "next_queries":  next_queries
        }

    def _run_spatiotemporal_mixing(self, p3_fused, p4, p5, t_abs):
        def update_st(block, p_feat):
            B_s, T_s, C_s, H_s, W_s = p_feat.shape
            pooled = F.avg_pool2d(p_feat.flatten(0, 1), 2, 2).view(B_s, T_s, C_s, H_s//2, W_s//2)
            st_out = block(pooled, t_abs)
            st_out_up = F.interpolate(st_out.flatten(0, 1), size=(H_s, W_s), mode="bilinear", align_corners=False).view(B_s, T_s, C_s, H_s, W_s)
            return st_out, p_feat + st_out_up

        next_st, spatiotemporal_p3 = update_st(self.st_block, p3_fused)
        next_st_p4, spatiotemporal_p4 = update_st(self.st_block_p4, p4)
        next_st_p5, spatiotemporal_p5 = update_st(self.st_block_p5, p5)
        return next_st, spatiotemporal_p3, spatiotemporal_p4, spatiotemporal_p5

    def _run_geometry_decoding(self, f1, f2, spatiotemporal_p3, spatiotemporal_p4, spatiotemporal_p5, B, T, h, w, K, K_inv, gt_pose=None):
        f1_t = self.f1_temporal(f1.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
        f2_t = self.f2_temporal(f2.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)

        # UnifiedGeometryDecoder 仅负责解译深度与遮挡
        geom_out = self.geom_decoder(
            f1_t.flatten(0, 1), f2_t.flatten(0, 1), spatiotemporal_p3.flatten(0, 1)
        )
        inv_depth_raw = geom_out["inv_depth"]
        
        # SE3PhysicsHead 预测位姿、对象锚点、覆盖权重，并组装出 dense_obj_mask 和 residual_flow
        se3_out = self.se3_physics_head([
            spatiotemporal_p3.flatten(0, 1),
            spatiotemporal_p4.flatten(0, 1),
            spatiotemporal_p5.flatten(0, 1)
        ])
        
        T_cam = se3_out["se3_cam"]["T"]
        
        # 伪造默认内参矩阵，防止 K=None
        if K is None:
            B_T = B * T
            device = f1.device
            K = torch.eye(3, device=device).unsqueeze(0).repeat(B_T, 1, 1)
            K[:, 0, 0] = w
            K[:, 1, 1] = h
            K[:, 0, 2] = w / 2.0
            K[:, 1, 2] = h / 2.0
            K_inv = torch.inverse(K)
            
        inv_depth_resized = F.interpolate(inv_depth_raw, size=(h, w), mode="bilinear", align_corners=False)
        
        if K is not None and K.shape[0] != B * T:
            K = K.expand(B * T, -1, -1)
            K_inv = K_inv.expand(B * T, -1, -1)

        # Invariant 2 (gradient decoupling) + Invariant 3 (fp32 geometry):
        #   All projective geometry runs in fp32 outside AMP — perspective division
        #   K@X / Z has dynamic range that exceeds fp16 and permanently poisons BN
        #   running_var when it overflows during the forward pass.
        #   inv_depth is detached for the flow computation: depth trains from its own
        #   GT-depth loss (loss_depth), so the -1/Z² amplifier on the depth→flow
        #   gradient path is architecturally eliminated, not just clipped.
        with torch.autocast(device_type=inv_depth_resized.device.type, enabled=False):
            inv_d = inv_depth_resized.detach().float()
            K32, K_inv32 = K.float(), K_inv.float()

            if gt_pose is not None:
                from models.custom_heads import rot6d_to_matrix, make_4x4_transform
                R_gt = rot6d_to_matrix(gt_pose[:, 3:9].float())
                T_cam_gt = make_4x4_transform(R_gt, gt_pose[:, :3].float())
                flow_rigid = self.rigid_projector(inv_d, T_cam_gt, K32, K_inv32)
            else:
                flow_rigid = self.rigid_projector(inv_d, T_cam.float(), K32, K_inv32)

            flow_final = self.obj_rigid_projector(
                inv_d,
                se3_out["dense_obj_twist"].float(),
                se3_out["dense_obj_mask"].float(),
                se3_out["residual_flow"].float(),
                flow_rigid, K32, K_inv32,
            )
            # flow_norm = tanh(flow_px / W) * 2  (equivalent to old two-step formula)
            flow_final_norm = torch.tanh(flow_final * (1.0 / float(w))) * 2.0

            # depth_pred retains its gradient — trained by loss_depth, not by flow loss
            depth_pred = (1.0 / (inv_depth_resized.float() + 1e-6)).view(B*T, h, w)

        return depth_pred, flow_final_norm, se3_out["se3_cam"]

    def _run_anomaly_detection(self, next_st, ego_action_vec, lw_anom, B, T, device):
        feat_err = torch.zeros(B, T, next_st.shape[-2], next_st.shape[-1], device=device)
        if lw_anom > 0 and T > 1:
            prev_st = next_st[:, :-1].flatten(0, 1)
            prev_ego = ego_action_vec.view(B, T, 9)[:, :-1].flatten(0, 1)
            
            predicted = self.feature_predictor(prev_st, prev_ego)
            pred_feat = predicted["pred_feat"].view(B, T-1, *next_st.shape[2:])
            uncertainty = predicted["uncertainty"].view(B, T-1, *next_st.shape[2:])
            
            error = F.smooth_l1_loss(pred_feat, next_st[:, 1:], reduction="none")
            # float16 下 error/uncertainty 最大可达 ~13.5/1e-4=135000 > float16 上限 65504 → inf → NaN
            # 升到 float32 计算，同时将 uncertainty 最小值提高到 1e-3 防止 loss_anom 数值爆炸
            anomaly_score = (error.float() / uncertainty.float().clamp(min=1e-3)).to(error.dtype).mean(dim=2)
            feat_err[:, 1:] = anomaly_score
        return feat_err

    def _run_tracking(self, spatiotemporal_p3, prev_queries=None, box_prompts=None):
        return self.track_module(spatiotemporal_p3, prev_queries, box_prompts)
