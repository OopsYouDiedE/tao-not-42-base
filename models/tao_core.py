import torch
import torch.nn as nn
import torch.nn.functional as F

from models.yolo_blocks import *
from models.custom_heads import *

# =====================================================================


class MyYOLOE(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.ModuleList([
            Conv(3, 32, 3, 2),  # 0
            Conv(32, 64, 3, 2),  # 1
            C3k2(64, 128, n=1, shortcut=True, c3k=False, e=0.25),  # 2
            Conv(128, 128, 3, 2),  # 3
            C3k2(128, 256, n=1, shortcut=True, c3k=False, e=0.25),  # 4
            Conv(256, 256, 3, 2),  # 5
            C3k2(256, 256, n=1, shortcut=False, c3k=True, e=0.5),  # 6
            Conv(256, 512, 3, 2),  # 7
            C3k2(512, 512, n=1, shortcut=False, c3k=True, e=0.5),  # 8
            SPPF(512, 512, k=5),  # 9
            C2PSA(512, 512, n=1, e=0.5),  # 10
            nn.Upsample(scale_factor=2.0, mode='nearest'),  # 11
            Concat(1),  # 12
            C3k2(768, 256, n=1, shortcut=False, c3k=True, e=0.5),  # 13
            nn.Upsample(scale_factor=2.0, mode='nearest'),  # 14
            Concat(1),  # 15
            C3k2(512, 128, n=1, shortcut=False, c3k=True, e=0.5),  # 16 (P3)
            Conv(128, 128, 3, 2),  # 17
            Concat(1),  # 18
            C3k2(384, 256, n=1, shortcut=False, c3k=True, e=0.5),  # 19 (P4)
            Conv(256, 256, 3, 2),  # 20
            Concat(1),  # 21
            C3k2Attention(768, 512, n=1, shortcut=False, e=0.5),  # 22 (P5)
            YOLOESegment26(nc=80, nm=32, npr=128, embed=80,
                           reg_max=1, ch=(128, 256, 512))  # 23
        ])

        self.routes = {12: [-1, 6], 15: [-1, 4],
                       18: [-1, 13], 21: [-1, 10], 23: [16, 19, 22]}

    def forward(self, x):
        y = []
        for i, m in enumerate(self.model):
            if i == 23:
                break
            if i in self.routes:
                f = self.routes[i]
                x = m([x if j == -1 else y[j] for j in f]
                      if isinstance(f, list) else (y[f] if f != -1 else x))
            else:
                x = m(x)
            y.append(x)
        return y[0], y[1], y[16], y[19], y[22]


class TAONot42VisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.segmenter = MyYOLOE()
        self.geom_decoder = UnifiedGeometryDecoder(128, 64, 32)
        self.st_block = SpatioTemporalMambaBlock(128)
        self.st_block_p4 = SpatioTemporalMambaBlock(256)
        self.st_block_p5 = SpatioTemporalMambaBlock(512)
        self.pose_head = EgoPoseHead(128)
        self.feature_predictor = FeaturePredictorHead(128)
        self.state_update_gate_head = nn.Sequential(
            nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1))
        self.track_module = TrackQueryModule(
            feat_channels=128, num_queries=32, num_heads=4, nc=4585, nm=32)

        self.f1_temporal = nn.Conv3d(32, 32, kernel_size=(
            3, 1, 1), padding=(1, 0, 0), groups=32)
        self.f2_temporal = nn.Conv3d(64, 64, kernel_size=(
            3, 1, 1), padding=(1, 0, 0), groups=64)

    def extract_features(self, peripheral):
        return self.segmenter(peripheral)

    def forward_physics(self, f1, f2, p3_fused, p4, p5, dt, step, get_loss_weights_fn=None, original_shape=None, tgts=None):
        B, T = f1.shape[:2]
        h, w = original_shape if original_shape else (
            f1.shape[3] * 2, f1.shape[4] * 2)

        t0 = torch.rand(B, 1, device=f1.device) * 1000.0
        t_abs = t0 + torch.cumsum(dt, dim=1)

        def update_st(block, p_feat):
            B_s, T_s, C_s, H_s, W_s = p_feat.shape
            pooled = F.avg_pool2d(p_feat.flatten(0, 1), 2, 2).view(
                B_s, T_s, C_s, H_s//2, W_s//2)
            st_out = block(pooled, t_abs)
            st_out_up = F.interpolate(st_out.flatten(0, 1), size=(
                H_s, W_s), mode="bilinear", align_corners=False).view(B_s, T_s, C_s, H_s, W_s)
            return st_out, p_feat + st_out_up

        next_st, spatiotemporal_p3 = update_st(self.st_block, p3_fused)
        next_st_p4, spatiotemporal_p4 = update_st(self.st_block_p4, p4)
        next_st_p5, spatiotemporal_p5 = update_st(self.st_block_p5, p5)

        preds = self.segmenter.model[-1]([
            spatiotemporal_p3.flatten(0, 1),
            spatiotemporal_p4.flatten(0, 1),
            spatiotemporal_p5.flatten(0, 1)
        ])

        lw = get_loss_weights_fn(step) if get_loss_weights_fn else {
            "flow": 1, "box": 1, "mask": 1, "anom": 1}

        ego_pose = self.pose_head(spatiotemporal_p3.flatten(0, 1))

        # f1: [B, T, 32, H, W] -> [B, 32, T, H, W] -> apply 3D conv -> back to [B, T, 32, H, W]
        f1_t = self.f1_temporal(f1.permute(
            0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
        f2_t = self.f2_temporal(f2.permute(
            0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)

        depth_raw, flow_raw = self.geom_decoder(
            f1_t.flatten(0, 1), f2_t.flatten(0, 1), spatiotemporal_p3.flatten(0, 1), ego_pose_feat=ego_pose, need_flow=(lw["flow"] > 0)
        )

        depth_pred = torch.exp(torch.clamp(F.interpolate(
            depth_raw, size=(h, w), mode="bilinear", align_corners=False).squeeze(1), min=-4.6, max=4.6)).view(B*T, h, w)
        flow_pred = flow_raw * 1.5 if flow_raw is not None else None

        if flow_pred is not None:
            from utils.geometry import quaternion_to_matrix, matrix_to_6d, compute_rigid_flow, generate_intrinsics
            pose_for_flow = ego_pose
            if tgts is not None and "cam_pos_t" in tgts and "cam_quat_next" in tgts:
                R_n_inv = quaternion_to_matrix(tgts["cam_quat_next"]).transpose(1, 2)
                trans_diff = torch.bmm(R_n_inv, (tgts["cam_pos_t"] - tgts["cam_pos_next"]).unsqueeze(-1)).squeeze(-1)
                rot_diff = matrix_to_6d(torch.bmm(R_n_inv, quaternion_to_matrix(tgts["cam_quat_t"])))
                pose_for_flow = torch.cat([trans_diff, rot_diff], dim=1)
                
            cam_f = tgts.get("camera_focal_length", None) if tgts else None
            cam_s = tgts.get("camera_sensor_width", None) if tgts else None
            
            K, K_inv = generate_intrinsics(h, w, f1.device, focal_length=cam_f, sensor_width=cam_s)
            rigid_flow = compute_rigid_flow(depth_pred.unsqueeze(1), pose_for_flow, K, K_inv, depth_is_distance=True)
            flow_pred = rigid_flow + flow_pred

        gate_logits = self.state_update_gate_head(
            spatiotemporal_p3.mean(dim=[3, 4]).flatten(0, 1))
        gate = torch.sigmoid(gate_logits).view(B*T)

        feat_err = torch.zeros(
            B, T, next_st.shape[-2], next_st.shape[-1], device=f1.device)
        if lw["anom"] > 0 and T > 1:
            prev_st = next_st[:, :-1].flatten(0, 1)
            prev_ego = ego_pose.view(B, T, 9)[:, :-1].flatten(0, 1)
            predicted_st = self.feature_predictor(
                prev_st, prev_ego).view(B, T-1, *next_st.shape[2:])
            feat_err[:, 1:] = F.smooth_l1_loss(
                predicted_st, next_st[:, 1:], reduction="none").mean(dim=2)

        track_out = self.track_module(spatiotemporal_p3)

        return {
            "objectness": preds["o2o_objectness"], "classification": preds["o2o_classification"],
            "box_dist": preds["o2o_boxes"] if lw["box"] > 0 else None,
            "boxes": decode_dfl_boxes(preds["o2o_boxes"], 32) if lw["box"] > 0 else None,
            "mask_coefficients": preds["o2o_mask_coefficients"] if lw["mask"] > 0 else None,
            "mask_prototypes": preds["mask_prototypes"] if lw["mask"] > 0 else None,
            "depth": depth_pred, "log_depth": torch.log(depth_pred), "ego_pose": ego_pose,
            "flow": flow_pred,
            "features": spatiotemporal_p3.flatten(0, 1), "anomaly_map": feat_err.flatten(0, 1),
            "feature_error": feat_err.mean(), "state_update_gate": gate,
            "dense_objectness": preds["objectness"], "dense_classification": preds["classification"],
            "dense_box_dist": preds["boxes"], "dense_mask_coefficients": preds["mask_coefficients"],
            "track_boxes":   track_out["track_boxes"],
            "track_classes": track_out["track_classes"],
            "track_alive":   track_out["track_alive"],
            "track_masks":   track_out["track_masks"],
            "attributes":    preds["attributes"],
        }

# =====================================================================
