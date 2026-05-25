import torch
import torch.nn as nn

from blocks import (
    C2PSA,
    SPPF,
    C3k2,
    Concat,
    Conv,
    DepthDecoder,
    EgoPoseHead,
    FastSAMStyleSegmenter,
    FeaturePredictorHead,
    FlowDecoder,
    TimeAwareConvGRUCell,
    YOLOESegment26,
    _FastSAMPredictionHead,
)
from utils import decode_dfl_boxes


class MyYOLOE(nn.Module):
    def __init__(self):
        super().__init__()

        def c(dim):
            return int(dim * 0.5)

        def n(depth):
            return max(round(depth * 0.5), 1)

        self.model = nn.Sequential(
            Conv(3, c(64), 3, 2),  # 0 (f1)
            Conv(c(64), c(128), 3, 2),  # 1 (f2)
            C3k2(c(128), c(256), n=n(2), c3k=False, e=0.25),  # 2
            Conv(c(256), c(256), 3, 2),  # 3
            C3k2(c(256), c(512), n=n(2), c3k=False, e=0.25),  # 4
            Conv(c(512), c(512), 3, 2),  # 5
            C3k2(c(512), c(512), n=n(2), c3k=True),  # 6
            Conv(c(512), c(1024), 3, 2),  # 7
            C3k2(c(1024), c(1024), n=n(2), c3k=True),  # 8
            SPPF(c(1024), c(1024), k=5, n=3, shortcut=True),  # 9
            C2PSA(c(1024), c(1024), n=n(2), e=0.5),  # 10
            nn.Upsample(scale_factor=2.0, mode="nearest"),  # 11
            Concat(dimension=1),  # 12. Takes [11, 6]
            C3k2(c(1024) + c(512), c(512), n=n(2), c3k=True),  # 13
            nn.Upsample(scale_factor=2.0, mode="nearest"),  # 14
            Concat(dimension=1),  # 15. Takes [14, 4]
            C3k2(c(512) + c(512), c(256), n=n(2), c3k=True),  # 16 (P3)
            Conv(c(256), c(256), 3, 2),  # 17
            Concat(dimension=1),  # 18. Takes [17, 13]
            C3k2(c(256) + c(512), c(512), n=n(2), c3k=True),  # 19 (P4)
            Conv(c(512), c(512), 3, 2),  # 20
            Concat(dimension=1),  # 21. Takes [20, 10]
            C3k2(
                c(512) + c(1024), c(1024), n=n(2), c3k=True, e=0.5, attn=True
            ),  # 22 (P5)
            YOLOESegment26(
                nc=80, nm=32, npr=256, embed=512, reg_max=32, ch=(128, 256, 512)
            ),  # 23
        )

        for m in self.model:
            m.f = -1

        self.model[12].f = [-1, 6]
        self.model[15].f = [-1, 4]
        self.model[18].f = [-1, 13]
        self.model[21].f = [-1, 10]
        self.model[23].f = [16, 19, 22]

    def forward(self, x):
        y = []
        for i, m in enumerate(self.model):
            x = m([y[j] for j in m.f] if isinstance(m.f, list) else x)
            y.append(x)

        # 提取各个尺度的特征
        f1 = y[0]
        f2 = y[1]
        p3_fused = y[16]
        p4 = y[19]
        p5 = y[22]
        return f1, f2, p3_fused, p4, p5


class TAONot42VisionModel(nn.Module):
    def __init__(self, base_channels=48, hidden_channels=768):
        super().__init__()
        self.segmenter = MyYOLOE()
        self.depth_decoder = DepthDecoder(128, 64, 32, ch_gru=128)
        self.conv_gru = TimeAwareConvGRUCell(128, 128)
        self.pose_head = EgoPoseHead(128)
        self.flow_head = FlowDecoder(128)
        self.feature_predictor = FeaturePredictorHead(128)
        self.state_update_gate_head = nn.Sequential(
            nn.Linear(128 + 1, 64), nn.SiLU(), nn.Linear(64, 1)
        )

    def extract_features(self, peripheral):
        f1, f2, p3_fused, p4, p5 = self.segmenter(peripheral)
        return f1, f2, p3_fused, p4, p5

    def forward_physics(
        self,
        f1,
        f2,
        p3_fused,
        p4,
        p5,
        dt,
        step,
        state=None,
        get_loss_weights_fn=None,
        original_shape=None,
    ):
        b = f1.shape[0]
        if original_shape:
            h, w = original_shape
        else:
            h, w = f1.shape[2] * 2, f1.shape[3] * 2
        state = state or {}

        # Step 2: GRU 时空融合
        p3_down = torch.nn.functional.avg_pool2d(p3_fused, kernel_size=2, stride=2)
        gru_state = state.get("gru", None)
        next_gru_state = self.conv_gru(p3_down, dt, gru_state)
        next_gru_state_up = torch.nn.functional.interpolate(
            next_gru_state,
            size=p3_fused.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        spatiotemporal_p3 = p3_fused + next_gru_state_up

        # Step 3: 直接调用检测头 (YOLOESegment26 是 segmenter.model 的最后一层)
        # 使用 Gradient Checkpointing 极大地节省这部分庞大图结构的显存占用！
        import torch.utils.checkpoint as checkpoint

        def run_yolo_head(p3, p4, p5):
            return self.segmenter.model[-1]([p3, p4, p5])

        preds = checkpoint.checkpoint(
            run_yolo_head, spatiotemporal_p3, p4, p5, use_reentrant=False
        )

        # The rest of the physics pipeline runs on the temporally tracked spatiotemporal_p3
        depth_logits = self.depth_decoder(f1, f2, spatiotemporal_p3)

        depth_logits = torch.nn.functional.interpolate(
            depth_logits, size=(h, w), mode="bilinear", align_corners=False
        ).squeeze(1)
        log_depth_pred = depth_logits
        depth_pred = torch.exp(torch.clamp(log_depth_pred, min=-4.6, max=4.6))

        ego_pose = self.pose_head(spatiotemporal_p3)

        # lw is required, assume it's passed or all active
        lw = (
            get_loss_weights_fn(step)
            if get_loss_weights_fn
            else {"flow": 1, "box": 1, "mask": 1, "anom": 1}
        )

        if lw["flow"] > 0:
            raw_flow = self.flow_head(spatiotemporal_p3)
            pred_flow = 1.5 * torch.tanh(raw_flow)
            preds["flow"] = pred_flow
        else:
            pred_flow = None

        selected_feature = spatiotemporal_p3.mean(dim=[2, 3])
        gate_in = torch.cat([selected_feature, dt.view(-1, 1)], dim=-1)
        raw_gate = self.state_update_gate_head(gate_in)
        gate = torch.sigmoid(raw_gate).view(-1, 1, 1, 1)

        # VERY IMPORTANT: final_gru_state MUST be 40x40 to feed back into next iteration!
        final_gru_state = (
            gru_state * (1.0 - gate) + next_gru_state * gate
            if gru_state is not None
            else next_gru_state
        )

        prev_ego_pose = state.get("prev_ego", torch.zeros_like(ego_pose))
        if gru_state is not None and lw["anom"] > 0:
            # Anomaly map can be 40x40 since it compares GRU states directly!
            pred_current_feature = self.feature_predictor(gru_state, prev_ego_pose)
            target_feature = final_gru_state.detach()
            feature_error_map = torch.nn.functional.smooth_l1_loss(
                pred_current_feature, target_feature, reduction="none"
            ).mean(dim=1)
        else:
            feature_error_map = torch.zeros(
                b, next_gru_state.shape[2], next_gru_state.shape[3], device=f1.device
            )

        # We output the tracked o2o predictions for active usage!
        return {
            "objectness": preds["o2o_objectness"],
            "classification": preds["o2o_classification"],
            "box_dist": preds["o2o_boxes"] if lw["box"] > 0 else None,
            "boxes": decode_dfl_boxes(preds["o2o_boxes"], reg_max=32)
            if lw["box"] > 0
            else None,
            "mask_coefficients": preds["o2o_mask_coefficients"]
            if lw["mask"] > 0
            else None,
            "mask_prototypes": preds["mask_prototypes"] if lw["mask"] > 0 else None,
            "depth": depth_pred,
            "log_depth": log_depth_pred,
            "ego_pose": ego_pose,
            "flow": pred_flow,
            "features": spatiotemporal_p3,
            "anomaly_map": feature_error_map,
            "feature_error": feature_error_map.mean(),
            "state_update_gate": gate.view(b),
            "next_state": {"gru": final_gru_state, "prev_ego": ego_pose},
            # We also attach dense supervision targets for computing gradients
            "dense_objectness": preds["objectness"],
            "dense_classification": preds["classification"],
            "dense_box_dist": preds["boxes"],
            "dense_mask_coefficients": preds["mask_coefficients"],
        }

    def forward(self, peripheral, dt, step, state=None, get_loss_weights_fn=None):
        b, _, h, w = peripheral.shape
        f1, f2, p3_fused, p4, p5 = self.extract_features(peripheral)
        return self.forward_physics(
            f1,
            f2,
            p3_fused,
            p4,
            p5,
            dt,
            step,
            state,
            get_loss_weights_fn,
            original_shape=(h, w),
        )
