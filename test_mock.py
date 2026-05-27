import sys
import types
import torch
import numpy as np
import cv2
import os

# 1. Define MockMamba class locally
class MockMamba(torch.nn.Module):
    def __init__(self, d_model, *args, **kwargs):
        super().__init__()
        self.proj = torch.nn.Linear(d_model, d_model)
    def forward(self, x, *args, **kwargs):
        return self.proj(x)

# 2. Inject MockMamba into sys.modules to satisfy other components
mamba_mock = types.ModuleType("mamba_ssm")
mamba_mock.Mamba = MockMamba
sys.modules["mamba_ssm"] = mamba_mock

# 2b. Inject scipy mock (for test_mock; Colab has real scipy for training)
scipy_mock = types.ModuleType("scipy")
scipy_opt_mock = types.ModuleType("scipy.optimize")
scipy_opt_mock.linear_sum_assignment = lambda cost: (
    np.arange(min(cost.shape)), np.arange(min(cost.shape)))
scipy_mock.optimize = scipy_opt_mock
sys.modules["scipy"] = scipy_mock
sys.modules["scipy.optimize"] = scipy_opt_mock

# 3. Import modules and perform dynamic monkey patching injection of Mamba before any block initializes
import models.custom_heads
models.custom_heads.Mamba = MockMamba

from models import TAONot42VisionModel
from utils import get_loss_weights, compute_physics_loss, save_visualization, compute_track_loss
from dataset import process_batch_on_gpu
from trainer import TAOTrainer

def load_yoloe_weights(model, path="yoloe-26s-seg-pf.pt"):
    import torch
    import torch.nn as nn
    import os
    import urllib.request
    
    if not os.path.exists(path):
        weights_url = f"https://github.com/ultralytics/assets/releases/download/v8.4.0/{path}"
        print(f"Downloading weights from {weights_url} ...")
        urllib.request.urlretrieve(weights_url, path)
        print("Download complete.")

    for name, module in model.named_modules():
        if module.__class__.__name__ == 'Conv':
            c1 = module.conv.in_channels
            c2 = module.conv.out_channels
            k = module.conv.kernel_size
            s = module.conv.stride
            p = module.conv.padding
            g = module.conv.groups
            d = module.conv.dilation
            
            new_conv = nn.Conv2d(c1, c2, k, s, p, groups=g, dilation=d, bias=True)
            new_conv.to(module.conv.weight.device)
            module.conv = new_conv
            module.bn = nn.Identity()
        elif module.__class__.__name__ == 'PSABlock':
            if hasattr(module, 'add_norm1'): module.add_norm1 = torch.nn.Identity()
            if hasattr(module, 'add_norm2'): module.add_norm2 = torch.nn.Identity()

    try:
        from ultralytics import YOLO
        ul_model = YOLO(path)
        sd = ul_model.model.state_dict()
    except ImportError:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd = ckpt["model"].state_dict() if isinstance(ckpt, dict) and "model" in ckpt else (ckpt.state_dict() if hasattr(ckpt, "state_dict") else ckpt)

    
    # 2. (REMOVED) 以前会把 4585 截断成 nc，现在不再截断，保留零样本能力

    tgt = model.state_dict()
    loaded_keys = {k for k, v in sd.items() if (k.replace("model.model.", "segmenter.model.").replace("model.", "segmenter.model.") if k.startswith("model.") else k) in tgt and tgt[(k.replace("model.model.", "segmenter.model.").replace("model.", "segmenter.model.") if k.startswith("model.") else k)].shape == v.shape}
    print(f"====================================================")
    print(f"[YOLO] Successfully loaded {len(loaded_keys)}/{len(sd)} keys from '{path}'!")
    print(f"====================================================")
    tgt.update({k.replace("model.model.", "segmenter.model.").replace("model.", "segmenter.model.") if k.startswith("model.") else k: v for k, v in sd.items() if k in loaded_keys})
    model.load_state_dict(tgt)

class DummyTrainer:
    """Minimal self substitute to call TAOTrainer helper methods directly."""
    def __init__(self, device, global_step):
        self.device = device
        self.global_step = global_step

def generate_synthetic_physical_video(B=1, T=12, H=256, W=256):
    """
    Generates a high-fidelity synthetic physical video of a 3D ball drifting closer 
    to the camera over a gradient background. Simulates proper geometric depth, 
    ego-motion flow, dynamic object flow, and object segmentation.
    """
    video = np.zeros((B, T, H, W, 3), dtype=np.uint8)
    depth = np.ones((B, T, H, W), dtype=np.float32) * 50.0  # Background distance: 50m
    seg = np.zeros((B, T, H, W), dtype=np.int16)
    flow = np.zeros((B, T, H, W, 2), dtype=np.float32)
    
    for b in range(B):
        for t in range(T):
            # Background gradient
            img = np.zeros((H, W, 3), dtype=np.uint8)
            for y in range(H):
                img[y, :, 0] = int(120 + 40 * (y / H))  # Blue
                img[y, :, 1] = int(60 + 80 * (y / H))   # Green
                img[y, :, 2] = int(40 + 20 * (y / H))   # Red
            
            # Simulated camera movement: pan right
            cam_dx = 1.0 * t
            img = np.roll(img, -int(cam_dx), axis=1)
            bg_flow_x = -1.0
            
            # Render a 3D red ball drifting closer
            center_x = int(60 + 10 * t)
            center_y = int(140 - 3 * t)
            radius = int(24 + 1.2 * t)
            
            cv2.circle(img, (center_x, center_y), radius, (0, 0, 255), -1)
            
            # Ball depth decreases as it gets closer
            ball_depth = 5.0 - 0.2 * t
            y_grid, x_grid = np.ogrid[:H, :W]
            dist_sq = (x_grid - center_x)**2 + (y_grid - center_y)**2
            ball_mask = dist_sq <= radius**2
            
            depth_frame = np.ones((H, W), dtype=np.float32) * 50.0
            depth_frame[ball_mask] = ball_depth
            
            seg_frame = np.zeros((H, W), dtype=np.int16)
            seg_frame[ball_mask] = 1
            
            flow_frame = np.zeros((H, W, 2), dtype=np.float32)
            flow_frame[..., 0] = bg_flow_x
            flow_frame[ball_mask, 0] = 10.0
            flow_frame[ball_mask, 1] = -3.0
            
            video[b, t] = img
            depth[b, t] = depth_frame
            seg[b, t] = seg_frame
            flow[b, t] = flow_frame
            
    return video, depth, seg, flow

def get_movi_e_or_fallback(npz_path="movi_e_sample.npz", B=1, T=12, H=256, W=256):
    """
    Dynamically loads a genuine Kubric MOVi-E dataset sample exported from Colab,
    extracting all 7 physical elements: video, depth, segmentation, flow, camera positions,
    camera quaternions, and is_dynamic parameters.
    """
    if os.path.exists(npz_path):
        print(f"====================================================")
        print(f"[INFO] Genuine MOVi-E Dataset sample detected: '{npz_path}'!")
        print(f"====================================================")
        try:
            data = np.load(npz_path, allow_pickle=True)
            v_np = data["video"]                  # Expected: [T, H, W, 3] or [B, T, H, W, 3]
            d_np = data["depth"]                  # Expected: [T, H, W] or [B, T, H, W]
            s_np = data["segmentation"]           # Expected: [T, H, W] or [B, T, H, W]
            f_np = data["forward_flow"].astype(np.float32)           # Expected: [T, H, W, 2] or [B, T, H, W, 2]
            cp_np = data.get("cam_pos")           # Expected: [T, 3] or [B, T, 3]
            cq_np = data.get("cam_quat")          # Expected: [T, 4] or [B, T, 4]
            id_np = data.get("is_dynamic")        # Expected: [NumInstances] or List of them
            
            # 1. Dimensional Alignment Helper
            def ensure_5d(arr, is_channel=False):
                if len(arr.shape) == 3 and not is_channel: # [T, H, W] -> [1, T, H, W]
                    arr = np.expand_dims(arr, axis=0)
                elif len(arr.shape) == 4 and is_channel:   # [T, H, W, C] -> [1, T, H, W, C]
                    arr = np.expand_dims(arr, axis=0)
                if arr.shape[0] < B:
                    arr = np.repeat(arr, B, axis=0)
                return arr
                
            v_np = ensure_5d(v_np, is_channel=True)[:, :T]
            d_np = ensure_5d(d_np, is_channel=False)[:, :T]
            s_np = ensure_5d(s_np, is_channel=False)[:, :T]
            f_np = ensure_5d(f_np, is_channel=True)[:, :T]
            
            # Align Camera Position
            if cp_np is not None:
                if len(cp_np.shape) == 2: # [T, 3] -> [1, T, 3]
                    cp_np = np.expand_dims(cp_np, axis=0)
                if cp_np.shape[0] < B:
                    cp_np = np.repeat(cp_np, B, axis=0)
                cp_np = cp_np[:, :T]
            else:
                cp_np = np.zeros((B, T, 3), dtype=np.float32)
                
            # Align Camera Rotation Quaternions
            if cq_np is not None:
                if len(cq_np.shape) == 2: # [T, 4] -> [1, T, 4]
                    cq_np = np.expand_dims(cq_np, axis=0)
                if cq_np.shape[0] < B:
                    cq_np = np.repeat(cq_np, B, axis=0)
                cq_np = cq_np[:, :T]
            else:
                cq_np = np.zeros((B, T, 4), dtype=np.float32)
                cq_np[..., 0] = 1.0
                
            # Align is_dynamic flags
            if id_np is not None:
                if isinstance(id_np, np.ndarray) and len(id_np.shape) == 1:
                    id_np = [id_np for _ in range(B)]
            else:
                id_np = [np.array([True], dtype=bool) for _ in range(B)]
            
            # 2. Spatial Resizing Helper
            if v_np.shape[2] != H or v_np.shape[3] != W:
                print(f"Resizing MOVi-E samples from {v_np.shape[2]}x{v_np.shape[3]} to {H}x{W}...")
                v_res, d_res, s_res, f_res = [], [], [], []
                for b in range(B):
                    v_res.append(np.stack([cv2.resize(v_np[b, t], (W, H)) for t in range(T)]))
                    d_res.append(np.stack([cv2.resize(d_np[b, t], (W, H), interpolation=cv2.INTER_NEAREST) for t in range(T)]))
                    s_res.append(np.stack([cv2.resize(s_np[b, t].astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int16) for t in range(T)]))
                    f_res.append(np.stack([cv2.resize(f_np[b, t], (W, H)) for t in range(T)]))
                v_np, d_np, s_np, f_np = np.stack(v_res), np.stack(d_res), np.stack(s_res), np.stack(f_res)
                
            print("Successfully loaded and prepared genuine Kubric MOVi-E sample!")
            return v_np, d_np, s_np, f_np, cp_np, cq_np, id_np
        except Exception as e:
            print(f"Error loading MOVi-E sample from npz: {e}")
            
    # Graceful fallback online download / 3D Ball Simulation
    print(f"Genuine local sample '{npz_path}' not found.")
    video_path = "sample_video.mp4"
    if not os.path.exists(video_path):
        url = "https://github.com/intel-iot-devkit/sample-videos/raw/master/bottle-detection.mp4"
        print(f"Attempting to download sample video from:\n  {url}")
        try:
            import urllib.request
            urllib.request.urlretrieve(url, video_path)
            print("Download completed successfully!")
        except Exception as e:
            print(f"Failed to download online video sample: {e}")
            print("Falling back to high-fidelity local 3D physical simulator...")
            v_np, d_np, s_np, f_np = generate_synthetic_physical_video(B, T, H, W)
            cp_np = np.zeros((B, T, 3), dtype=np.float32)
            cq_np = np.zeros((B, T, 4), dtype=np.float32)
            cq_np[..., 0] = 1.0
            id_np = [np.array([True], dtype=bool) for _ in range(B)]
            return v_np, d_np, s_np, f_np, cp_np, cq_np, id_np
            
    try:
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened() and len(frames) < T:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (W, H))
            frames.append(frame_resized)
        cap.release()
        
        if len(frames) < T:
            while len(frames) < T:
                frames.append(frames[-1] if len(frames) > 0 else np.zeros((H, W, 3), dtype=np.uint8))
                
        video_np = np.stack(frames, axis=0)
        video_np = np.expand_dims(video_np, axis=0)
        if B > 1:
            video_np = np.repeat(video_np, B, axis=0)
            
        print(f"Successfully loaded fallback video: '{video_path}'!")
        
        depth = np.ones((B, T, H, W), dtype=np.float32) * 10.0
        seg = np.zeros((B, T, H, W), dtype=np.int16)
        flow = np.zeros((B, T, H, W, 2), dtype=np.float32)
        
        for b in range(B):
            for t in range(T):
                gray = cv2.cvtColor(video_np[b, t], cv2.COLOR_RGB2GRAY)
                depth[b, t] = 20.0 - 15.0 * (gray.astype(np.float32) / 255.0)
                _, thresh = cv2.threshold(gray, 128, 1, cv2.THRESH_BINARY)
                seg[b, t] = thresh.astype(np.int16)
                flow[b, t, ..., 0] = -1.0
                
        cp_np = np.zeros((B, T, 3), dtype=np.float32)
        cq_np = np.zeros((B, T, 4), dtype=np.float32)
        cq_np[..., 0] = 1.0
        id_np = [np.array([True], dtype=bool) for _ in range(B)]
        
        return video_np, depth, seg, flow, cp_np, cq_np, id_np
    except Exception as e:
        print(f"Error reading backup video: {e}")
        print("Falling back to high-fidelity local 3D physical simulator...")
        v_np, d_np, s_np, f_np = generate_synthetic_physical_video(B, T, H, W)
        cp_np = np.zeros((B, T, 3), dtype=np.float32)
        cq_np = np.zeros((B, T, 4), dtype=np.float32)
        cq_np[..., 0] = 1.0
        id_np = [np.array([True], dtype=bool) for _ in range(B)]
        return v_np, d_np, s_np, f_np, cp_np, cq_np, id_np

def test_all_stages():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"====================================================")
    print(f"[RUN] RUNNING MULTI-STAGE REAL PHYSICAL TESTING ON {device.type.upper()}")
    print(f"====================================================")
    
    # 1. Instantiate vision model
    model = TAONot42VisionModel().to(device)
    load_yoloe_weights(model, 'yoloe-26s-seg-pf.pt')
    model.train()
    
    # Model Parameter Summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params / 1e6:.2f} M")
    print(f"Static weight memory footprint: {total_params * 4 / (1024**2):.2f} MB\n")
    
    # 2. Get real MOVi-E genuine sample or fallback
    B, T, img_size = 1, 12, 256
    v_np, d_np, s_np, f_np, cp_np, cq_np, id_np = get_movi_e_or_fallback("movi_e_sample.npz", B, T, img_size, img_size)
    
    # Prepare batch dictionary matching the dataset output format
    batch = {
        "video": torch.from_numpy(v_np),
        "depth": torch.from_numpy(d_np),
        "segmentation": torch.from_numpy(s_np),
        "forward_flow": torch.from_numpy(f_np),
        "cam_pos": torch.from_numpy(cp_np),
        "cam_quat": torch.from_numpy(cq_np),
        "is_dynamic": [torch.from_numpy(x) for x in id_np]
    }
    
    # Run GPU preprocessing pipeline
    print("Running GPU batch preprocessing...")
    gpu_batch = process_batch_on_gpu(batch, device, img_size)
    
    # 3. Define curriculum training steps covering all 5 core stages
    stages = {
        1: ("Stage 1: Detection & Depth Focused", 50),
        2: ("Stage 2: Camera Pose Introduction", 250),
        3: ("Stage 3: Optical Flow Introduction", 500),
        4: ("Stage 4: Photo-Error & Class-learning", 1500),
        5: ("Stage 5: Anomaly Self-Supervision Active", 5000)
    }
    
    # Extract preprocessed input sequences
    v_seq = gpu_batch["video"]
    t_max = v_seq.shape[1]
    
    # Mock future frames for warp-based photometric computation in compute_physics_loss
    img_next = torch.zeros_like(v_seq)
    for t in range(t_max):
        img_next[:, t] = v_seq[:, min(t + 1, t_max - 1)]
    
    print("\nStarting multi-stage verification...\n")
    
    for stage_id, (stage_name, step) in stages.items():
        print(f"----------------------------------------------------")
        print(f"[RUN] Testing Curriculum {stage_name} (Step = {step})")
        print(f"----------------------------------------------------")
        
        # Get active loss weights for this step
        lw = get_loss_weights(step)
        active_losses = [k for k, v in lw.items() if v > 0]
        print(f"Active Loss Components: {active_losses}")
        
        # Feature extraction from Segmenter
        with torch.no_grad():
            feats = [f.view(B, T, *f.shape[1:]) for f in model.extract_features(v_seq.flatten(0, 1))]
        
        # Enable gradients for features
        for f in feats:
            f.requires_grad = True
            
        dt = torch.full((B, T), 1.0 / 24.0, device=device)
        
        # Forward Pass
        preds = model.forward_physics(
            *feats, dt, step=step, 
            get_loss_weights_fn=get_loss_weights, 
            original_shape=(img_size, img_size)
        )
        
        # 4. Extract targets directly reusing TAOTrainer._extract_target_chunk to eliminate duplicates
        dummy_trainer = DummyTrainer(device, step)
        tgts = TAOTrainer._extract_target_chunk(dummy_trainer, gpu_batch, c_start=0, c_end=T, max_t=t_max)
        
        # Loss Computation
        loss, l_dict, w_img = compute_physics_loss(
            preds, tgts, 
            v_seq.flatten(0, 1), 
            img_next.flatten(0, 1), 
            mode="supervised", 
            step=step
        )
        
        # Backward Pass
        loss.backward()
        print(f"Loss computed: {loss.item():.4f}")
        for l_name, l_val in l_dict.items():
            if lw.get(l_name.lower()[:4], 0.0) > 0 or l_name == "Tot":
                print(f"  - {l_name} Loss: {l_val.item():.4f}")
                
        # 5. Generate Visualization Output for Visual Inspection at Stage 5
        if stage_id == 5:
            print("\nGenerating visual verification grid for inspection...")
            # Slice the second frame (index 1) from the flattened B*T tensors,
            # matching TAOTrainer visualization slice logic.
            vis_frame_idx = 1  # second frame: has both t-1 and t+1 neighbours
            def slice_vis_frame(v):
                if v is None: return None
                if isinstance(v, list):
                    return [x[(B - 1) * T + vis_frame_idx : (B - 1) * T + vis_frame_idx + 1] if (x.dim() > 0 and x.shape[0] == B * T) else (x[-B:] if x.dim() > 0 else x) for x in v]
                if v.dim() == 0: return v
                if v.shape[0] == B * T:
                    return v[(B - 1) * T + vis_frame_idx : (B - 1) * T + vis_frame_idx + 1]
                return v[-B:]

            vis_dir = "vis_outputs"
            fp = save_visualization(
                v_seq[:, vis_frame_idx],
                {k: slice_vis_frame(v) for k, v in tgts.items()},
                {k: slice_vis_frame(v) for k, v in preds.items()},
                step=step,
                warped_img=slice_vis_frame(w_img) if w_img is not None else None,
                output_dir=vis_dir
            )
            abs_fp = os.path.abspath(fp)
            print(f"[SUCCESS] Visualization saved successfully at: {abs_fp}")
            
    print("\n====================================================")
    print("[SUCCESS] SUCCESS: All 5 curriculum stages verified!")
    print("====================================================")

    # -----------------------------------------------------------------
    # Stage 6: End-to-End Tracking Verification
    # -----------------------------------------------------------------
    print("\n----------------------------------------------------")
    print("[RUN] Stage 6: End-to-End Tracking Module Verification")
    print("----------------------------------------------------")

    # Re-run forward at step=1000 (track loss is active: ramp(500,2000,1.0))
    track_step = 1000
    with torch.no_grad():
        feats6 = [f.view(B, T, *f.shape[1:]) for f in model.extract_features(v_seq.flatten(0, 1))]
    for f in feats6:
        f.requires_grad = True

    dt6 = torch.full((B, T), 1.0 / 24.0, device=device)
    preds6 = model.forward_physics(
        *feats6, dt6, step=track_step,
        get_loss_weights_fn=get_loss_weights,
        original_shape=(img_size, img_size)
    )

    # --- Shape verification ---
    N = 16
    assert "track_boxes"   in preds6, "track_boxes missing from preds"
    assert "track_classes" in preds6, "track_classes missing from preds"
    assert "track_alive"   in preds6, "track_alive missing from preds"
    assert "track_masks"   in preds6, "track_masks missing from preds"

    tb = preds6["track_boxes"]
    tc = preds6["track_classes"]
    ta = preds6["track_alive"]
    tm = preds6["track_masks"]

    assert tb.shape == (B, T, N, 4),   f"track_boxes shape mismatch: {tb.shape}"
    assert tc.shape == (B, T, N, 4585),  f"track_classes shape mismatch: {tc.shape}"
    assert ta.shape == (B, T, N, 1),   f"track_alive shape mismatch: {ta.shape}"
    assert tm.shape == (B, T, N, 32),  f"track_masks shape mismatch: {tm.shape}"
    print(f"  track_boxes:   {list(tb.shape)}  OK")
    print(f"  track_classes: {list(tc.shape)}  OK")
    print(f"  track_alive:   {list(ta.shape)}  OK")
    print(f"  track_masks:   {list(tm.shape)}  OK")

    # --- Value verification ---
    alive_prob = ta.sigmoid()
    print(f"  track_alive sigmoid mean (expect <0.5 initially): {alive_prob.mean().item():.4f}")
    assert alive_prob.mean().item() < 0.5, "Initial alive probability too high (bias init failed?)"

    assert tb.min().item() >= 0.0 and tb.max().item() <= 1.0, \
        f"track_boxes out of [0,1]: min={tb.min().item():.4f}, max={tb.max().item():.4f}"
    print(f"  track_boxes range: [{tb.min().item():.4f}, {tb.max().item():.4f}]  (expected [0,1])  OK")

    # --- Loss and gradient verification ---
    dummy_trainer6 = DummyTrainer(device, track_step)
    tgts6 = TAOTrainer._extract_target_chunk(dummy_trainer6, gpu_batch, c_start=0, c_end=T, max_t=t_max)

    track_loss_val = compute_track_loss(preds6, tgts6, track_step)
    assert torch.isfinite(track_loss_val), f"compute_track_loss returned non-finite: {track_loss_val}"
    print(f"  compute_track_loss: {track_loss_val.item():.4f}  (finite)  OK")

    track_loss_val.backward()
    grad_norms = [f.grad.norm().item() for f in feats6 if f.grad is not None]
    assert len(grad_norms) > 0, "No gradients flowed back through tracking loss!"
    print(f"  Gradient norms through feats: {[f'{g:.4f}' for g in grad_norms]}  OK")

    print("\n====================================================")
    print("[SUCCESS] SUCCESS: Stage 6 Tracking Verification PASSED!")
    print("====================================================")

if __name__ == "__main__":
    test_all_stages()
