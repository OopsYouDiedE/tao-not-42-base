import os
import cv2
import numpy as np
import torch

def generate_synthetic_physical_video(B=1, T=12, H=256, W=256):
    """
    生成 3D 球在渐变背景上漂移靠近摄像机的高保真合成物理视频。
    模拟适当的几何深度、自我运动流、动态物体流和物体分割。
    """
    video = np.zeros((B, T, H, W, 3), dtype=np.uint8)
    depth = np.ones((B, T, H, W), dtype=np.float32) * 50.0  # 背景距离：50m
    seg = np.zeros((B, T, H, W), dtype=np.int16)
    flow = np.zeros((B, T, H, W, 2), dtype=np.float32)
    
    for b in range(B):
        for t in range(T):
            # 背景渐变
            img = np.zeros((H, W, 3), dtype=np.uint8)
            for y in range(H):
                img[y, :, 0] = int(120 + 40 * (y / H))  # 蓝色
                img[y, :, 1] = int(60 + 80 * (y / H))   # 绿色
                img[y, :, 2] = int(40 + 20 * (y / H))   # 红色
            
            # 模拟摄像机运动：向右平移
            cam_dx = 1.0 * t
            img = np.roll(img, -int(cam_dx), axis=1)
            bg_flow_x = -1.0
            
            # 渲染一个向近处漂移的 3D 红球
            center_x = int(60 + 10 * t)
            center_y = int(140 - 3 * t)
            radius = int(24 + 1.2 * t)
            
            cv2.circle(img, (center_x, center_y), radius, (0, 0, 255), -1)
            
            # 球越近，深度越小
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

def get_movi_e_or_fallback(npz_path="movi_e_sample_0000.npz", B=1, T=12, H=256, W=256):
    """
    动态加载从 Colab 导出的真实 Kubric MOVi-E 数据集样本，
    提取所有 7 个物理元素：视频、深度、分割、流、摄像机位置、
    摄像机四元数和 is_dynamic 参数。
    """
    if not os.path.exists(npz_path) and npz_path == "movi_e_sample_0000.npz" and os.path.exists("movi_e_sample.npz"):
        npz_path = "movi_e_sample.npz"
    elif not os.path.exists(npz_path) and npz_path == "movi_e_sample.npz" and os.path.exists("movi_e_sample_0000.npz"):
        npz_path = "movi_e_sample_0000.npz"

    if not os.path.exists(npz_path):
        print("====================================================")
        print("[警告] 数据缺失！未找到 movi_e_sample_0000.npz。")
        print("要获取数据，请在 Google Colab 中运行以下代码：")
        print("----------------------------------------------------")
        print("!pip install tensorflow-datasets imageio")
        print("import tensorflow_datasets as tfds")
        print("import numpy as np")
        print("ds = tfds.load('movi_e/256x256', data_dir='gs://kubric-public/tfds', split='test')")
        print("sample = next(iter(tfds.as_numpy(ds.take(1))))")
        print("np.savez_compressed('movi_e_sample_0000.npz', **sample)")
        print("----------------------------------------------------")
        print("然后将生成的 'movi_e_sample_0000.npz' 下载到项目根目录。")
        print("====================================================")

    if os.path.exists(npz_path):
        print(f"====================================================")
        print(f"[信息] 检测到真实的 MOVi-E 数据集样本：'{npz_path}'！")
        print(f"====================================================")
        try:
            data = np.load(npz_path, allow_pickle=True)
            v_np = data["video"]                  # 预期：[T, H, W, 3] 或 [B, T, H, W, 3]
            d_np = data["depth_m"] if "depth_m" in data else data["depth"]
            s_np = data["segmentation"]           # 预期：[T, H, W] 或 [B, T, H, W]
            f_np = data["forward_flow_px"].astype(np.float32) if "forward_flow_px" in data else data["forward_flow"].astype(np.float32)
            cp_np = data.get("cam_pos")           # 预期：[T, 3] 或 [B, T, 3]
            cq_np = data.get("cam_quat")          # 预期：[T, 4] 或 [B, T, 4]
            id_np = data.get("is_dynamic")        # 预期：[实例数] 或它们的列表
            cat_np = data.get("category")
            vel_np = data.get("velocities")
            avel_np = data.get("angular_velocities")
            vis_np = data.get("visibility")
            col_np = data.get("collisions")
            
            # 维度对齐助手
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
            
            # 对齐摄像机位置
            if cp_np is not None:
                if len(cp_np.shape) == 2: # [T, 3] -> [1, T, 3]
                    cp_np = np.expand_dims(cp_np, axis=0)
                if cp_np.shape[0] < B:
                    cp_np = np.repeat(cp_np, B, axis=0)
                cp_np = cp_np[:, :T]
            else:
                cp_np = np.zeros((B, T, 3), dtype=np.float32)
                
            # 对齐摄像机旋转四元数
            if cq_np is not None:
                if len(cq_np.shape) == 2: # [T, 4] -> [1, T, 4]
                    cq_np = np.expand_dims(cq_np, axis=0)
                if cq_np.shape[0] < B:
                    cq_np = np.repeat(cq_np, B, axis=0)
                cq_np = cq_np[:, :T]
            else:
                cq_np = np.zeros((B, T, 4), dtype=np.float32)
                cq_np[..., 0] = 1.0
                
            # 对齐实例列表
            def align_list(arr, default_gen):
                if arr is not None:
                    if isinstance(arr, np.ndarray) and (arr.ndim == 1 or arr.ndim == 3 or arr.ndim == 2):
                        return [arr for _ in range(B)]
                    elif isinstance(arr, list):
                        return arr
                return [default_gen() for _ in range(B)]
                
            id_np = align_list(id_np, lambda: np.array([True], dtype=bool))
            cat_np = align_list(cat_np, lambda: np.array([0], dtype=np.int32))
            vel_np = align_list(vel_np, lambda: np.zeros((1, T, 3), dtype=np.float32))
            avel_np = align_list(avel_np, lambda: np.zeros((1, T, 3), dtype=np.float32))
            vis_np = align_list(vis_np, lambda: np.ones((1, T), dtype=np.float32))
            col_np = align_list(col_np, lambda: np.zeros((0, 2), dtype=np.int32))
            
            # 空间缩放助手
            if v_np.shape[2] != H or v_np.shape[3] != W:
                print(f"正在将 MOVi-E 样本从 {v_np.shape[2]}x{v_np.shape[3]} 缩放到 {H}x{W}...")
                v_res, d_res, s_res, f_res = [], [], [], []
                for b in range(B):
                    v_res.append(np.stack([cv2.resize(v_np[b, t], (W, H)) for t in range(T)]))
                    d_res.append(np.stack([cv2.resize(d_np[b, t], (W, H), interpolation=cv2.INTER_NEAREST) for t in range(T)]))
                    s_res.append(np.stack([cv2.resize(s_np[b, t].astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int16) for t in range(T)]))
                    f_res.append(np.stack([cv2.resize(f_np[b, t], (W, H)) for t in range(T)]))
                v_np, d_np, s_np, f_np = np.stack(v_res), np.stack(d_res), np.stack(s_res), np.stack(f_res)
                
            print("成功加载并准备好真实的 Kubric MOVi-E 样本！")
            return v_np, d_np, s_np, f_np, cp_np, cq_np, id_np, cat_np, vel_np, avel_np, vis_np, col_np
        except Exception as e:
            print(f"从 npz 加载 MOVi-E 样本时出错：{e}")
            
    # 优雅退避在线下载 / 3D 球模拟
    print(f"未找到本地真实样本 '{npz_path}'。")
    video_path = "sample_video.mp4"
    if not os.path.exists(video_path):
        url = "https://github.com/intel-iot-devkit/sample-videos/raw/master/bottle-detection.mp4"
        print(f"尝试从以下地址下载示例视频：\n  {url}")
        try:
            import urllib.request
            urllib.request.urlretrieve(url, video_path)
            print("下载成功完成！")
        except Exception as e:
            print(f"下载在线视频样本失败：{e}")
            print("退避到高保真本地 3D 物理模拟器...")
            v_np, d_np, s_np, f_np = generate_synthetic_physical_video(B, T, H, W)
            cp_np = np.zeros((B, T, 3), dtype=np.float32)
            cq_np = np.zeros((B, T, 4), dtype=np.float32)
            cq_np[..., 0] = 1.0
            id_np = [np.array([True], dtype=bool) for _ in range(B)]
            cat_np = [np.array([0], dtype=np.int32) for _ in range(B)]
            vel_np = [np.zeros((1, T, 3), dtype=np.float32) for _ in range(B)]
            avel_np = [np.zeros((1, T, 3), dtype=np.float32) for _ in range(B)]
            vis_np = [np.ones((1, T), dtype=np.float32) for _ in range(B)]
            col_np = [np.zeros((0, 2), dtype=np.int32) for _ in range(B)]
            return v_np, d_np, s_np, f_np, cp_np, cq_np, id_np, cat_np, vel_np, avel_np, vis_np, col_np
            
    try:
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened() and len(frames) < T:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # 转换为灰度
            frame_resized = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (W, H))
            frames.append(frame_resized)
        cap.release()
        
        if len(frames) < T:
            while len(frames) < T:
                frames.append(frames[-1] if len(frames) > 0 else np.zeros((H, W, 3), dtype=np.uint8))
                
        video_np = np.stack(frames, axis=0)
        video_np = np.expand_dims(video_np, axis=0)
        if B > 1:
            video_np = np.repeat(video_np, B, axis=0)
            
        print(f"成功加载退避视频：'{video_path}'！")
        
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
        cat_np = [np.array([0], dtype=np.int32) for _ in range(B)]
        vel_np = [np.zeros((1, T, 3), dtype=np.float32) for _ in range(B)]
        avel_np = [np.zeros((1, T, 3), dtype=np.float32) for _ in range(B)]
        vis_np = [np.ones((1, T), dtype=np.float32) for _ in range(B)]
        col_np = [np.zeros((0, 2), dtype=np.int32) for _ in range(B)]
        
        return video_np, depth, seg, flow, cp_np, cq_np, id_np, cat_np, vel_np, avel_np, vis_np, col_np
    except Exception as e:
        print(f"读取备份视频出错：{e}")
        print("退避到高保真本地 3D 物理模拟器...")
        v_np, d_np, s_np, f_np = generate_synthetic_physical_video(B, T, H, W)
        cp_np = np.zeros((B, T, 3), dtype=np.float32)
        cq_np = np.zeros((B, T, 4), dtype=np.float32)
        cq_np[..., 0] = 1.0
        id_np = [np.array([True], dtype=bool) for _ in range(B)]
        cat_np = [np.array([0], dtype=np.int32) for _ in range(B)]
        vel_np = [np.zeros((1, T, 3), dtype=np.float32) for _ in range(B)]
        avel_np = [np.zeros((1, T, 3), dtype=np.float32) for _ in range(B)]
        vis_np = [np.ones((1, T), dtype=np.float32) for _ in range(B)]
        col_np = [np.zeros((0, 2), dtype=np.int32) for _ in range(B)]
        return v_np, d_np, s_np, f_np, cp_np, cq_np, id_np, cat_np, vel_np, avel_np, vis_np, col_np
