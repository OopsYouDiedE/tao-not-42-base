import time
import queue
import random
import threading
import atexit
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

try:
    import tensorflow as tf
    import tensorflow_datasets as tfds
except ImportError:
    tf = None
    tfds = None


def decode_uint16_range(encoded, value_range):
    encoded = encoded.astype(np.float32)
    minv, maxv = np.asarray(value_range, dtype=np.float32)
    return encoded / 65535.0 * (maxv - minv) + minv


def maybe_pin_memory(tensor):
    """将张量固定在分页锁定内存中，以便进行高速 CUDA PCIe 传输。"""
    if isinstance(tensor, torch.Tensor):
        return tensor.pin_memory()
    return tensor


def numpy_to_pinned_tensor(value):
    """将 numpy/标量值转换为张量，并在有用时固定它们。

    TFDS 嵌套特征（如事件/碰撞）可能以字典而非数组的形式到达。
    这些在这里有意不进行转换，因为它们不被当前的训练目标消耗，
    并且尝试对字典执行 torch.from_numpy 是从 Colab 报告的崩溃原因。
    """
    if value is None or isinstance(value, dict):
        return None
    if isinstance(value, torch.Tensor):
        return maybe_pin_memory(value)
    if isinstance(value, np.ndarray):
        return maybe_pin_memory(torch.from_numpy(value))
    if np.isscalar(value):
        return maybe_pin_memory(torch.as_tensor(value))
    return None

# =====================================================================

# 5. 数据流加载 (Data Loader & Pipeline)
# =====================================================================


class AsyncDataBuffer:
    def __init__(self, split="train", max_buffer_size=64, batch_size=16, wait_timeout_sec=180, offline_path=None):
        self.split = split
        self.offline_path = offline_path
        self.max_buffer_size = max_buffer_size
        self.batch_size = batch_size
        self.wait_timeout_sec = wait_timeout_sec
        self.buffer = deque(maxlen=max_buffer_size)
        self.lock = threading.Lock()
        self.has_data = threading.Condition(self.lock)
        self.error = None
        self.num_fetched = 0
        self._last_wait_log = 0.0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._fetch_loop, daemon=True)
        self.thread.start()
        atexit.register(self.stop)

    def stop(self):
        self.stop_event.set()
        with self.lock:
            self.has_data.notify_all()
        thread = getattr(self, "thread", None)
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)

    def _fetch_loop(self):
        try:
            self._fetch_loop_impl()
        except Exception as e:
            with self.lock:
                self.error = e
                self.has_data.notify_all()
            print(f"[DataBuffer] 获取线程失败: {type(e).__name__}: {e}", flush=True)

    def _fetch_loop_impl(self):
        if self.offline_path is not None:
            print(f"[DataBuffer] 正在从离线文件 {self.offline_path} 读取数据 ...", flush=True)
            try:
                data = np.load(self.offline_path, allow_pickle=True)
                item = dict(data)
                
                def get_nested(obj):
                    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.ndim == 0:
                        return obj.item()
                    return obj
                
                # Unpack object arrays if needed
                for k in item:
                    item[k] = get_nested(item[k])
                    
            except Exception as e:
                raise RuntimeError(f"无法加载离线文件 {self.offline_path}: {e}")
                
            while not self.stop_event.is_set():
                p_item = {}
                p_item["video"] = numpy_to_pinned_tensor(item.get("video"))
                
                # Check for nested camera structures from tfds
                if "camera" in item and isinstance(item["camera"], dict):
                    p_item["cam_pos"] = numpy_to_pinned_tensor(item["camera"].get("positions"))
                    p_item["cam_quat"] = numpy_to_pinned_tensor(item["camera"].get("quaternions"))
                    p_item["camera_focal_length"] = numpy_to_pinned_tensor(item["camera"].get("focal_length", 0.7))
                    p_item["camera_sensor_width"] = numpy_to_pinned_tensor(item["camera"].get("sensor_width", 36.0))
                else:
                    p_item["cam_pos"] = numpy_to_pinned_tensor(item.get("cam_pos"))
                    p_item["cam_quat"] = numpy_to_pinned_tensor(item.get("cam_quat"))
                    p_item["camera_focal_length"] = numpy_to_pinned_tensor(item.get("camera_focal_length", np.array(0.7, dtype=np.float32)))
                    p_item["camera_sensor_width"] = numpy_to_pinned_tensor(item.get("camera_sensor_width", np.array(36.0, dtype=np.float32)))

                if "segmentations" in item:
                    p_item["segmentation"] = numpy_to_pinned_tensor(item["segmentations"][..., 0])
                else:
                    p_item["segmentation"] = numpy_to_pinned_tensor(item.get("segmentation"))
                    
                if "depth_m" in item:
                    p_item["depth"] = numpy_to_pinned_tensor(item["depth_m"])
                elif "depth" in item and item["depth"].ndim > 3:
                    p_item["depth"] = numpy_to_pinned_tensor(item["depth"][..., 0])
                else:
                    p_item["depth"] = numpy_to_pinned_tensor(item.get("depth"))

                if "metadata" in item and isinstance(item["metadata"], dict):
                    p_item["depth_range"] = numpy_to_pinned_tensor(item["metadata"].get("depth_range", [0.0, 100.0]))
                    p_item["forward_flow_range"] = numpy_to_pinned_tensor(item["metadata"].get("forward_flow_range", [0.0, 10.0]))
                else:
                    p_item["depth_range"] = numpy_to_pinned_tensor(item.get("depth_range", np.array([0.0, 100.0], dtype=np.float32)))
                    p_item["forward_flow_range"] = numpy_to_pinned_tensor(item.get("forward_flow_range", np.array([0.0, 10.0], dtype=np.float32)))

                if "instances" in item and isinstance(item["instances"], dict):
                    insts = item["instances"]
                    if "is_dynamic" in insts: p_item["is_dynamic"] = numpy_to_pinned_tensor(insts["is_dynamic"])
                    if "category" in insts: p_item["category"] = numpy_to_pinned_tensor(insts["category"])
                    if "velocities" in insts: p_item["velocities"] = numpy_to_pinned_tensor(insts["velocities"])
                    if "angular_velocities" in insts: p_item["angular_velocities"] = numpy_to_pinned_tensor(insts["angular_velocities"])
                    if "visibility" in insts: p_item["visibility"] = numpy_to_pinned_tensor(insts["visibility"])
                else:
                    if "is_dynamic" in item: p_item["is_dynamic"] = numpy_to_pinned_tensor(item["is_dynamic"])
                    if "category" in item: p_item["category"] = numpy_to_pinned_tensor(item["category"])
                    if "velocities" in item: p_item["velocities"] = numpy_to_pinned_tensor(item["velocities"])
                    if "angular_velocities" in item: p_item["angular_velocities"] = numpy_to_pinned_tensor(item["angular_velocities"])
                    if "visibility" in item: p_item["visibility"] = numpy_to_pinned_tensor(item["visibility"])

                if "forward_flow_px" in item:
                    p_item["forward_flow"] = numpy_to_pinned_tensor(item["forward_flow_px"])
                else:
                    p_item["forward_flow"] = numpy_to_pinned_tensor(item.get("forward_flow"))

                with self.lock:
                    self.buffer.append(p_item)
                    self.num_fetched += 1
                    self.has_data.notify_all()
                time.sleep(0.01)
            return

        try:
            import tensorflow_datasets as tfds
            import tensorflow as tf
        except ImportError:
            raise ImportError("缺少 tensorflow_datasets 或 tensorflow 依赖，且未指定 offline_path。")

        print(f"[DataBuffer] 正在从 gs://kubric-public/tfds 加载 TFDS movi_e split='{self.split}' ...", flush=True)
        ds = tfds.load("movi_e", data_dir="gs://kubric-public/tfds", split=self.split,
                       read_config=tfds.ReadConfig(interleave_cycle_length=16)).repeat()

        def map_fn(x):
            insts = x.get("instances", {})
            return {
                "video": x["video"], "segmentations": x["segmentations"], "depth": x["depth"],
                "forward_flow": x["forward_flow"], "cam_pos": x["camera"]["positions"],
                "cam_quat": x["camera"]["quaternions"],
                "depth_range": x["metadata"]["depth_range"],
                "forward_flow_range": x["metadata"]["forward_flow_range"],
                "camera_focal_length": x["camera"]["focal_length"],
                "camera_sensor_width": x["camera"]["sensor_width"],
                **({"is_dynamic": insts["is_dynamic"]} if "is_dynamic" in insts else {}),
                **({"category": insts["category"]} if "category" in insts else {}),
                **({"velocities": insts["velocities"]} if "velocities" in insts else {}),
                **({"angular_velocities": insts["angular_velocities"]} if "angular_velocities" in insts else {}),
                **({"visibility": insts["visibility"]} if "visibility" in insts else {}),
                # 不要在热训练路径中实例化事件/碰撞。
                # TFDS 将此特征作为嵌套字典返回，而不是 ndarray，并且
                # 当前的损失函数不消耗它。在这里加载它导致
                # 预取线程在训练开始前崩溃。
            }

        ds = ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE).prefetch(
            tf.data.AUTOTUNE)
        print("[DataBuffer] TFDS 流水线就绪；正在填充异步缓冲区 ...", flush=True)

        for item in tfds.as_numpy(ds):
            if self.stop_event.is_set():
                break
            p_item = {k: numpy_to_pinned_tensor(item[k_i]) for k, k_i in [
                ("video", "video"), ("cam_pos", "cam_pos"), ("cam_quat", "cam_quat")]}
            p_item["segmentation"] = numpy_to_pinned_tensor(item["segmentations"][..., 0])
            p_item["depth"] = numpy_to_pinned_tensor(item["depth"][..., 0])
            p_item["depth_range"] = numpy_to_pinned_tensor(item["depth_range"])
            p_item["forward_flow_range"] = numpy_to_pinned_tensor(item["forward_flow_range"])
            p_item["camera_focal_length"] = numpy_to_pinned_tensor(item["camera_focal_length"])
            p_item["camera_sensor_width"] = numpy_to_pinned_tensor(item["camera_sensor_width"])

            if "is_dynamic" in item:
                p_item["is_dynamic"] = numpy_to_pinned_tensor(item["is_dynamic"])
            if "category" in item:
                p_item["category"] = numpy_to_pinned_tensor(item["category"])
            if "velocities" in item:
                p_item["velocities"] = numpy_to_pinned_tensor(item["velocities"])
            if "angular_velocities" in item:
                p_item["angular_velocities"] = numpy_to_pinned_tensor(item["angular_velocities"])
            if "visibility" in item:
                p_item["visibility"] = numpy_to_pinned_tensor(item["visibility"])

            p_item["forward_flow"] = numpy_to_pinned_tensor(item["forward_flow"])

            with self.lock:
                self.buffer.append(p_item)
                self.num_fetched += 1
                if self.num_fetched == 1 or self.num_fetched % 16 == 0:
                    print(f"[DataBuffer] 已缓冲 {min(len(self.buffer), self.max_buffer_size)}/{self.batch_size} 个样本；总计已获取={self.num_fetched}", flush=True)
                self.has_data.notify_all()

    def get_batch(self):
        start_wait = time.time()
        with self.lock:
            while len(self.buffer) < self.batch_size:
                if self.error is not None:
                    raise RuntimeError("AsyncDataBuffer 获取线程失败") from self.error

                now = time.time()
                if now - self._last_wait_log >= 10.0:
                    print(f"[DataBuffer] 正在等待样本: {len(self.buffer)}/{self.batch_size}", flush=True)
                    self._last_wait_log = now

                if now - start_wait > self.wait_timeout_sec:
                    raise TimeoutError(
                        f"等待数据缓冲区 {self.wait_timeout_sec} 秒后超时 "
                        f"({len(self.buffer)}/{self.batch_size} 个样本就绪)。"
                    )

                self.has_data.wait(timeout=2.0)
                if len(self.buffer) == 0 and not IN_COLAB:
                    return None
            batch = random.sample(self.buffer, self.batch_size)

        keys = [
            "video", "segmentation", "depth", "forward_flow", "cam_pos",
            "cam_quat", "is_dynamic", "category", "velocities", "angular_velocities",
            "visibility", "depth_range", "forward_flow_range",
            "camera_focal_length", "camera_sensor_width",
        ]
        return {k: [i.get(k) for i in batch] for k in keys}


from utils.label_generator import process_batch_on_gpu


class CUDAPrefetcher:
    def __init__(self, buffer, device, target_size=256, wait_timeout_sec=180):
        self.buffer = buffer
        self.device = device
        self.target_size = target_size
        self.wait_timeout_sec = wait_timeout_sec
        self.queue = queue.Queue(maxsize=4)
        self.error = None
        self._last_wait_log = 0.0
        self.stop_event = threading.Event()
        self.stream = torch.cuda.Stream(device=device)
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        atexit.register(self.stop)

    def stop(self):
        self.stop_event.set()
        thread = getattr(self, "thread", None)
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)

    def _worker(self):
        while not self.stop_event.is_set():
            batch = self.buffer.get_batch()
            if batch is None:
                time.sleep(1)
                continue
            try:
                # 显式同步，强制确保 pinned memory 异步拷贝已安全落入 GPU 显存，彻底切断 scatter_reduce_ 异步竞争诱发的非法访问死锁
                torch.cuda.synchronize(self.device)
                batch_gpu = process_batch_on_gpu(
                    batch, self.device, self.target_size)
                torch.cuda.synchronize(self.device)
                while not self.stop_event.is_set():
                    try:
                        self.queue.put(batch_gpu, timeout=1.0)
                        break
                    except queue.Full:
                        continue
            except Exception as e:
                self.error = e
                try:
                    self.queue.put_nowait(e)
                except queue.Full:
                    pass
                print(f"[Prefetcher] 工作线程失败: {type(e).__name__}: {e}", flush=True)
                return

    def next(self):
        start_wait = time.time()
        while True:
            if self.error is not None and self.queue.empty():
                raise RuntimeError("CUDAPrefetcher 工作线程失败") from self.error
            try:
                batch = self.queue.get(timeout=2.0)
                break
            except queue.Empty:
                now = time.time()
                if now - self._last_wait_log >= 10.0:
                    print("[Prefetcher] 正在等待处理好的 GPU 批次 ...", flush=True)
                    self._last_wait_log = now
                if now - start_wait > self.wait_timeout_sec:
                    raise TimeoutError(
                        f"等待处理好的批次 {self.wait_timeout_sec} 秒后超时。"
                    )

        if isinstance(batch, Exception):
            raise RuntimeError("CUDAPrefetcher 工作线程失败") from batch

        torch.cuda.current_stream().wait_stream(self.stream)

        def record(obj):
            if isinstance(obj, torch.Tensor):
                obj.record_stream(torch.cuda.current_stream())
            elif isinstance(obj, (list, tuple)):
                for x in obj:
                    record(x)
            elif isinstance(obj, dict):
                for x in obj.values():
                    record(x)

        record(batch)
        return batch

# =====================================================================
