
import queue

class CUDAPrefetcher:
    """Overlaps GPU data processing with training to maximize GPU utilization."""
    def __init__(self, buffer, device, target_size=256, max_prefetch=2):
        self.buffer = buffer
        self.device = device
        self.target_size = target_size
        self.queue = queue.Queue(maxsize=max_prefetch)
        self.stream = torch.cuda.Stream(device=device) if device.type == 'cuda' else None
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while True:
            try:
                batch = self.buffer.get_batch()
                if self.stream is not None:
                    with torch.cuda.stream(self.stream):
                        batch_gpu = process_batch_on_gpu(batch, self.device, self.target_size)
                else:
                    batch_gpu = process_batch_on_gpu(batch, self.device, self.target_size)
                
                self.queue.put(batch_gpu)
            except Exception as e:
                print(f"Prefetcher worker error: {e}")
                import time
                time.sleep(1)

    def next(self):
        batch_gpu = self.queue.get()
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            for k, v in batch_gpu.items():
                if isinstance(v, torch.Tensor):
                    v.record_stream(torch.cuda.current_stream())
        return batch_gpu
