import torch
import time
import sys
from threading import Thread, Event

class VRAMOccluder:
    def __init__(self, target_reserve=10):
        self.target = target_reserve * 1024**2  # 转换为字节
        self.device = torch.device("cuda") if torch.cuda.is_available() else None
        self.alloc_blocks = []
        self.min_chunk = 2**18  # 256KB (最小分配单元)
        self.running = Event()
        self.monitor_thread = None
        self.total_vram = torch.cuda.get_device_properties(self.device).total_memory if self.device else 0

    def _auto_adjust_chunk(self, free_mem):
        """ 动态调整内存块大小策略 """
        if free_mem < 2 * self.target:  # 接近目标值时
            return max(self.min_chunk, free_mem - self.target)
        elif free_mem < 512 * 1024**2:  # <512MB时
            return min(2**20, free_mem//4)  # 最大1MB
        else:  # 充足时激进分配
            return min(2**27, free_mem//2)  # 最大128MB

    def _memory_worker(self):
        """ 核心内存分配线程 """
        while self.running.is_set():
            try:
                free, _ = torch.cuda.mem_get_info()
                if free <= self.target:
                    time.sleep(0.05)  # 短时休眠
                    continue
                
                chunk = self._auto_adjust_chunk(free)
                alloc_size = min(chunk, free - self.target)
                
                if alloc_size < self.min_chunk:
                    continue  # 避免过小分配

                # 梯度式分配策略
                block = torch.empty((alloc_size//4,), dtype=torch.float32, device=self.device)
                self.alloc_blocks.append(block)
                del block  # 仅保留张量引用
                
                # 实时状态显示
                sys.stdout.write(f"\r已占用: {(self.total_vram - free)/1024**2:.1f}MB | 剩余: {free/1024**2:.1f}MB")
                sys.stdout.flush()

            except torch.cuda.OutOfMemoryError:
                time.sleep(0.1)  # 显存竞争时退让
            except Exception as e:
                print(f"\n发生错误: {str(e)}")
                break

    def start(self):
        """ 启动监控线程 """
        if not self.device:
            print("CUDA不可用")
            return
            
        self.running.set()
        self.monitor_thread = Thread(target=self._memory_worker)
        self.monitor_thread.start()
        print(f"启动显存占用器，目标保留: {self.target/1024**2}MB")

    def stop(self):
        """ 安全停止并释放显存 """
        self.running.clear()
        if self.monitor_thread:
            self.monitor_thread.join()
        del self.alloc_blocks
        torch.cuda.empty_cache()
        print("\n显存已释放")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

# 使用示例
if __name__ == "__main__":
    occluder = VRAMOccluder(target_reserve=10)  # 保留10MB
    
    try:
        occluder.start()
        while True:  # 主线程保持活跃
            time.sleep(1)
    except KeyboardInterrupt:
        occluder.stop()
