# src/moshousapient/processors/base_processor.py
import threading
from abc import ABC, abstractmethod

class BaseProcessor(ABC):
    def __init__(self, name: str):
        self.thread = None
        self.stop_event = threading.Event()
        self.name = name
    @abstractmethod
    def _target_func(self):
        pass
    def start(self):
        if self.thread is None or not self.thread.is_alive():
            self.stop_event.clear()
            self.thread = threading.Thread(target=self._target_func, name=self.name)
            self.thread.daemon = True
            self.thread.start()
    def stop(self):
        self.stop_event.set()
    def is_alive(self) -> bool:
        return self.thread is not None and self.thread.is_alive()