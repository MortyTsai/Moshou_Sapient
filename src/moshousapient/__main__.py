# src/moshousapient/__main__.py
"""
讓 moshousapient 套件可以透過 `python -m moshousapient` 指令執行的主入口。

這個檔案現在從核心的應用協調器 (app orchestrator) 中導入並執行主應用邏輯。
"""

from .core.app_orchestrator import main

if __name__ == "__main__":
    main()
