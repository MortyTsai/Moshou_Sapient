# src/moshousapient/__main__.py

"""
讓 moshousapient 套件可以透過 python -m moshousapient 指令執行的主入口。

這個檔案現在從 core 子套件中導入並執行主應用程式邏輯。
"""

from .core.main import main

if __name__ == "__main__":
    main()