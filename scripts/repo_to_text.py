# scripts/repo_to_text.py

"""
專案快照生成工具。

此腳本會自動掃描指定專案目錄，將符合條件的檔案內容合併到單一的文字檔案中，
並在檔案頂部生成專案的目錄結構樹。

主要用於快速打包專案上下文，以提供給大型語言模型 (LLM) 進行分析。
"""

import datetime
import os
import sys

# ==============================================================================
# ---                           組態設定                            ---
# ---          請在這裡修改所有參數，無需更動下方的程式碼            ---
# ==============================================================================

# 1. [可選] 要掃描的專案根目錄路徑
# 保持 'auto' 以自動偵測，因為我們的腳本在 'scripts' 目錄下。
START_PATH = r"auto"

# 2. [可選] 最終輸出的檔案名稱和位置
# 使用一個更具描述性的名稱，並明確指出它屬於 MoshouSapient。
OUTPUT_FILENAME = "moshousapient_snapshot.txt"

# 3. [可選] 要包含的檔案類型 (副檔名)
# 保持預設，這些都是與我們專案相關的文字檔類型。
INCLUDED_EXTENSIONS = [
    ".py",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".toml",
    ".html",
    ".gitignore",
    ".env.example",
]

# 4. [可選] 要完全忽略的目錄名稱
# 核心優化點：我們必須排除所有執行時生成的、非原始碼的目錄。
EXCLUDED_DIRS = {
    # --- 標準排除 ---
    "__pycache__",
    ".git",
    ".vscode",
    ".idea",
    "venv",
    ".venv",
    "dist",
    "build",
    ".ruff_cache",
    "moshousapient.egg-info",
    # --- MoshouSapient 特定排除 ---
    # 'data' 目錄包含日誌、資料庫、錄影等大量執行時數據，對理解程式碼邏輯無益，必須排除。
    "data",
    # 'models' 目錄包含大型二進位模型檔案，無法被讀取且無關程式碼邏輯。
    "models",
}

# 5. [可選] 要完全忽略的特定檔案名稱
# 排除此工具自身和生成的快照檔案。
EXCLUDED_FILES = {
    "moshousapient_snapshot.txt",
    "repo_to_text.py",
    "LICENSE",
}

# 6. [可選] 忽略大於此大小的檔案 (單位: KB) (設為 0 表示不限制)
# 保持預設，我們的原始碼檔案都不會很大。
MAX_FILE_SIZE_KB = 1024

# 7. [可選] 檔案標頭與頁尾格式
# 使用更簡潔的標頭，以減少 Token 消耗。
HEADER_FORMAT = "--- START OF FILE {path} ---\n\n"
FOOTER_FORMAT = "\n\n--- END OF FILE {path} ---\n" + "=" * 80 + "\n\n"

# ==============================================================================
# ---                          腳本主要邏輯                          ---
# ---                    (通常情況下無需修改以下內容)                    ---
# ==============================================================================


def get_project_root():
    """
    自動偵測專案根目錄。

    如果腳本位於 'scripts' 或 'script' 目錄中，則認定根目錄為上一層。
    否則，認定根目錄為腳本所在的當前目錄。

    Returns:
        str: 專案根目錄的絕對路徑。
    """
    if getattr(sys, "frozen", False):
        script_path = os.path.dirname(sys.executable)
    else:
        script_path = os.path.dirname(os.path.abspath(__file__))

    if os.path.basename(script_path).lower() in ["scripts", "script"]:
        return os.path.dirname(script_path)
    else:
        return script_path


def generate_tree_structure(start_path, local_excluded_files):
    """
    生成專案目錄的文字表示結構樹，並尊重排除規則。

    Args:
        start_path (str): 要生成結構樹的起始路徑。
        local_excluded_files (set): 需要排除的檔案名稱集合。

    Returns:
        list: 包含結構樹每一行的字串列表。
    """
    tree_lines = []

    def recurse(directory, prefix=""):
        """遞迴地建構目錄樹的內部輔助函式."""
        try:
            items = sorted(os.listdir(directory))
        except OSError:
            return

        filtered_items = []
        for item in items:
            full_path = os.path.join(directory, item)
            is_dir = os.path.isdir(full_path)

            if is_dir and item in EXCLUDED_DIRS:
                continue
            if not is_dir and item in local_excluded_files:
                continue

            if not is_dir and INCLUDED_EXTENSIONS:
                _, ext = os.path.splitext(item)
                if ext not in INCLUDED_EXTENSIONS and item not in INCLUDED_EXTENSIONS:
                    continue

            filtered_items.append((item, is_dir))

        pointers = ["├── "] * (len(filtered_items) - 1) + ["└── "]
        for pointer, (name, is_dir) in zip(pointers, filtered_items):
            tree_lines.append(f"{prefix}{pointer}{name}")
            if is_dir:
                extension = "│   " if pointer == "├── " else "    "
                recurse(os.path.join(directory, name), prefix + extension)

    abs_path = os.path.abspath(start_path)
    tree_lines.append(f"{os.path.basename(abs_path)}/")
    recurse(abs_path)
    return tree_lines


def create_project_snapshot():
    """
    主執行函式。

    協調結構樹的生成與檔案內容的彙編，最終生成完整的專案快照檔案。
    """
    if START_PATH.lower() == "auto":
        project_root = get_project_root()
    else:
        project_root = os.path.abspath(START_PATH)

    output_filepath = os.path.join(project_root, OUTPUT_FILENAME)

    print(f"專案根目錄已設定為: '{project_root}'")
    print(f"輸出檔案將儲存至: '{output_filepath}'")

    local_excluded_files = EXCLUDED_FILES.copy()
    local_excluded_files.add(OUTPUT_FILENAME)
    local_excluded_files.add(os.path.basename(__file__))

    files_processed_count = 0
    files_skipped_count = 0

    try:
        with open(output_filepath, "w", encoding="utf-8") as outfile:
            outfile.write(f"# 專案快照: {project_root}\n")
            outfile.write(f"# 生成時間: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            outfile.write("=" * 80 + "\n\n")

            print("正在生成專案結構樹...")
            tree_lines = generate_tree_structure(project_root, local_excluded_files)
            outfile.write("### Project Directory Tree ###\n\n")
            outfile.write("\n".join(tree_lines))
            outfile.write("\n\n" + "=" * 80 + "\n\n")
            print("結構樹生成完畢。")

            print("正在處理檔案內容...")
            for dirpath, dirnames, filenames in os.walk(project_root, topdown=True):
                dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]

                for filename in sorted(filenames):
                    if filename in local_excluded_files:
                        continue

                    full_path = os.path.join(dirpath, filename)
                    relative_path = os.path.relpath(full_path, project_root).replace("\\", "/")

                    if INCLUDED_EXTENSIONS:
                        _, ext = os.path.splitext(filename)
                        if ext not in INCLUDED_EXTENSIONS and filename not in INCLUDED_EXTENSIONS:
                            continue

                    try:
                        if MAX_FILE_SIZE_KB > 0:
                            file_size_kb = os.path.getsize(full_path) / 1024
                            if file_size_kb > MAX_FILE_SIZE_KB:
                                files_skipped_count += 1
                                continue

                        # 檔案讀取的核心邏輯
                        with open(full_path, "r", encoding="utf-8", errors="ignore") as infile:
                            content = infile.read()
                            outfile.write(HEADER_FORMAT.format(path=relative_path))
                            outfile.write(content)
                            outfile.write(FOOTER_FORMAT.format(path=relative_path))
                            files_processed_count += 1
                    except (IOError, UnicodeDecodeError, OSError) as e:
                        print(f"警告：跳過檔案 {relative_path}，原因: {e}")
                        files_skipped_count += 1

            print("檔案內容處理完畢。")

        print("\n" + "=" * 40)
        print("處理完成！")
        print(f"總共處理檔案數: {files_processed_count}")
        print(f"總共跳過檔案數: {files_skipped_count}")
        print(f"所有內容已儲存至: '{output_filepath}'")
        print("=" * 40)

    except Exception as e:
        print(f"\n發生嚴重錯誤: {e}")


if __name__ == "__main__":
    create_project_snapshot()
