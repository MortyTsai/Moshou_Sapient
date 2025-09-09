# utils/video_utils.py
import subprocess
import json
import logging

def get_video_resolution(video_path: str) -> tuple[int, int] | None:
    """
    使用 ffprobe 獲取影片的寬度和高度。
    :param video_path: 影片檔案的路徑。
    :return: 一個包含 (width, height) 的元組，如果失敗則返回 None。
    """
    command = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'json',
        video_path
    ]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
        data = json.loads(result.stdout)
        if 'streams' in data and len(data['streams']) > 0:
            stream = data['streams'][0]
            width = stream.get('width')
            height = stream.get('height')
            if width and height:
                logging.info(f"[系統] 已成功偵測到影片解析度: {width}x{height}")
                return int(width), int(height)
        logging.error("[系統] ffprobe 輸出中未找到有效的影像串流資訊。")
        return None
    except FileNotFoundError:
        logging.critical("[嚴重錯誤] ffprobe 未安裝或未在系統 PATH 中。無法自動偵測影片解析度。")
        return None
    except subprocess.CalledProcessError as e:
        logging.error(f"[系統] 執行 ffprobe 時發生錯誤: {e.stderr}")
        return None
    except json.JSONDecodeError:
        logging.error("[系統] 解析 ffprobe 的 JSON 輸出時失敗。")
        return None