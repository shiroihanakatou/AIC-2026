import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


def get_project_root() -> Path:
    """
    Xác định đường dẫn thư mục gốc của dự án (project/).
    File này nằm tại: project/sources/ingestion/metadata_loader.py
    Thư mục gốc là cấp cha thứ 3 (parents[2]).
    """
    return Path(__file__).resolve().parents[2]


def _read_single_json(json_file_path: Path) -> Tuple[str, Dict[str, Any]]:
    """
    Hàm phụ trợ: Đọc 1 file JSON metadata và trả về cặp (video_id, metadata_dict).
    Lấy tên file (không bao gồm đuôi .json) làm video_id.
    """
    video_id = json_file_path.stem
    with open(json_file_path, mode="r", encoding="utf-8") as f:
        data = json.load(f)
    return video_id, data


def load_metadata(
    video_ids: Optional[Union[str, List[str]]] = None,
    base_dir: Optional[Union[str, Path]] = None,
    max_workers: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Đọc dữ liệu các file JSON từ thư mục data/metadata/ và trả về dictionary
    chứa thông tin thuộc tính mô tả video.

    Args:
        video_ids (Optional[Union[str, List[str]]]):
            - Nếu là None: Tự động quét toàn bộ file .json trong thư mục metadata.
            - Nếu là str: Nhận vào 1 video_id cụ thể (ví dụ: "L01_V001").
            - Nếu là List[str]: Nhận vào danh sách các video_id cần load.
        base_dir (Optional[Union[str, Path]]):
            - Đường dẫn tới thư mục gốc `project/`.
            - Mặc định là None (tự động tính toán dựa vào vị trí script).
        max_workers (Optional[int]):
            - Số lượng luồng chạy song song khi đọc file.

    Returns:
        Dict[str, Dict[str, Any]]:
            Dictionary dạng {video_id: metadata_dict}.
    """
    root_dir = Path(base_dir) if base_dir else get_project_root()
    metadata_dir = root_dir / "data" / "metadata"

    if not metadata_dir.exists():
        raise FileNotFoundError(
            f"Thư mục chứa metadata không tồn tại tại: {metadata_dir}"
        )

    # 1. Xác định danh sách file JSON cần đọc
    target_files: List[Path] = []

    if video_ids is None:
        # Lấy tất cả file có đuôi .json (tự động bỏ qua các file như .gitkeep)
        target_files = list(metadata_dir.glob("*.json"))
    else:
        if isinstance(video_ids, str):
            v_list = [video_ids]
        else:
            v_list = list(video_ids)

        for v_id in v_list:
            json_path = metadata_dir / f"{v_id}.json"
            if json_path.exists():
                target_files.append(json_path)

    result: Dict[str, Dict[str, Any]] = {}

    if not target_files:
        return result

    # 2. Đọc file song song bằng ThreadPoolExecutor
    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 4) * 2)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(_read_single_json, f_path): f_path
            for f_path in target_files
        }

        for future in as_completed(future_to_file):
            v_id, meta_dict = future.result()
            result[v_id] = meta_dict

    return result


if __name__ == "__main__":
    print("Đang tiến hành load dữ liệu metadata...")
    metadata_map = load_metadata()
    print(f"Đã load thành công metadata của {len(metadata_map)} video.\n")

    for v_id, meta in metadata_map.items():
        print(f"- Video ID: {v_id}")
        for k, v in meta.items():
            print(f"    + {k}: {v}")
        print()