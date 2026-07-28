import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


# =====================================================================
# Cấu hình Đường dẫn Tuyệt đối & System Path (Tránh lỗi Import)
# =====================================================================
CURRENT_FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE_PATH.parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_root_dir(base_dir: Optional[Union[str, Path]]) -> Path:
    return Path(base_dir) if base_dir else get_project_root()


def _normalize_video_ids(video_ids: Optional[Union[str, List[str]]], metadata_dir: Path) -> List[str]:
    if video_ids is None:
        return [path.stem for path in metadata_dir.glob("*.json")]
    if isinstance(video_ids, str):
        return [video_ids]
    return list(video_ids)


def _read_single_json(json_file_path: Path) -> Tuple[str, Dict[str, Any]]:
    video_id = json_file_path.stem
    with open(json_file_path, mode="r", encoding="utf-8") as file_handle:
        return video_id, json.load(file_handle)


def _load_metadata_files(target_files: List[Path], max_workers: int) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(_read_single_json, file_path): file_path
            for file_path in target_files
        }

        for future in as_completed(future_to_file):
            video_id, metadata = future.result()
            result[video_id] = metadata

    return result


def load_metadata(
    video_ids: Optional[Union[str, List[str]]] = None,
    base_dir: Optional[Union[str, Path]] = None,
    max_workers: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """Đọc JSON metadata và trả về dictionary {video_id: metadata_dict}."""
    root_dir = _resolve_root_dir(base_dir)
    metadata_dir = root_dir / "data" / "metadata"

    if not metadata_dir.exists():
        raise FileNotFoundError(f"Thư mục chứa metadata không tồn tại tại: {metadata_dir}")

    target_video_ids = _normalize_video_ids(video_ids, metadata_dir)
    target_files = [metadata_dir / f"{video_id}.json" for video_id in target_video_ids if (metadata_dir / f"{video_id}.json").exists()]

    if not target_files:
        return {}

    worker_count = max_workers or min(32, (os.cpu_count() or 4) * 2)
    return _load_metadata_files(target_files, worker_count)


if __name__ == "__main__":
    print("Đang tiến hành load dữ liệu metadata...")
    metadata_map = load_metadata()
    print(f"Đã load thành công metadata của {len(metadata_map)} video.\n")

    for video_id, metadata in metadata_map.items():
        print(f"- Video ID: {video_id}")
        for key, value in metadata.items():
            print(f"    + {key}: {value}")
        print()