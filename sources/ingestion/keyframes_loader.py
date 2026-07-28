import csv
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


def _normalize_video_ids(video_ids: Optional[Union[str, List[str]]], keyframes_base_dir: Path) -> List[str]:
    if video_ids is None:
        return [directory.name for directory in keyframes_base_dir.iterdir() if directory.is_dir()]
    if isinstance(video_ids, str):
        return [video_ids]
    return list(video_ids)


def _load_csv_map(csv_file_path: Path) -> Dict[str, Dict[str, Any]]:
    if not csv_file_path.exists():
        return {}

    with open(csv_file_path, mode="r", encoding="utf-8") as file_handle:
        reader = csv.DictReader(file_handle)
        return {row["n"]: row for row in reader}


def _build_keyframe_record(
    video_id: str,
    image_path: Path,
    csv_map: Dict[str, Dict[str, Any]],
    root_dir: Path,
) -> Dict[str, Any]:
    frame_id = image_path.stem
    record: Dict[str, Any] = {
        "video_id": video_id,
        "frame_id": frame_id,
        "image_path": str(image_path.resolve()),
        "relative_path": str(image_path.relative_to(root_dir)),
    }

    csv_row = csv_map.get(str(int(frame_id)))
    if csv_row:
        record.update(csv_row)

    return record


def _process_single_video(
    video_id: str,
    keyframes_base_dir: Path,
    map_base_dir: Path,
    root_dir: Path,
) -> Tuple[str, List[Dict[str, Any]]]:
    video_keyframe_dir = keyframes_base_dir / video_id
    csv_file_path = map_base_dir / f"{video_id}.csv"

    if not video_keyframe_dir.exists():
        return video_id, []

    csv_map = _load_csv_map(csv_file_path)
    image_files = sorted(video_keyframe_dir.glob("*.jpg"), key=lambda path: int(path.stem))
    keyframes = [
        _build_keyframe_record(video_id, image_path, csv_map, root_dir)
        for image_path in image_files
    ]

    return video_id, keyframes


def load_keyframes_parallel(
    video_ids: Optional[Union[str, List[str]]] = None,
    base_dir: Optional[Union[str, Path]] = None,
    max_workers: Optional[int] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Đọc dữ liệu keyframes và CSV song song bằng ThreadPoolExecutor."""
    root_dir = _resolve_root_dir(base_dir)
    keyframes_base_dir = root_dir / "data" / "keyframes"
    map_base_dir = root_dir / "data" / "map-keyframes"

    target_video_ids = _normalize_video_ids(video_ids, keyframes_base_dir)
    if not target_video_ids:
        return {}

    worker_count = max_workers or min(32, (os.cpu_count() or 4) * 2)
    result: Dict[str, List[Dict[str, Any]]] = {}

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_video_id = {
            executor.submit(_process_single_video, video_id, keyframes_base_dir, map_base_dir, root_dir): video_id
            for video_id in target_video_ids
        }

        for future in as_completed(future_to_video_id):
            video_id, keyframes = future.result()
            if keyframes:
                result[video_id] = keyframes

    return result


if __name__ == "__main__":
    print("Đang tiến hành load dữ liệu keyframes (Chạy song song)...")
    data = load_keyframes_parallel()
    print(f"Đã load thành công {len(data)} video.\n")

    for video_id, keyframes in data.items():
        print(f"- Video ID: {video_id} | Tổng số keyframes: {len(keyframes)}")