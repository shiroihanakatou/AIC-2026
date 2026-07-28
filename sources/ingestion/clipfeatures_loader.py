import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


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


def _normalize_video_ids(video_ids: Optional[Union[str, List[str]]], clip_dir: Path) -> List[str]:
    if video_ids is None:
        return [path.stem for path in clip_dir.glob("*.npy")]
    if isinstance(video_ids, str):
        return [video_ids]
    return list(video_ids)


def _load_single_npy(npy_path: Path, mmap_mode: Optional[str] = None) -> Tuple[str, np.ndarray]:
    return npy_path.stem, np.load(npy_path, mmap_mode=mmap_mode)


def _load_features_parallel(
    target_files: List[Path],
    mmap_mode: Optional[str],
    max_workers: int,
) -> Dict[str, np.ndarray]:
    result: Dict[str, np.ndarray] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(_load_single_npy, file_path, mmap_mode): file_path
            for file_path in target_files
        }

        for future in as_completed(future_to_file):
            video_id, matrix = future.result()
            result[video_id] = matrix

    return result


def load_clip_features(
    video_ids: Optional[Union[str, List[str]]] = None,
    base_dir: Optional[Union[str, Path]] = None,
    mmap_mode: Optional[str] = None,
    max_workers: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Đọc CLIP feature .npy theo từng video và trả về map {video_id: matrix}."""
    root_dir = _resolve_root_dir(base_dir)
    clip_dir = root_dir / "data" / "clip-features"

    if not clip_dir.exists():
        raise FileNotFoundError(f"Thư mục chứa clip-features không tồn tại tại: {clip_dir}")

    target_video_ids = _normalize_video_ids(video_ids, clip_dir)
    target_files = [clip_dir / f"{video_id}.npy" for video_id in target_video_ids if (clip_dir / f"{video_id}.npy").exists()]

    if not target_files:
        return {}

    worker_count = max_workers or min(32, (os.cpu_count() or 4) * 2)
    return _load_features_parallel(target_files, mmap_mode, worker_count)


if __name__ == "__main__":
    print("Đang tiến hành load dữ liệu CLIP features...")
    features_map = load_clip_features()
    print(f"Đã load thành công CLIP features của {len(features_map)} video.\n")

    for video_id, matrix in features_map.items():
        print(
            f"- Video ID: {video_id} | Shape: {matrix.shape} | "
            f"Dtype: {matrix.dtype} | Size: {matrix.nbytes / 1024:.2f} KB"
        )