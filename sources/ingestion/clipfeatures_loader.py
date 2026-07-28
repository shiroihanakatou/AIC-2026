import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


def get_project_root() -> Path:
    """
    Xác định đường dẫn thư mục gốc của dự án (project/).
    File này nằm tại: project/sources/ingestion/clipfeatures_loader.py
    Thư mục gốc là cấp cha thứ 3 (parents[2]).
    """
    return Path(__file__).resolve().parents[2]


def _load_single_npy(
    npy_path: Path, mmap_mode: Optional[str] = None
) -> Tuple[str, np.ndarray]:
    """
    Hàm phụ trợ: Đọc 1 file .npy chứa ma trận CLIP features và trả về tuple (video_id, matrix).
    """
    video_id = npy_path.stem
    # Nạp mảng NumPy từ file nhị phân .npy
    matrix = np.load(npy_path, mmap_mode=mmap_mode)
    return video_id, matrix


def load_clip_features(
    video_ids: Optional[Union[str, List[str]]] = None,
    base_dir: Optional[Union[str, Path]] = None,
    mmap_mode: Optional[str] = None,
    max_workers: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Đọc dữ liệu CLIP features từ các file .npy trong project/data/clip-features/<video_id>.npy.

    Args:
        video_ids (Optional[Union[str, List[str]]]):
            - Nếu là None: Tự động quét toàn bộ file .npy trong thư mục clip-features.
            - Nếu là str: Nhận vào 1 video_id cụ thể (ví dụ: "L01_V001").
            - Nếu là List[str]: Nhận vào danh sách các video_id cần load.
        base_dir (Optional[Union[str, Path]]):
            - Đường dẫn tới thư mục gốc `project/`.
            - Mặc định là None (tự động tính toán dựa vào vị trí script).
        mmap_mode (Optional[str]):
            - Chế độ memory-mapping cho np.load ('r', 'r+', 'c'). 
              Truyền 'r' nếu muốn đọc ma trận trực tiếp từ đĩa mà không load toàn bộ vào RAM.
              Mặc định là None (nạp trực tiếp toàn bộ mảng vào RAM).
        max_workers (Optional[int]):
            - Số luồng chạy song song (Thread). Mặc định tự động tính theo số nhân CPU.

    Returns:
        Dict[str, np.ndarray]:
            Dictionary ánh xạ video_id tới ma trận NumPy 2 chiều kích thước (N, D).
    """
    root_dir = Path(base_dir) if base_dir else get_project_root()
    clip_dir = root_dir / "data" / "clip-features"

    if not clip_dir.exists():
        raise FileNotFoundError(
            f"Thư mục chứa clip-features không tồn tại tại: {clip_dir}"
        )

    # 1. Lọc danh sách file .npy cần đọc (bỏ qua .gitkeep và các file không liên quan)
    target_files: List[Path] = []

    if video_ids is None:
        target_files = list(clip_dir.glob("*.npy"))
    else:
        if isinstance(video_ids, str):
            v_list = [video_ids]
        else:
            v_list = list(video_ids)

        for v_id in v_list:
            npy_path = clip_dir / f"{v_id}.npy"
            if npy_path.exists():
                target_files.append(npy_path)

    result: Dict[str, np.ndarray] = {}

    if not target_files:
        return result

    # 2. Nạp dữ liệu song song bằng ThreadPoolExecutor
    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 4) * 2)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(_load_single_npy, f_path, mmap_mode): f_path
            for f_path in target_files
        }

        for future in as_completed(future_to_file):
            v_id, matrix = future.result()
            result[v_id] = matrix

    return result


if __name__ == "__main__":
    print("Đang tiến hành load dữ liệu CLIP features...")
    features_map = load_clip_features()
    print(f"Đã load thành công CLIP features của {len(features_map)} video.\n")

    for v_id, matrix in features_map.items():
        print(
            f"- Video ID: {v_id} | Shape: {matrix.shape} | "
            f"Dtype: {matrix.dtype} | Size: {matrix.nbytes / 1024:.2f} KB"
        )