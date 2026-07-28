import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


def get_project_root() -> Path:
    """Xác định đường dẫn thư mục gốc của dự án (project/)."""
    return Path(__file__).resolve().parents[2]


def _process_single_video(
    v_id: str,
    keyframes_base_dir: Path,
    map_base_dir: Path,
    root_dir: Path
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Hàm phụ trợ: Đọc và gộp dữ liệu cho duy nhất 1 video_id.
    Được thiết kế để chạy độc lập trên từng Thread.
    """
    video_keyframe_dir = keyframes_base_dir / v_id
    csv_file_path = map_base_dir / f"{v_id}.csv"

    if not video_keyframe_dir.exists():
        return v_id, []

    # 1. Đọc dữ liệu CSV
    csv_map: Dict[str, Dict[str, Any]] = {}
    if csv_file_path.exists():
        with open(csv_file_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            csv_map = {row["n"]: row for row in reader}

    # 2. Đọc và sắp xếp danh sách file ảnh
    image_files = sorted(
        video_keyframe_dir.glob("*.jpg"),
        key=lambda p: int(p.stem)
    )

    video_keyframes: List[Dict[str, Any]] = []

    # 3. Merge thông tin
    for img_path in image_files:
        frame_id = img_path.stem
        frame_n_key = str(int(frame_id))

        item: Dict[str, Any] = {
            "video_id": v_id,
            "frame_id": frame_id,
            "image_path": str(img_path.resolve()),
            "relative_path": str(img_path.relative_to(root_dir)),
        }

        if frame_n_key in csv_map:
            item.update(csv_map[frame_n_key])

        video_keyframes.append(item)

    return v_id, video_keyframes


def load_keyframes_parallel(
    video_ids: Optional[Union[str, List[str]]] = None,
    base_dir: Optional[Union[str, Path]] = None,
    max_workers: Optional[int] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Đọc dữ liệu keyframes và CSV song song sử dụng ThreadPoolExecutor.

    Args:
        video_ids: Danh sách các video_id cần load.
        base_dir: Thư mục gốc project.
        max_workers: Số luồng chạy song song (Mặc định: min(32, os.cpu_count() + 4)).
    """
    root_dir = Path(base_dir) if base_dir else get_project_root()
    keyframes_base_dir = root_dir / "data" / "keyframes"
    map_base_dir = root_dir / "data" / "map-keyframes"

    if video_ids is None:
        target_video_ids = [
            d.name for d in keyframes_base_dir.iterdir() if d.is_dir()
        ]
    elif isinstance(video_ids, str):
        target_video_ids = [video_ids]
    else:
        target_video_ids = list(video_ids)

    result: Dict[str, List[Dict[str, Any]]] = {}

    # Nếu số lượng worker không truyền vào, mặc định tự động tính dựa theo số luồng I/O
    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 4) * 2)

    # Khởi tạo ThreadPoolExecutor để quản lý các luồng đọc ghi
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Gửi tất cả các tác vụ xử lý từng video vào pool
        future_to_vid = {
            executor.submit(
                _process_single_video, v_id, keyframes_base_dir, map_base_dir, root_dir
            ): v_id
            for v_id in target_video_ids
        }

        # Thu thập kết quả ngay khi một Thread hoàn tất (as_completed)
        for future in as_completed(future_to_vid):
            v_id, video_keyframes = future.result()
            if video_keyframes:
                result[v_id] = video_keyframes

    return result


if __name__ == "__main__":
    print("Đang tiến hành load dữ liệu keyframes (Chạy song song)...")
    data = load_keyframes_parallel()
    print(f"Đã load thành công {len(data)} video.\n")

    for v_id, keyframes in data.items():
        print(f"- Video ID: {v_id} | Tổng số keyframes: {len(keyframes)}")