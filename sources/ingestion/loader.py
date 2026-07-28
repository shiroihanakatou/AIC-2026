import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Import linh hoạt các loader thành phần trong cùng thư mục
try:
    from .clipfeatures_loader import load_clip_features
    from .keyframes_loader import load_keyframes_parallel as load_keyframes
    from .metadata_loader import load_metadata
    from .objects_loader import load_objects
except ImportError:
    from clipfeatures_loader import load_clip_features
    from keyframes_loader import load_keyframes_parallel as load_keyframes
    from metadata_loader import load_metadata
    from objects_loader import load_objects


def get_project_root() -> Path:
    """Xác định đường dẫn thư mục gốc của dự án (project/)."""
    return Path(__file__).resolve().parents[2]


def load_all_data(
    video_ids: Optional[Union[str, List[str]]] = None,
    base_dir: Optional[Union[str, Path]] = None,
    convert_float32: bool = True,
) -> Tuple[List[Dict[str, Any]], Tuple[np.ndarray, List[str]]]:
    """
    Hàm tổng hợp dữ liệu chạy SONG SONG ĐỒNG THỜI cả 4 loader thành phần:
    - Keyframes Loader (Đọc .jpg & .csv)
    - Metadata Loader (Đọc .json)
    - Objects Loader (Đọc .json keyframe)
    - CLIP Features Loader (Đọc .npy)

    Sau khi cả 4 loader nạp xong dữ liệu thô, tiến hành ghép nối dữ liệu phẳng cho CSDL.
    """
    root_dir = Path(base_dir) if base_dir else get_project_root()

    print("[INFO] Khởi chạy SONG SONG đồng thời 4 loader thành phần...")

    # 1. Tạo ThreadPoolExecutor với 4 workers cho 4 tác vụ nạp dữ liệu độc lập
    with ThreadPoolExecutor(max_workers=4) as top_executor:
        future_kf = top_executor.submit(load_keyframes, video_ids, root_dir)
        future_meta = top_executor.submit(load_metadata, video_ids, root_dir)
        future_objs = top_executor.submit(load_objects, video_ids, root_dir)
        future_feats = top_executor.submit(load_clip_features, video_ids, root_dir)

        # Chờ cả 4 luồng hoàn tất và lấy kết quả
        keyframes_data = future_kf.result()
        metadata_data = future_meta.result()
        objects_data = future_objs.result()
        clip_features_data = future_feats.result()

    print("[INFO] Đã hoàn tất đọc toàn bộ dữ liệu thô. Tiến hành ghép nối bản ghi...")

    # 2. Khởi tạo cấu trúc kết quả đầu ra
    metadata_records: List[Dict[str, Any]] = []
    vector_matrices_list: List[np.ndarray] = []
    global_frame_ids: List[str] = []

    # 3. Duyệt qua từng video để tổng hợp dữ liệu
    for v_id, kf_list in keyframes_data.items():
        v_meta = metadata_data.get(v_id, {})
        v_objs = objects_data.get(v_id, {})
        v_features = clip_features_data.get(v_id)

        has_features = v_features is not None and len(v_features) == len(kf_list)

        if not has_features and v_features is not None:
            print(
                f"[WARN] Số lượng keyframes ({len(kf_list)}) không khớp "
                f"với số hàng trong CLIP feature ({len(v_features)}) ở video '{v_id}'."
            )

        for idx, kf_info in enumerate(kf_list):
            frame_id = kf_info["frame_id"]
            global_id = f"{v_id}_{frame_id}"
            obj_info = v_objs.get(frame_id, {})

            # Bản ghi phẳng nạp vào SQLite
            record: Dict[str, Any] = {
                # Keyframe Identity
                "global_id": global_id,
                "video_id": v_id,
                "frame_id": frame_id,
                "n": int(kf_info.get("n", 0)),
                "pts_time": float(kf_info.get("pts_time", 0.0)),
                "fps": float(kf_info.get("fps", 0.0)),
                "frame_idx": int(kf_info.get("frame_idx", 0)),
                "image_path": kf_info.get("image_path"),
                # Video Metadata (JSON)
                "video_author": v_meta.get("author"),
                "video_channel_id": v_meta.get("channel_id"),
                "video_channel_url": v_meta.get("channel_url"),
                "video_description": v_meta.get("description"),
                "video_keywords": json.dumps(
                    v_meta.get("keywords", []), ensure_ascii=False
                ),
                "video_length": v_meta.get("length"),
                "video_publish_date": v_meta.get("publish_date"),
                "video_title": v_meta.get("title"),
                "video_watch_url": v_meta.get("watch_url"),
                # Objects Detection (JSON String)
                "detection_class_names": json.dumps(
                    obj_info.get("detection_class_names", []), ensure_ascii=False
                ),
                "detection_class_labels": json.dumps(
                    obj_info.get("detection_class_labels", []), ensure_ascii=False
                ),
                "detection_scores": json.dumps(
                    obj_info.get("detection_scores", [])
                ),
                "detection_boxes": json.dumps(
                    obj_info.get("detection_boxes", [])
                ),
                "detection_class_entities": json.dumps(
                    obj_info.get("detection_class_entities", []), ensure_ascii=False
                )
            }

            metadata_records.append(record)

            if has_features:
                global_frame_ids.append(global_id)

        if has_features:
            vector_matrices_list.append(v_features)

    # 4. Tách khối ma trận vector cho FAISS
    if vector_matrices_list:
        full_vector_matrix = np.vstack(vector_matrices_list)
        if convert_float32 and full_vector_matrix.dtype == np.float16:
            full_vector_matrix = full_vector_matrix.astype(np.float32)
    else:
        full_vector_matrix = np.empty((0, 512), dtype=np.float32)

    vector_matrix_data = (full_vector_matrix, global_frame_ids)

    return metadata_records, vector_matrix_data


if __name__ == "__main__":
    print("=== BẮT ĐẦU TỔNG HỢP DỮ LIỆU ĐỒNG THỜI ===")
    records, (matrix, frame_ids) = load_all_data()

    print("\n=== KẾT QUẢ XUẤT DỮ LIỆU ===")
    print(f"1. Tổng số bản ghi metadata: {len(records)}")
    print(f"2. Kích thước ma trận FAISS: {matrix.shape} ({matrix.dtype})")
    print(f"3. Số lượng Global Frame IDs: {len(frame_ids)}")