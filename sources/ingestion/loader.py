import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


# =====================================================================
# Cấu hình Đường dẫn Tuyệt đối & System Path (Tránh lỗi Import)
# =====================================================================
CURRENT_FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE_PATH.parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sources.ingestion.clipfeatures_loader import load_clip_features
from sources.ingestion.keyframes_loader import load_keyframes_parallel as load_keyframes
from sources.ingestion.metadata_loader import load_metadata
from sources.ingestion.objects_loader import load_objects


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_root_dir(base_dir: Optional[Union[str, Path]]) -> Path:
    return Path(base_dir) if base_dir else get_project_root()


def _load_sources_in_parallel(
    video_ids: Optional[Union[str, List[str]]],
    root_dir: Path,
) -> tuple[
    Dict[str, List[Dict[str, Any]]],
    Dict[str, Dict[str, Any]],
    Dict[str, Dict[str, Dict[str, Any]]],
    Dict[str, np.ndarray],
]:
    print("[INFO] Khởi chạy SONG SONG đồng thời 4 loader thành phần...")
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_keyframes = executor.submit(load_keyframes, video_ids, root_dir)
        future_metadata = executor.submit(load_metadata, video_ids, root_dir)
        future_objects = executor.submit(load_objects, video_ids, root_dir)
        future_features = executor.submit(load_clip_features, video_ids, root_dir)

        return (
            future_keyframes.result(),
            future_metadata.result(),
            future_objects.result(),
            future_features.result(),
        )


def _build_record(
    video_id: str,
    keyframe_info: Dict[str, Any],
    video_metadata: Dict[str, Any],
    detection_info: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "global_id": f"{video_id}_{keyframe_info['frame_id']}",
        "video_id": video_id,
        "frame_id": keyframe_info["frame_id"],
        "n": int(keyframe_info.get("n", 0)),
        "pts_time": float(keyframe_info.get("pts_time", 0.0)),
        "fps": float(keyframe_info.get("fps", 0.0)),
        "frame_idx": int(keyframe_info.get("frame_idx", 0)),
        "image_path": keyframe_info.get("image_path"),
        "video_author": video_metadata.get("author"),
        "video_channel_id": video_metadata.get("channel_id"),
        "video_channel_url": video_metadata.get("channel_url"),
        "video_description": video_metadata.get("description"),
        "video_keywords": json.dumps(video_metadata.get("keywords", []), ensure_ascii=False),
        "video_length": video_metadata.get("length"),
        "video_publish_date": video_metadata.get("publish_date"),
        "video_title": video_metadata.get("title"),
        "video_watch_url": video_metadata.get("watch_url"),
        "detection_class_names": json.dumps(detection_info.get("detection_class_names", []), ensure_ascii=False),
        "detection_class_labels": json.dumps(detection_info.get("detection_class_labels", []), ensure_ascii=False),
        "detection_scores": json.dumps(detection_info.get("detection_scores", [])),
        "detection_boxes": json.dumps(detection_info.get("detection_boxes", [])),
        "detection_class_entities": json.dumps(detection_info.get("detection_class_entities", []), ensure_ascii=False),
    }


def _collect_video_records(
    keyframes_data: Dict[str, List[Dict[str, Any]]],
    metadata_data: Dict[str, Dict[str, Any]],
    objects_data: Dict[str, Dict[str, Dict[str, Any]]],
    clip_features_data: Dict[str, np.ndarray],
    convert_float32: bool,
) -> Tuple[List[Dict[str, Any]], Tuple[np.ndarray, List[str]]]:
    metadata_records: List[Dict[str, Any]] = []
    vector_matrices: List[np.ndarray] = []
    global_frame_ids: List[str] = []

    for video_id, keyframe_list in keyframes_data.items():
        video_metadata = metadata_data.get(video_id, {})
        video_objects = objects_data.get(video_id, {})
        video_features = clip_features_data.get(video_id)
        has_features = video_features is not None and len(video_features) == len(keyframe_list)

        if not has_features and video_features is not None:
            print(
                f"[WARN] Số lượng keyframes ({len(keyframe_list)}) không khớp "
                f"với số hàng trong CLIP feature ({len(video_features)}) ở video '{video_id}'."
            )

        for keyframe_info in keyframe_list:
            frame_id = keyframe_info["frame_id"]
            detection_info = video_objects.get(frame_id, {})
            metadata_records.append(
                _build_record(video_id, keyframe_info, video_metadata, detection_info)
            )

            if has_features:
                global_frame_ids.append(f"{video_id}_{frame_id}")

        if has_features:
            vector_matrices.append(video_features)

    if vector_matrices:
        full_vector_matrix = np.vstack(vector_matrices)
        if convert_float32 and full_vector_matrix.dtype == np.float16:
            full_vector_matrix = full_vector_matrix.astype(np.float32)
    else:
        full_vector_matrix = np.empty((0, 512), dtype=np.float32)

    return metadata_records, (full_vector_matrix, global_frame_ids)


def load_all_data(
    video_ids: Optional[Union[str, List[str]]] = None,
    base_dir: Optional[Union[str, Path]] = None,
    convert_float32: bool = True,
) -> Tuple[List[Dict[str, Any]], Tuple[np.ndarray, List[str]]]:
    """Tổng hợp dữ liệu từ 4 loader thành phần và ghép thành bản ghi phẳng."""
    root_dir = _resolve_root_dir(base_dir)
    keyframes_data, metadata_data, objects_data, clip_features_data = _load_sources_in_parallel(
        video_ids, root_dir
    )
    print("[INFO] Đã hoàn tất đọc toàn bộ dữ liệu thô. Tiến hành ghép nối bản ghi...")
    return _collect_video_records(
        keyframes_data,
        metadata_data,
        objects_data,
        clip_features_data,
        convert_float32,
    )


if __name__ == "__main__":
    print("=== BẮT ĐẦU TỔNG HỢP DỮ LIỆU ĐỒNG THỜI ===")
    records, (matrix, frame_ids) = load_all_data()

    print("\n=== KẾT QUẢ XUẤT DỮ LIỆU ===")
    print(f"1. Tổng số bản ghi metadata: {len(records)}")
    print(f"2. Kích thước ma trận FAISS: {matrix.shape} ({matrix.dtype})")
    print(f"3. Số lượng Global Frame IDs: {len(frame_ids)}")