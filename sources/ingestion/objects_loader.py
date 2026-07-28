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


def _normalize_video_ids(video_ids: Optional[Union[str, List[str]]], objects_base_dir: Path) -> List[str]:
    if video_ids is None:
        return [directory.name for directory in objects_base_dir.iterdir() if directory.is_dir()]
    if isinstance(video_ids, str):
        return [video_ids]
    return list(video_ids)


def _load_single_video_objects(video_id: str, objects_base_dir: Path) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    video_dir = objects_base_dir / video_id
    if not video_dir.exists():
        return video_id, {}

    keyframe_objects_map: Dict[str, Dict[str, Any]] = {}
    json_files = sorted(
        video_dir.glob("*.json"),
        key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem,
    )

    for json_path in json_files:
        try:
            with open(json_path, mode="r", encoding="utf-8") as file_handle:
                keyframe_objects_map[json_path.stem] = json.load(file_handle)
        except Exception as error:
            print(f"[WARN] Lỗi khi đọc file '{json_path}': {error}")

    return video_id, keyframe_objects_map


def _load_objects_parallel(video_ids: List[str], objects_base_dir: Path, max_workers: int) -> Dict[str, Dict[str, Dict[str, Any]]]:
    result: Dict[str, Dict[str, Dict[str, Any]]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_video_id = {
            executor.submit(_load_single_video_objects, video_id, objects_base_dir): video_id
            for video_id in video_ids
        }

        for future in as_completed(future_to_video_id):
            video_id, object_map = future.result()
            if object_map:
                result[video_id] = object_map

    return result


def load_objects(
    video_ids: Optional[Union[str, List[str]]] = None,
    base_dir: Optional[Union[str, Path]] = None,
    max_workers: Optional[int] = None,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Đọc object detection JSON theo từng video và trả về map 2 cấp."""
    root_dir = _resolve_root_dir(base_dir)
    objects_base_dir = root_dir / "data" / "objects"

    if not objects_base_dir.exists():
        raise FileNotFoundError(f"Thư mục chứa objects không tồn tại tại: {objects_base_dir}")

    target_video_ids = _normalize_video_ids(video_ids, objects_base_dir)
    if not target_video_ids:
        return {}

    worker_count = max_workers or min(32, (os.cpu_count() or 4) * 2)
    return _load_objects_parallel(target_video_ids, objects_base_dir, worker_count)


def parse_detection_to_list(detection_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Chuyển cấu trúc mảng song song trong JSON detection thành danh sách object."""
    names = detection_dict.get("detection_class_names", [])
    labels = detection_dict.get("detection_class_labels", [])
    entities = detection_dict.get("detection_class_entities", [])
    scores = detection_dict.get("detection_scores", [])
    boxes = detection_dict.get("detection_boxes", [])

    parsed_objects: List[Dict[str, Any]] = []
    for index in range(len(names)):
        parsed_objects.append(
            {
                "class_name": names[index] if index < len(names) else "",
                "class_label": labels[index] if index < len(labels) else 0,
                "class_entity": entities[index] if index < len(entities) else "",
                "score": scores[index] if index < len(scores) else 0.0,
                "box": boxes[index] if index < len(boxes) else [],
            }
        )

    return parsed_objects


if __name__ == "__main__":
    print("Đang tiến hành load dữ liệu objects...")
    objects_data = load_objects()
    print(f"Đã load thành công dữ liệu objects của {len(objects_data)} video.\n")

    for video_id, keyframes in objects_data.items():
        print(f"- Video ID: {video_id} | Số keyframes có objects: {len(keyframes)}")

        first_frame_id = next(iter(keyframes.keys()), None)
        if first_frame_id:
            raw_object = keyframes[first_frame_id]
            print(f"  + Keyframe ID mẫu: {first_frame_id}")
            print(f"    * Số lượng objects phát hiện: {len(raw_object.get('detection_class_names', []))}")

            parsed_objects = parse_detection_to_list(raw_object)
            if parsed_objects:
                print(f"    * Object đầu tiên trong frame: {parsed_objects[0]}")
        print()