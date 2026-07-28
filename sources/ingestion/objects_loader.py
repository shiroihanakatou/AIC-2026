import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


def get_project_root() -> Path:
    """
    Xác định đường dẫn thư mục gốc của dự án (project/).
    File này nằm tại: project/sources/ingestion/objects_loader.py
    Thư mục gốc là cấp cha thứ 3 (parents[2]).
    """
    return Path(__file__).resolve().parents[2]


def _process_video_objects(
    v_id: str, objects_base_dir: Path
) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    """
    Hàm phụ trợ: Đọc toàn bộ các file JSON keyframe thuộc về 1 video_id.
    Chạy độc lập trên từng Thread.
    
    Returns:
        Tuple[str, Dict[str, Dict[str, Any]]]: 
            (video_id, {keyframe_id: raw_detection_dict})
    """
    video_dir = objects_base_dir / v_id
    if not video_dir.exists():
        return v_id, {}

    keyframe_objects_map: Dict[str, Dict[str, Any]] = {}

    # Lấy toàn bộ file JSON trong thư mục video và sắp xếp theo thứ tự số của frame_id
    json_files = sorted(
        video_dir.glob("*.json"),
        key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem,
    )

    for json_path in json_files:
        keyframe_id = json_path.stem  # Ví dụ: "0001"
        try:
            with open(json_path, mode="r", encoding="utf-8") as f:
                data = json.load(f)
            keyframe_objects_map[keyframe_id] = data
        except Exception as e:
            print(f"[WARN] Lỗi khi đọc file '{json_path}': {e}")

    return v_id, keyframe_objects_map


def load_objects(
    video_ids: Optional[Union[str, List[str]]] = None,
    base_dir: Optional[Union[str, Path]] = None,
    max_workers: Optional[int] = None,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Đọc dữ liệu object detection từ các file JSON trong project/data/objects/<video_id>/<keyframe_id>.json.

    Args:
        video_ids (Optional[Union[str, List[str]]]):
            - Nếu là None: Tự động quét toàn bộ các video_id trong thư mục objects.
            - Nếu là str: Nhận vào 1 video_id cụ thể (ví dụ: "L01_V001").
            - Nếu là List[str]: Nhận vào danh sách các video_id cần load.
        base_dir (Optional[Union[str, Path]]):
            - Đường dẫn tới thư mục gốc `project/`.
            - Mặc định là None (tự động tính toán dựa vào vị trí script).
        max_workers (Optional[int]):
            - Số luồng chạy song song (Thread). Mặc định tự động tính theo CPU.

    Returns:
        Dict[str, Dict[str, Dict[str, Any]]]:
            Cấu trúc 2 cấp: { video_id: { keyframe_id: detection_data_dict } }
    """
    root_dir = Path(base_dir) if base_dir else get_project_root()
    objects_base_dir = root_dir / "data" / "objects"

    if not objects_base_dir.exists():
        raise FileNotFoundError(
            f"Thư mục chứa objects không tồn tại tại: {objects_base_dir}"
        )

    # 1. Chuẩn hóa danh sách video_id cần load
    if video_ids is None:
        target_video_ids = [
            d.name for d in objects_base_dir.iterdir() if d.is_dir()
        ]
    elif isinstance(video_ids, str):
        target_video_ids = [video_ids]
    else:
        target_video_ids = list(video_ids)

    result: Dict[str, Dict[str, Dict[str, Any]]] = {}

    if not target_video_ids:
        return result

    # 2. Đọc dữ liệu song song theo từng video_id
    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 4) * 2)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_vid = {
            executor.submit(_process_video_objects, v_id, objects_base_dir): v_id
            for v_id in target_video_ids
        }

        for future in as_completed(future_to_vid):
            v_id, keyframe_map = future.result()
            if keyframe_map:
                result[v_id] = keyframe_map

    return result


def parse_detection_to_list(
    detection_dict: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Hàm tiện ích (Utility): Chuyển đổi các mảng song song (Parallel Arrays) từ JSON 
    thành danh sách các dictionary object riêng lẻ để dễ truy vấn/lọc.

    Input (Gốc):
        {
            "detection_class_names": ["person", "car"],
            "detection_scores": [0.95, 0.88],
            ...
        }

    Output (Đã parse):
        [
            {"class_name": "person", "score": 0.95, ...},
            {"class_name": "car", "score": 0.88, ...}
        ]
    """
    names = detection_dict.get("detection_class_names", [])
    labels = detection_dict.get("detection_class_labels", [])
    entities = detection_dict.get("detection_class_entities", [])
    scores = detection_dict.get("detection_scores", [])
    boxes = detection_dict.get("detection_boxes", [])

    parsed_objects: List[Dict[str, Any]] = []
    num_objects = len(names)

    for i in range(num_objects):
        parsed_objects.append({
            "class_name": names[i] if i < len(names) else "",
            "class_label": labels[i] if i < len(labels) else 0,
            "class_entity": entities[i] if i < len(entities) else "",
            "score": scores[i] if i < len(scores) else 0.0,
            "box": boxes[i] if i < len(boxes) else [],  # [y_min, x_min, y_max, x_max]
        })

    return parsed_objects


if __name__ == "__main__":
    print("Đang tiến hành load dữ liệu objects...")
    objects_data = load_objects()
    print(f"Đã load thành công dữ liệu objects của {len(objects_data)} video.\n")

    # Hiển thị mẫu dữ liệu của video và keyframe đầu tiên
    for v_id, keyframes in objects_data.items():
        print(f"- Video ID: {v_id} | Số keyframes có objects: {len(keyframes)}")
        
        first_frame_id = next(iter(keyframes.keys()), None)
        if first_frame_id:
            raw_obj = keyframes[first_frame_id]
            print(f"  + Keyframe ID mẫu: {first_frame_id}")
            print(f"    * Số lượng objects phát hiện: {len(raw_obj.get('detection_class_names', []))}")
            
            # Thử parse sang dạng danh sách đối tượng
            parsed = parse_detection_to_list(raw_obj)
            if parsed:
                print(f"    * Object đầu tiên trong frame: {parsed[0]}")
        print()