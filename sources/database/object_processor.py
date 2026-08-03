import json
from pathlib import Path
from typing import List, Dict, Any
import logging

def process_objects(ordered_frames: List[Dict[str, Any]], data_dir: Path, out_objects: Path, out_vocab: Path):
    """
    Đọc, lọc (score >= 0.2) và tạo objects_master.json, 
    đồng thời trích xuất từ vựng thành openimages_v4_vocab.json.
    """
    objects_dir = data_dir / "objects"
    
    master_dict = {}
    entities_set = set()
    label_map = {}
    
    for frame in ordered_frames:
        global_id = str(frame["global_frame_id"])
        video_id = frame["video_id"]
        frame_filename = Path(frame["frame_path"]).stem  # Lấy tên file không lấy extension
        
        json_path = objects_dir / video_id / f"{frame_filename}.json"
        
        filtered_objects = []
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                try:
                    frame_objects = json.load(f)
                    scores = frame_objects.get("detection_scores", [])
                    entity_ids = frame_objects.get("detection_class_names", [])
                    entity_names = frame_objects.get("detection_class_entities", [])
                    bbox_lists = frame_objects.get("detection_boxes", [])
                    class_ids = frame_objects.get("detection_class_labels", [])
                    number_of_objects = len(scores)
                    for i in range(number_of_objects):
                        # Theo định dạng mẫu: [score, entity_id, entity_name, bbox_list, class_id]
                        score = float(scores[i])
                        if score >= 0.2:
                            filtered_objects.append((
                                score,
                                entity_names[i],
                                entity_ids[i],
                                bbox_lists[i],
                                class_ids[i]
                            ))
                            
                            entity_name = entity_names[i]
                            class_id = int(class_ids[i])
                            
                            entities_set.add(entity_name)
                            if entity_name not in label_map:
                                label_map[entity_name] = class_id
                            elif label_map[entity_name] != class_id:
                                logging.warning(
                                    f"Trùng class_id cho entity '{entity_name}': "
                                    f"{label_map[entity_name]} vs {class_id}. "
                                    "Giữ class_id đầu tiên."
                                )
                except (json.JSONDecodeError, IndexError, ValueError) as e:
                    logging.warning(f"Lỗi parse JSON tại {json_path}: {e}")
                    
        # Dù không có file json hoặc lọc xong rỗng thì vẫn lưu array rỗng để đồng bộ khung hình
        master_dict[global_id] = filtered_objects
        
    # Ghi xuất files
    with open(out_objects, 'w', encoding='utf-8') as f:
        json.dump(master_dict, f, separators=(',', ':')) # Dùng separator để nén dung lượng
        
    vocab_data = {
        "entities": sorted(list(entities_set)),
        "label_map": label_map
    }
    with open(out_vocab, 'w', encoding='utf-8') as f:
        json.dump(vocab_data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    from frame_indexer import get_ordered_frame_list
    base_dir = Path(__file__).resolve().parents[2]
    data_dir = base_dir / "data"
    out_dir = data_dir / "processed"
    
    frames = get_ordered_frame_list(data_dir)
    process_objects(frames, data_dir, out_dir / "objects_master.json", out_dir / "openimages_v4_vocab.json")
    print("✅ Đã xử lý objects và trích xuất vocab.")