import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import logging

def process_clip_features(ordered_frames: List[Dict[str, Any]], data_dir: Path, output_path: Path):
    """
    Chuẩn hóa L2 và gộp vector CLIP thành ma trận master (N x 512).
    Sử dụng pre-allocation array để tránh tràn RAM khi append liên tục.
    """
    if not ordered_frames:
        logging.warning("Danh sách frame trống, bỏ qua tạo CLIP master.")
        return

    clip_dir = data_dir / "clip-features"
    total_frames = len(ordered_frames)
    feature_dim = 512
    
    # Pre-allocate memory (dtype float32 tiết kiệm RAM)
    master_clip = np.zeros((total_frames, feature_dim), dtype=np.float32)
    
    # Gom nhóm các frames theo video_id để load file .npy 1 lần duy nhất cho mỗi video
    video_groups: Dict[str, List[int]] = {}
    for f in ordered_frames:
        video_groups.setdefault(f["video_id"], []).append(f["global_frame_id"])
        
    for video_id, global_ids in video_groups.items():
        npy_path = clip_dir / f"{video_id}.npy"
        
        if not npy_path.exists():
            raise FileNotFoundError(f"Lỗi: Thiếu file CLIP feature cho video {video_id}: {npy_path}")
            
        video_features = np.load(npy_path)
        
        # Bắt lỗi số lượng vector không khớp số keyframe
        if video_features.shape[0] != len(global_ids):
            raise ValueError(
                f"Lỗi độ dài: {video_id}.npy có {video_features.shape[0]} vectors, "
                f"nhưng có {len(global_ids)} keyframes trong thư mục."
            )
        
        # Đổ vào vị trí tương ứng trên master matrix
        for i, g_id in enumerate(global_ids):
            master_clip[g_id] = video_features[i]
            
    np.save(output_path, master_clip)

if __name__ == "__main__":
    from frame_indexer import get_ordered_frame_list
    base_dir = Path(__file__).resolve().parents[2]
    data_dir = base_dir / "data"
    frames = get_ordered_frame_list(data_dir)
    
    process_clip_features(frames, data_dir, data_dir / "processed" / "clip_master.npy")
    print("✅ Đã xử lý và chuẩn hóa ma trận CLIP.")