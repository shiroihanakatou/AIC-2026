import sqlite3
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def _find_csv_columns(columns: pd.Index) -> tuple[str, str]:
    """Tìm tên cột linh hoạt trong CSV cho frame_idx và pts_time."""
    col_idx, col_pts = None, None
    for col in columns:
        col_lower = str(col).lower()
        if col_idx is None and ('frame_idx' in col_lower):
            col_idx = col
        if col_pts is None and ('pts_time' in col_lower):
            col_pts = col
    return col_idx or columns[0], col_pts or columns[1] # Dự phòng nếu không khớp

def get_ordered_frame_list(data_dir: Path) -> List[Dict[str, Any]]:
    """
    Quét thư mục keyframes và ánh xạ chuẩn xác thành list dictionary
    được sắp xếp theo video_id và tên file keyframe.
    """
    keyframes_dir = data_dir / "keyframes"
    map_csv_dir = data_dir / "map-keyframes"
    metadata_dir = data_dir / "metadata"
    
    if not keyframes_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục keyframes tại: {keyframes_dir}")

    # Sắp xếp các thư mục video theo alphabet
    video_dirs = sorted([d for d in keyframes_dir.iterdir() if d.is_dir()])
    
    ordered_frames = []
    global_id = 0

    for v_dir in video_dirs:
        video_id = v_dir.name
        csv_path = map_csv_dir / f"{video_id}.csv"
        meta_path = metadata_dir / f"{video_id}.json"
        
        # Load CSV metadata
        if not csv_path.exists():
            raise FileNotFoundError(f"Thiếu file CSV ánh xạ cho video {video_id}: {csv_path}")
        
        df = pd.read_csv(csv_path)
        col_idx, col_pts = _find_csv_columns(df.columns)
        
        # Tạo mapping từ tên file không có đuôi mở rộng sang các attributes
        # Thường tên file hình ảnh sẽ khớp với frame_idx hoặc chuỗi định dạng
        # Ở đây ta giả định tên file liên kết theo chỉ mục dòng, hoặc có cột tương ứng.
        
        frame_files = sorted([f for f in v_dir.glob("*.jpg")])
        if len(frame_files) != len(df):
            logging.warning(f"Cảnh báo: Số lượng ảnh ({len(frame_files)}) không khớp với số dòng CSV ({len(df)}) tại {video_id}")
            
        for i, frame_file in enumerate(frame_files):
            # Cố gắng lấy data từ CSV, fallback nếu vượt index
            try:
                frame_idx = int(df.iloc[i][col_idx])
                pts_time = float(df.iloc[i][col_pts])
            except IndexError:
                frame_idx, pts_time = -1, 0.0

            frame_data = {
                "global_frame_id": global_id,
                "video_id": video_id,
                "frame_path": str(frame_file.relative_to(data_dir.parent)),
                "pts_time": pts_time,
                "frame_idx": frame_idx,
                "metadata_path": str(meta_path.relative_to(data_dir.parent)) if meta_path.exists() else None
            }
            ordered_frames.append(frame_data)
            global_id += 1

    return ordered_frames

def build_frames_db(ordered_frames: List[Dict[str, Any]], output_db_path: Path):
    """Khởi tạo và insert dữ liệu vào SQLite database."""
    conn = sqlite3.connect(output_db_path)
    cursor = conn.cursor()
    
    cursor.execute("DROP TABLE IF EXISTS frames")
    cursor.execute('''
        CREATE TABLE frames (
            global_frame_id INTEGER PRIMARY KEY,
            video_id TEXT,
            frame_path TEXT,
            pts_time REAL,
            frame_idx INTEGER,
            metadata_path TEXT
        )
    ''')
    
    records = [
        (
            f["global_frame_id"], f["video_id"], f["frame_path"], 
            f["pts_time"], f["frame_idx"], f["metadata_path"]
        )
        for f in ordered_frames
    ]
    
    cursor.executemany(
        "INSERT INTO frames VALUES (?, ?, ?, ?, ?, ?)", 
        records
    )
    conn.commit()
    conn.close()

if __name__ == "__main__":
    # Test độc lập
    base_dir = Path(__file__).resolve().parents[2]
    data_dir = base_dir / "data"
    out_dir = data_dir / "processed"
    out_dir.mkdir(exist_ok=True)
    
    frames = get_ordered_frame_list(data_dir)
    build_frames_db(frames, out_dir / "frames_info.db")
    print(f"✅ Đã lập chỉ mục {len(frames)} frames.")