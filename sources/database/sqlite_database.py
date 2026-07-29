import json
import os
import sqlite3
import sys
from pathlib import Path
from contextlib import closing
from typing import Any, Dict, List


# =====================================================================
# Cấu hình Đường dẫn Tuyệt đối & System Path (Tránh lỗi Import)
# =====================================================================
CURRENT_FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE_PATH.parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _ensure_database_directory(db_path: str) -> None:
    if db_path != ":memory:":
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)


def _create_connection(db_path: str) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def _serialize_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _deserialize_value(value: Any) -> Any:
    if isinstance(value, str):
        trimmed_value = value.strip()
        if (trimmed_value.startswith("[") and trimmed_value.endswith("]")) or (
            trimmed_value.startswith("{") and trimmed_value.endswith("}")
        ):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
    return value


def _records_to_rows(records: List[Dict[str, Any]], columns: List[str]) -> List[tuple[Any, ...]]:
    return [tuple(_serialize_value(record.get(column)) for column in columns) for record in records]


def _rows_to_records(rows: List[sqlite3.Row]) -> Dict[str, Dict[str, Any]]:
    records_map: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        record = dict(row)
        for key, value in record.items():
            record[key] = _deserialize_value(value)
        records_map[record["global_id"]] = record
    return records_map


class SQLiteDatabase:
    """SQLite database for storing flattened keyframe metadata."""

    COLUMNS = [
        "global_id",
        "video_id",
        "frame_id",
        "n",
        "pts_time",
        "fps",
        "frame_idx",
        "image_path",
        "video_author",
        "video_channel_id",
        "video_channel_url",
        "video_description",
        "video_keywords",
        "video_length",
        "video_publish_date",
        "video_title",
        "video_watch_url",
        "detection_class_names",
        "detection_class_labels",
        "detection_scores",
        "detection_boxes",
        "detection_class_entities",
    ]

    def __init__(self, db_path: str = "data/processed/metadata.db"):
        self.db_path = db_path
        _ensure_database_directory(db_path)
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        return _create_connection(self.db_path)

    def init_db(self) -> None:
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS keyframes (
            global_id TEXT PRIMARY KEY,
            video_id TEXT,
            frame_id TEXT,
            n INTEGER,
            pts_time REAL,
            fps REAL,
            frame_idx INTEGER,
            image_path TEXT,
            video_author TEXT,
            video_channel_id TEXT,
            video_channel_url TEXT,
            video_description TEXT,
            video_keywords TEXT,
            video_length REAL,
            video_publish_date TEXT,
            video_title TEXT,
            video_watch_url TEXT,
            detection_class_names TEXT,
            detection_class_labels TEXT,
            detection_scores TEXT,
            detection_boxes TEXT,
            detection_class_entities TEXT
        );
        """

        create_index_global_id = "CREATE INDEX IF NOT EXISTS idx_global_id ON keyframes(global_id);"
        create_index_video_id = "CREATE INDEX IF NOT EXISTS idx_video_id ON keyframes(video_id);"

        with closing(self.get_connection()) as connection:
            cursor = connection.cursor()
            cursor.execute(create_table_sql)
            cursor.execute(create_index_global_id)
            cursor.execute(create_index_video_id)
            connection.commit()

        print(f"✅ Đã khởi tạo thành công CSDL SQLite tại: {self.db_path}")

    def insert_records(self, metadata_records: List[Dict[str, Any]]) -> None:
        if not metadata_records:
            print("⚠️ Danh sách metadata_records rỗng, bỏ qua quá trình insert.")
            return

        columns_str = ", ".join(self.COLUMNS)
        placeholders_str = ", ".join(["?"] * len(self.COLUMNS))
        insert_sql = f"INSERT OR REPLACE INTO keyframes ({columns_str}) VALUES ({placeholders_str});"
        rows_to_insert = _records_to_rows(metadata_records, self.COLUMNS)

        try:
            with closing(self.get_connection()) as connection:
                cursor = connection.cursor()
                cursor.executemany(insert_sql, rows_to_insert)
                connection.commit()
            print(f"💾 Đã chèn/cập nhật thành công {len(metadata_records)} bản ghi vào CSDL SQLite.")
        except sqlite3.Error as e:
            print(f"❌ LỖI SQLITE khi chèn dữ liệu: {e}")
            print("💡 Gợi ý: Kiểm tra xem file database có đang bị ứng dụng khác khóa không, hoặc cấu trúc bảng có bị thay đổi không.")

    def get_records_by_global_ids(self, global_ids: List[str]) -> List[Dict[str, Any]]:
        if not global_ids:
            return []

        records_map = self._get_records_map_by_global_ids(global_ids)
        return self._build_ordered_records(global_ids, records_map)

    def _get_records_map_by_global_ids(self, global_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        records_map: Dict[str, Dict[str, Any]] = {}
        CHUNK_SIZE = 900  # Đảm bảo nằm trong giới hạn an toàn của SQLite

        with closing(self.get_connection()) as connection:
            cursor = connection.cursor()

            for chunk_ids in self._chunk_values(global_ids, CHUNK_SIZE):
                rows = self._fetch_rows_by_global_ids(cursor, chunk_ids)
                records_map.update(_rows_to_records(rows))

        return records_map

    def _chunk_values(self, values: List[str], chunk_size: int) -> List[List[str]]:
        return [values[i:i + chunk_size] for i in range(0, len(values), chunk_size)]

    def _fetch_rows_by_global_ids(self, cursor: sqlite3.Cursor, chunk_ids: List[str]) -> List[sqlite3.Row]:
        placeholders = ", ".join(["?"] * len(chunk_ids))
        query_sql = f"SELECT * FROM keyframes WHERE global_id IN ({placeholders});"
        cursor.execute(query_sql, chunk_ids)
        return cursor.fetchall()

    def _build_ordered_records(
        self,
        global_ids: List[str],
        records_map: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        return [records_map[g_id] for g_id in global_ids if g_id in records_map]


if __name__ == "__main__":
    print("--- 🧪 BẮT ĐẦU KIỂM THỬ MODULE SQLITE_DB ---")

    test_db_path = "data/processed_test/test_metadata.db"
    db = SQLiteDatabase(db_path=test_db_path)

    dummy_records = [
        {
            "global_id": "L01_V001_0001",
            "video_id": "L01_V001",
            "frame_id": "0001",
            "n": 1,
            "pts_time": 0.04,
            "fps": 25.0,
            "frame_idx": 1,
            "image_path": "data/keyframes/L01_V001/0001.jpg",
            "video_author": "Channel Test A",
            "video_channel_id": "UC123456789",
            "video_channel_url": "https://youtube.com/channel/UC123456789",
            "video_description": "Video thử nghiệm hệ thống tra cứu keyframe.",
            "video_keywords": ["test", "ai", "retrieval"],
            "video_length": 120.5,
            "video_publish_date": "2026-01-01",
            "video_title": "Video Thử Nghiệm 1",
            "video_watch_url": "https://youtube.com/watch?v=L01_V001",
            "detection_class_names": ["person", "car"],
            "detection_class_labels": [1, 3],
            "detection_scores": [0.95, 0.88],
            "detection_boxes": [[10, 20, 100, 200], [50, 60, 150, 250]],
            "detection_class_entities": ["Person", "Car"],
        },
        {
            "global_id": "L01_V001_0002",
            "video_id": "L01_V001",
            "frame_id": "0002",
            "n": 2,
            "pts_time": 0.08,
            "fps": 25.0,
            "frame_idx": 2,
            "image_path": "data/keyframes/L01_V001/0002.jpg",
            "video_author": "Channel Test A",
            "video_channel_id": "UC123456789",
            "video_channel_url": "https://youtube.com/channel/UC123456789",
            "video_description": "Video thử nghiệm hệ thống tra cứu keyframe.",
            "video_keywords": ["test", "ai", "retrieval"],
            "video_length": 120.5,
            "video_publish_date": "2026-01-01",
            "video_title": "Video Thử Nghiệm 1",
            "video_watch_url": "https://youtube.com/watch?v=L01_V001",
            "detection_class_names": ["dog"],
            "detection_class_labels": [18],
            "detection_scores": [0.92],
            "detection_boxes": [[30, 40, 120, 180]],
            "detection_class_entities": ["Dog"],
        },
        {
            "global_id": "L01_V002_0010",
            "video_id": "L01_V002",
            "frame_id": "0010",
            "n": 10,
            "pts_time": 0.40,
            "fps": 25.0,
            "frame_idx": 10,
            "image_path": "data/keyframes/L01_V002/0010.jpg",
            "video_author": "Channel Test B",
            "video_channel_id": "UC987654321",
            "video_channel_url": "https://youtube.com/channel/UC987654321",
            "video_description": "Cảnh quay phong cảnh thiên nhiên.",
            "video_keywords": ["nature", "landscape"],
            "video_length": 200.0,
            "video_publish_date": "2026-02-01",
            "video_title": "Thiên Nhiên Kỳ Diệu",
            "video_watch_url": "https://youtube.com/watch?v=L01_V002",
            "detection_class_names": ["tree", "mountain"],
            "detection_class_labels": [60, 62],
            "detection_scores": [0.99, 0.91],
            "detection_boxes": [[0, 0, 500, 500], [100, 100, 400, 400]],
            "detection_class_entities": ["Tree", "Mountain"],
        },
    ]

    db.insert_records(dummy_records)

    query_ids = ["L01_V002_0010", "L01_V001_0001"]
    retrieved_records = db.get_records_by_global_ids(query_ids)

    print(f"\n🔍 KẾT QUẢ TRUY XUẤT CHO {len(query_ids)} GLOBAL IDs:")
    for index, record in enumerate(retrieved_records, 1):
        print(f"\n--- Bản ghi #{index} ---")
        print(f"  • Global ID: {record['global_id']}")
        print(f"  • Video Title: {record['video_title']}")
        print(f"  • Frame Path: {record['image_path']}")
        print(f"  • Detected Objects: {record['detection_class_names']}")
        print(f"  • Detection Scores: {record['detection_scores']}")

    assert len(retrieved_records) == 2, "Lỗi: Số lượng bản ghi lấy ra không đúng!"
    assert retrieved_records[0]["global_id"] == "L01_V002_0010", "Lỗi: Thứ tự bản ghi thứ nhất bị sai!"
    assert retrieved_records[1]["global_id"] == "L01_V001_0001", "Lỗi: Thứ tự bản ghi thứ hai bị sai!"
    assert isinstance(retrieved_records[0]["detection_class_names"], list), "Lỗi: Dữ liệu JSON chưa được deserialize đúng dạng list!"

    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    test_dir = os.path.dirname(test_db_path)
    if os.path.exists(test_dir):
        os.rmdir(test_dir)

    print("\n✅ KIỂM THỬ HOÀN TẤT THÀNH CÔNG!")