import json
import os
import pickle
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np


# =====================================================================
# Cấu hình Đường dẫn Tuyệt đối & System Path (Tránh lỗi Import)
# =====================================================================
CURRENT_FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE_PATH.parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _ensure_float32(matrix: np.ndarray) -> np.ndarray:
    return matrix if matrix.dtype == np.float32 else matrix.astype(np.float32)


def _normalize_l2(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return matrix / norms


def _save_id_map(map_path: str, id_map: List[str]) -> None:
    if map_path.endswith(".json"):
        with open(map_path, "w", encoding="utf-8") as file_handle:
            json.dump(id_map, file_handle, ensure_ascii=False, indent=2)
        return

    with open(map_path, "wb") as file_handle:
        pickle.dump(id_map, file_handle)


def _load_id_map(map_path: str) -> List[str]:
    if map_path.endswith(".json"):
        with open(map_path, "r", encoding="utf-8") as file_handle:
            return json.load(file_handle)

    with open(map_path, "rb") as file_handle:
        return pickle.load(file_handle)


class VectorDatabase:
    """FAISS-based vector database for storing and searching CLIP embeddings."""

    def __init__(self, dimension: int = 512):
        self.dimension = dimension
        self.index: Optional[faiss.IndexFlatIP] = None
        self.id_map: List[str] = []

    def build_index(self, vector_matrix: np.ndarray, global_frame_ids: List[str]) -> None:
        vector_matrix = _ensure_float32(vector_matrix)

        n_samples, dimension = vector_matrix.shape
        assert dimension == self.dimension, (
            f"Kích thước vector không khớp! Kỳ vọng {self.dimension}, nhận {dimension}."
        )
        assert n_samples == len(global_frame_ids), (
            "Số lượng vector và số lượng global_frame_ids phải bằng nhau."
        )

        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(vector_matrix)
        self.id_map = list(global_frame_ids)

        print(f"✅ Đã xây dựng thành công FAISS Index với {self.index.ntotal} vectors ({self.dimension}D).")

    def save_index(self, index_path: str, map_path: str) -> None:
        if self.index is None:
            raise ValueError("Chưa có Index để lưu. Vui lòng gọi build_index() hoặc load_index() trước.")

        os.makedirs(Path(index_path).parent, exist_ok=True)
        os.makedirs(Path(map_path).parent, exist_ok=True)

        try:
            faiss.write_index(self.index, index_path)
            _save_id_map(map_path, self.id_map)
            print(f"💾 Đã lưu Index tại: {index_path}")
            print(f"💾 Đã lưu ID Map tại: {map_path}")
        except (IOError, OSError) as e:
            print(f"❌ LỖI I/O khi lưu FAISS Index: {e}")
            print("💡 Gợi ý: Kiểm tra xem ổ cứng còn trống không, hoặc bạn có quyền ghi vào thư mục này không.")

    def load_index(self, index_path: str, map_path: str) -> None:
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Không tìm thấy file index tại: {index_path}")
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"Không tìm thấy file map tại: {map_path}")

        self.index = faiss.read_index(index_path)
        self.dimension = self.index.d
        self.id_map = _load_id_map(map_path)

        assert self.index.ntotal == len(self.id_map), (
            "Lỗi: Số lượng phần tử trong FAISS index và ID map không khớp!"
        )

        print(f"⚡ Đã nạp Index thành công: {self.index.ntotal} vectors ({self.dimension}D).")

    def search(self, query_vector: np.ndarray, top_k: int = 100) -> List[Tuple[str, float]]:
        if self.index is None:
            raise ValueError("Chưa nạp hoặc xây dựng Index. Vui lòng kiểm tra lại.")

        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)

        query_vector = _ensure_float32(query_vector)
        query_vector = _normalize_l2(query_vector)

        actual_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vector, actual_k)

        results: List[Tuple[str, float]] = []
        for index, score in zip(indices[0], scores[0]):
            if index != -1:
                results.append((self.id_map[index], float(score)))

        return results


if __name__ == "__main__":
    print("--- 🧪 BẮT ĐẦU KIỂM THỬ MODULE VECTOR_DB ---")

    sample_count = 10
    dimension = 512
    np.random.seed(42)

    dummy_matrix = np.random.randn(sample_count, dimension).astype(np.float32)
    dummy_matrix = _normalize_l2(dummy_matrix)
    dummy_ids = [f"L01_V001_{index + 1:04d}" for index in range(sample_count)]

    test_dir = Path("data/processed_test")
    index_file = test_dir / "test_faiss.index"
    map_file = test_dir / "test_id_map.json"

    writer = VectorDatabase(dimension=dimension)
    writer.build_index(dummy_matrix, dummy_ids)
    writer.save_index(str(index_file), str(map_file))

    reader = VectorDatabase()
    reader.load_index(str(index_file), str(map_file))

    results = reader.search(dummy_matrix[0], top_k=3)

    print("\n🔍 KẾT QUẢ TRUY VẤN (Top 3):")
    for rank, (global_id, score) in enumerate(results, 1):
        print(f"  {rank}. Global ID: {global_id:<15} | Score: {score:.6f}")

    assert results[0][0] == dummy_ids[0], "Lỗi: Kết quả hàng đầu không khớp!"
    assert abs(results[0][1] - 1.0) < 1e-4, "Lỗi: Điểm Cosine/IP trùng khớp không xấp xỉ 1.0!"

    if index_file.exists():
        index_file.unlink()
    if map_file.exists():
        map_file.unlink()
    if test_dir.exists():
        test_dir.rmdir()

    print("\n✅ KIỂM THỬ HOÀN TẤT THÀNH CÔNG!")