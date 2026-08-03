import logging
import sqlite3
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SearchEngine:
    def __init__(self, processed_dir: Path):
        self.processed_dir = processed_dir
        
        # 1. Load CLIP Master Matrix
        self.clip_master = np.load(processed_dir / "clip_master.npy")
        
        # 2. Load Objects Master JSON
        with open(processed_dir / "objects_master.json", 'r', encoding='utf-8') as f:
            self.objects_master = json.load(f)
            
        # 3. Load Vocab & Khởi tạo Tfidf để đo độ tương đồng văn bản
        with open(processed_dir / "openimages_v4_vocab.json", 'r', encoding='utf-8') as f:
            self.vocab_entities = json.load(f).get("entities", [])
            
        if self.vocab_entities:
            self.tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
            self.tfidf_matrix = self.tfidf.fit_transform(self.vocab_entities)
            
        # 4. Caching SQLite Database vào RAM
        disk_db = sqlite3.connect(processed_dir / "frames_info.db")
        self.db = sqlite3.connect(':memory:', check_same_thread=False)
        disk_db.backup(self.db)
        disk_db.close()
        self.db.row_factory = sqlite3.Row
        
        self.sim_lookup = {}

    def compute_text_sim_lookup(self, E_q: List[str]) -> Dict[Tuple[str, str], float]:
        """Tạo bảng tra cứu độ tương đồng cosine Text TF-IDF giữa E_q và Vocab."""
        lookup = {}
        if not E_q or not self.vocab_entities:
            return lookup
            
        q_vecs = self.tfidf.transform(E_q)
        sim_scores = cosine_similarity(q_vecs, self.tfidf_matrix)
        
        for i, eq in enumerate(E_q):
            for j, entity in enumerate(self.vocab_entities):
                # Chỉ lưu sim > 0 để tiết kiệm bộ nhớ
                score = float(sim_scores[i, j])
                if score > 0:
                    lookup[(eq, entity)] = score

        return lookup

    def search(self, query_data: Dict[str, Any], w_clip: float = 0.7, w_obj: float = 0.3, top_k: int = 50) -> List[Dict[str, Any]]:
        q = query_data["query_vector"]
        E_q = query_data["entities"]
        
        # 1. Tính Vectorized S_CLIP (Ma trận V (N x 512) dot q^T (512,))
        s_raw = self.clip_master @ q.T
        s_clip = (s_raw + 1.0) / 2.0
        
        # Two-pass Strategy: Chỉ lấy Top 500 S_CLIP cao nhất để tính Object Score
        top_candidates = min(500, len(s_clip))
        top_500_idx = np.argsort(s_clip)[::-1][:top_candidates]
        
        s_obj = np.zeros_like(s_clip)
        
        # 2. Tính S_OBJ
        if not E_q:
            w_clip = 1.0
            w_obj = 0.0
        else:
            self.sim_lookup = self.compute_text_sim_lookup(E_q)
            m = len(E_q)
            
            for global_id in top_500_idx:
                frame_objs = self.objects_master.get(str(global_id), [])
                if not frame_objs:
                    continue
                
                # frame_objs element theo format Task 1: [score, mid, entity_name, bbox, class_id]
                score_obj_frame = 0.0
                for eq in E_q:
                    max_sim_for_eq = 0.0
                    for obj in frame_objs:
                        s_j = obj[0]       # detection_score
                        e_j = obj[1]       # entity_name
                        sim = self.sim_lookup.get((eq, e_j), 0.0)
                        max_sim_for_eq = max(max_sim_for_eq, s_j * sim)
                    score_obj_frame += max_sim_for_eq
                    
                s_obj[global_id] = score_obj_frame / m
                
        # 3. Tính điểm tổng và sắp xếp lại trong phạm vi Top 500
        final_scores = w_clip * s_clip + w_obj * s_obj
        
        # Rút trích các ứng viên từ tập 500, sort để tìm ra Top K cuối cùng
        top_500_final_scores = final_scores[top_500_idx]
        top_k_rel_idx = np.argsort(top_500_final_scores)[::-1][:min(top_k, top_candidates)]
        top_k_idx = top_500_idx[top_k_rel_idx]
        
        # 4. Trích xuất metadata siêu tốc từ RAM DB
        results = []
        cursor = self.db.cursor()
        
        for rank, g_id in enumerate(top_k_idx):
            cursor.execute("SELECT * FROM frames WHERE global_frame_id = ?", (int(g_id),))
            row = cursor.fetchone()
            
            results.append({
                "global_frame_id": int(g_id),
                "score": round(float(final_scores[g_id]), 4),
                "s_clip": round(float(s_clip[g_id]), 4),
                "s_obj": round(float(s_obj[g_id]), 4),
                "video_id": row["video_id"] if row else "Unknown",
                "frame_path": row["frame_path"] if row else "Unknown",
                "pts_time": float(row["pts_time"]) if row else 0.0,
                "frame_idx": int(row["frame_idx"]) if row else -1
            })
            
        return results