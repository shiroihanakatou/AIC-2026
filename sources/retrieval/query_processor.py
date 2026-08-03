import re
import json
import torch
import open_clip
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

class QueryProcessor:
    def __init__(self, vocab_path: Path):
        self.vocab_path = vocab_path
        self.model = None
        self.tokenizer = None
        self.vocab_entities: List[str] = []
        
        # Load vocab để trích xuất đối tượng (E_q)
        if self.vocab_path.exists():
            with open(self.vocab_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.vocab_entities = [str(v).lower() for v in data.get("entities", [])]

    def _load_model(self):
        """Lazy initialization mô hình CLIP."""
        if self.model is None:
            # Sử dụng kiến trúc ViT-B-32 theo chuẩn (tương đương với clip features của data)
            model_name = 'ViT-B-32'
            pretrained = 'openai'
            self.model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
            self.tokenizer = open_clip.get_tokenizer(model_name)
            self.model.eval()

    def process_query(self, query_text: str) -> Dict[str, Any]:
        """Chuẩn hóa, mã hóa CLIP và trích xuất Entity."""
        self._load_model()
        
        # 1. Chuẩn hóa Prompt
        query_cleaned = query_text.strip().lower()
        if not query_cleaned.startswith("a photo of") and not query_cleaned.startswith("a video of"):
            prompt = f"a photo of {query_cleaned}"
        else:
            prompt = query_cleaned
            
        # 2 & 3. Mã hóa & Chuẩn hóa L2
        text_tokens = self.tokenizer([prompt])
        with torch.no_grad():
            q_vec = self.model.encode_text(text_tokens).numpy()[0]
        q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-12)
        
        # 4. Trích xuất E_q bằng NLP nhẹ (Vocab Substring Matching)
        E_q = set()
        if self.vocab_entities:
            for entity in self.vocab_entities:
                # Sử dụng Regex boundaries \b để match chính xác từ vựng
                pattern = r'\b' + re.escape(entity) + r'\b'
                if re.search(pattern, query_cleaned):
                    E_q.add(entity.title())  # Lưu lại ở dạng Capitalize giống vocab gốc

        return {
            "query_vector": q_norm,
            "entities": list(E_q),
            "translated_text": query_cleaned
        }