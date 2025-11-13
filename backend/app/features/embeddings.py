""" 
Store embeddings in a SQLite database for speed
"""

import sqlite3
import numpy as np
from typing import List, Dict, Optional, Tuple
# config
from app.core.config import settings


class EmbeddingDB:
    def __init__(self, cfg=None, nlp=None):
        self.cfg = cfg or settings
        self.conn = sqlite3.connect(self.cfg.EMBEDDING_DB_PATH)
        self.create_table()

    def create_table(self):
        with self.conn:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY,
                    text TEXT,
                    embedding BLOB
                )
            ''')

    def add_embedding(self, text: str, embedding: np.ndarray):
        with self.conn:
            self.conn.execute('''
                INSERT INTO embeddings (text, embedding) VALUES (?, ?)
            ''', (text, embedding.tobytes()))

    def get_all_embeddings(self) -> List[Tuple[str, np.ndarray]]:
        cursor = self.conn.cursor()
        cursor.execute('SELECT text, embedding FROM embeddings')
        rows = cursor.fetchall()
        return [(text, np.frombuffer(embedding)) for text, embedding in rows]

    def close(self):
        self.conn.close()