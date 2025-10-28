"""
build_faiss_index.py - Build FAISS index from extracted Wikipedia
"""
import os
import json
import bz2
from tqdm import tqdm
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from faiss_vecdb import FaissVecDB


def read_wikiextractor_output(extracted_path: str, max_articles: int = None):
    """Read articles from wikiextractor output."""
    articles = []
    count = 0

    print(f"Reading from {extracted_path}...")

    for root, dirs, files in os.walk(extracted_path):
        for filename in files:
            if not filename.startswith('wiki_'):
                continue

            filepath = os.path.join(root, filename)

            # handle both compressed and uncompressed files
            if filename.endswith('.bz2'):
                opener = bz2.open
            else:
                opener = open

            try:
                with opener(filepath, 'rt', encoding='utf-8') as f:
                    for line in f:
                        try:
                            article = json.loads(line)
                            articles.append(article)
                            count += 1

                            if max_articles and count >= max_articles:
                                print(f"Reached limit: {max_articles} articles")
                                return articles

                            if count % 10000 == 0:
                                print(f"  Loaded {count} articles...")

                        except json.JSONDecodeError:
                            continue

            except Exception as e:
                print(f"Error reading {filepath}: {e}")

    print(f"Total articles: {len(articles)}")
    return articles


def chunk_article(text: str, title: str, chunk_size: int = 500):
    """Chunk article into passages."""
    sentences = text.replace('\n', ' ').split('. ')

    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        words = sentence.split()
        current_word_count += len(words)
        current_chunk.append(sentence)

        if current_word_count >= chunk_size:
            # Add title for context
            chunk_text = f"{title}. {'. '.join(current_chunk)}."
            chunks.append(chunk_text)
            current_chunk = []
            current_word_count = 0
    if current_chunk:
        chunk_text = f"{title}. {'. '.join(current_chunk)}."
        chunks.append(chunk_text)

    return chunks


def build_index(extracted_path: str, output_path: str, max_articles: int = None):
    """Build FAISS index from extracted Wikipedia."""
    print("="*70)
    print("Building FAISS Index from Wikipedia")
    print("="*70)

    # Step 1: Read articles
    articles = read_wikiextractor_output(extracted_path, max_articles)

    if not articles:
        print("ERROR: No articles found!")
        return

    # Step 2: Chunk articles
    print("\nChunking articles...")
    all_chunks = []
    all_metadata = []

    for article in tqdm(articles):
        title = article.get('title', '')
        text = article.get('text', '')
        article_id = article.get('id', '')
        url = article.get('url', f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}")

        if len(text) < 100:
            continue

        chunks = chunk_article(text, title)

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadata.append({
                'title': title,
                'url': url,
                'article_id': article_id,
                'chunk_index': i,
                'total_chunks': len(chunks)
            })

    print(f"Generated {len(all_chunks)} chunks from {len(articles)} articles")

    # Step 3: Build FAISS index using your FaissVecDB
    print("\nBuilding FAISS index...")
    vector_db = FaissVecDB()
    vector_db.add_documents(all_chunks, all_metadata)

    # Step 4: Save
    print(f"\nSaving to {output_path}...")
    vector_db.save(output_path)

    print("\n" + "="*70)
    print("FAISS Index Built Successfully!")
    print("="*70)
    print(f"  Articles processed: {len(articles)}")
    print(f"  Total chunks: {len(all_chunks)}")
    print(f"  Index location: {output_path}")


if __name__ == "__main__":
    # Configuration
    EXTRACTED_PATH = '../data/wikipedia/extracted'
    OUTPUT_PATH = '../data/wikipedia/faiss_index'

    # Start with 10k articles for testing (None = all articles)
    MAX_ARTICLES = 10000

    build_index(EXTRACTED_PATH, OUTPUT_PATH, MAX_ARTICLES)