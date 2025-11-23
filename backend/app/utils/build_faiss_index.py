"""
build_faiss_index.py - Build FAISS index from extracted Wikipedia
"""
import faiss
import multiprocessing
import os
import json
from tqdm import tqdm
from app.utils.faiss_vecdb import FaissVecDB


def read_wikiextractor_output(extracted_path: str, max_articles: int = 100):
    articles = []
    documents = []
    filenames = []

    count = 0

    print(f"Reading from {extracted_path}...")

    for root, dirs, files in os.walk(extracted_path):
        for filename in files:
            if not filename.startswith('wiki'):
                continue
            filepath = os.path.join(root, filename)
            # handle both compressed and uncompressed files
            try:
                with open(filepath, 'rt', encoding='utf-8') as f:
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
                print(f"error reading {filepath}: {e}")
    print(f"Total articles: {len(articles)}")
    return articles


def read_wikiextractor_batches(extracted_path: str, batch_size: int = 10000):
    batch = []
    count = 0
    for root, dirs, files in os.walk(extracted_path):
        for filename in files:
            if not filename.startswith("wiki"):
                continue
            filepath = os.path.join(root, filename)
            with open(filepath, 'rt', encoding='utf-8') as f:
                for line in f:
                    try:
                        article = json.loads(line)
                        batch.append(article)
                        count += 1
                        if len(batch) >= batch_size:
                            yield batch
                            batch = []
                    except json.JSONDecodeError:
                        continue
    if batch:
        yield batch


def chunk_article(text: str, title: str, chunk_size: int = 500):
    if not text or not isinstance(text, str):
        return []

    if not title or not isinstance(title, str):
        title = ""
        
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
            chunk_text = f"{title}. {'. '.join(current_chunk)}."
            chunks.append(chunk_text)
            current_chunk = []
            current_word_count = 0
    if current_chunk:
        chunk_text = f"{title}. {'. '.join(current_chunk)}."
        chunks.append(chunk_text)
    chunks = [c.strip() for c in chunks if c.strip()]
    return chunks


def build_index(extracted_path: str, output_path: str, max_articles=None, batch_size=512, cfg=None, overwrite: bool = False):
    print("="*70)
    print("Building FAISS Index from Wikipedia")
    print("="*70)

    index_file = f"{output_path}.index"
    docs_file = f"{output_path}_docs.pkl"
    if not overwrite and os.path.exists(index_file) and os.path.exists(docs_file):
        print(f"FAISS index already exists at {output_path}. Skipping build.")
        return

    # create the db
    vector_db = FaissVecDB(cfg=cfg)
    vector_db.batch_size = batch_size
    # initiate the counts
    total_articles = 0
    total_chunks = 0

    print(f"\nReading and chunking articles in batches of size {batch_size}..")
    if max_articles is not None:
        total_articles_to_process = max_articles
    else:
        total_articles_to_process = estimate_total_articles(extracted_path)    
    pbar = tqdm(total=total_articles_to_process, desc="Encoding Wikipedia articles")
    for batch_articles in read_wikiextractor_batches(extracted_path, batch_size=batch_size):
        batch_chunks = []
        batch_metadata = []

        for article in batch_articles:
            title = article.get('title', '')
            text = article.get('text', '')
            article_id = article.get('id', '')
            url = article.get('url', f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}")

            chunks = chunk_article(text, title)
            for i, chunk in enumerate(chunks):
                batch_chunks.append(chunk)
                batch_metadata.append({
                    'title': title,
                    'url': url,
                    'article_id': article_id,
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                })

        # add this batch to db
        if batch_chunks:
            vector_db.add_documents(batch_chunks, batch_metadata)
            total_chunks += len(batch_chunks)
            total_articles += len(batch_articles)
            print(f"Processed {total_articles} articles, total chunks so far: {total_chunks}")
        # update progress bar
        pbar.update(len(batch_articles))
        # stop if max_articles reached
        if max_articles and total_articles >= max_articles:
            print(f"Reached max_articles={max_articles}. Stopping.")
            break

    # save vector db
    print(f"\nSaving FAISS index to {output_path}...")
    vector_db.save(output_path)

    print("\n" + "="*70)
    print("FAISS Index Built Successfully!")
    print("="*70)
    print(f"  Articles processed: {total_articles}")
    print(f"  Total chunks: {total_chunks}")
    print(f"  Index location: {output_path}")


def estimate_total_articles(extracted_path):
    total = 0
    for root, dirs, files in os.walk(extracted_path):
        for file in files:
            if not file.startswith("wiki"):
                continue
            filepath = os.path.join(root, file)
            with open(filepath, 'r', encoding='utf-8') as f:
                total += sum(1 for _ in f)
    return total


if __name__ == "__main__":
    EXTRACTED_PATH = 'data/wikipedia/extracted'
    OUTPUT_PATH = 'data/vector_db/faiss_index'
    MAX_ARTICLES = 10000

    build_index(EXTRACTED_PATH, OUTPUT_PATH, MAX_ARTICLES)
