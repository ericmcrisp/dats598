import os
import json
import pickle
from tqdm import tqdm

from app.utils.faiss_vecdb import FaissVecDB
from app.utils.build_faiss_index import read_wikiextractor_batches
from app.utils.build_faiss_index import chunk_article


def save_progress(progress_path, article_id, articles_done, chunks_done):
    progress = {
        "last_article_id": article_id,
        "articles_done": articles_done,
        "chunks_done": chunks_done
    }
    with open(progress_path, "w") as f:
        json.dump(progress, f)


def load_progress(progress_path):
    if not os.path.exists(progress_path):
        return None
    with open(progress_path, "r") as f:
        return json.load(f)


def build_index_resumable(extracted_path: str, output_path: str, cfg=None, batch_size=256): 
    print("\n" + "=" * 80)
    print(" RESUMABLE WIKIPEDIA INDEX BUILDER ")
    print("=" * 80)

    index_file = output_path + ".index"
    docs_file = output_path + "_docs.pkl"
    progress_file = output_path + "_progress.json"

    # ------------------------------------------------------------------
    # Load Progress
    # ------------------------------------------------------------------
    progress = load_progress(progress_file)
    resume_mode = progress is not None

    if resume_mode:
        print("\n🔄 Resuming indexing...")
        last_article_id = progress["last_article_id"]
        articles_done = progress["articles_done"]
        chunks_done = progress["chunks_done"]

        print(f" Last processed article: {last_article_id}")
        print(f" Articles done: {articles_done}")
        print(f" Chunks done: {chunks_done}")

        # load FAISS + docs
        vector_db = FaissVecDB(cfg)
        vector_db.load(output_path)
        with open(docs_file, "rb") as f:
            vector_db.documents = pickle.load(f)
        print("✔ Loaded existing DB for resuming.")

    else:
        print("\n🆕 Starting fresh index build...")
        vector_db = FaissVecDB(cfg)
        articles_done = 0
        chunks_done = 0
        last_article_id = None

    # ------------------------------------------------------------------
    # Main Loop
    # ------------------------------------------------------------------
    print("\n📄 Reading Wikipedia extract in batches...\n")

    resume_skipping = resume_mode
    pbar = tqdm(desc="Building index")

    for batch_articles in read_wikiextractor_batches(extracted_path, batch_size=batch_size):

        # ------------------------------------------------------------
        # Skip until reaching resume point
        # ------------------------------------------------------------
        if resume_skipping:
            batch_ids = [a.get("id", "") for a in batch_articles]

            if last_article_id not in batch_ids:
                # skip entire batch
                articles_done += len(batch_articles)
                pbar.update(len(batch_articles))
                continue
            else:
                # found batch with last processed article → resume *after* it
                idx = batch_ids.index(last_article_id)
                batch_articles = batch_articles[idx + 1 :]
                resume_skipping = False
                print("✔ Resuming within batch.")

        # ------------------------------------------------------------
        # Normal processing
        # ------------------------------------------------------------
        batch_chunks = []
        batch_metadata = []

        for article in batch_articles:
            title = article.get("title") or article.get("id", "")
            text = article.get("text", "")
            article_id = article.get("id", "")
            url = article.get("url", f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}")

            # Build chunks
            if article.get("lines"):
                lines = [line.split("\t")[1] for line in article["lines"].split("\n") if "\t" in line]
                chunks = []
                for line in lines:
                    chunks.extend(chunk_article(line, title))
            else:
                chunks = chunk_article(text, title)

            for i, chunk in enumerate(chunks):
                batch_chunks.append(chunk)
                batch_metadata.append({
                    "title": title,
                    "url": url,
                    "article_id": article_id,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                })

            # Update counters
            articles_done += 1
            chunks_done += len(chunks)
            last_article_id = article_id
            pbar.update(1)

        # ------------------------------------------------------------
        # Add to vector DB
        # ------------------------------------------------------------
        if batch_chunks:
            vector_db.add_documents(batch_chunks, batch_metadata)

        # ------------------------------------------------------------
        # Save progress + partial DB every batch
        # ------------------------------------------------------------
        vector_db.save(output_path)
        with open(docs_file, "wb") as f:
            pickle.dump(vector_db.documents, f)

        save_progress(progress_file, last_article_id, articles_done, chunks_done)

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" 🎉 INDEX BUILD COMPLETE ")
    print("=" * 80)
    print(f" Articles processed: {articles_done}")
    print(f" Total chunks: {chunks_done}")
    print(f" Index path: {output_path}")
    print(" Progress stored at:", progress_file)
