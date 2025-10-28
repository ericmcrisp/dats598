"""
bring in the most recent snapshot of wikipedia locally
"""
import requests
import bz2
import xml.etree.ElementTree as ET
from tqdm import tqdm

def get():
    dump_url = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2"
    output_path = "../data/wikipedia/enwiki-latest-pages-articles.xml.bz2"
    save_db_path = "../data/wikipedia"
    print("Downloading Wikipedia dump... This will take a while.")
    response = requests.get(dump_url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    with open(output_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    print(f"Downloaded to {output_path}")
    return output_path
