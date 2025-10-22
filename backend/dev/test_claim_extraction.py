""" 
Test the claim detection process. This should use similarity to test whether the
detected claims match the ground truth claims given the same embedding
"""

from claim_pipe import ClaimDetectionPipeline as CDP

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util


def test_claim_extraction(original_claims, dir=None):
    pipe = CDP()
    cleaned_claims = []
    expected_claims = []
    for c in original_claims: 
        cleaned_text = pipe.preprocessor.clean_text(c)
        sentences = pipe.preprocessor.segment_sentences(cleaned_text)
        # print(f'The number of sentence in this claim is: {len(sentences)}')
        for sentence in sentences:
            is_claim, confidence, claim_type = pipe.detector.is_factual_claim(sentence)
            if is_claim and confidence > pipe.detector.claim_threshold:
                components = pipe.extractor.extract_claim_components(sentence)
                search_queries = pipe.extractor.generate_search_queries(components)
                if search_queries:
                    cleaned_claims.append(search_queries[0])
                    expected_claims.append(c)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    original_embeds = model.encode(expected_claims, convert_to_tensor=True)
    cleaned_embeds = model.encode(cleaned_claims, convert_to_tensor=True)
    # get similarities for each claim
    similarities = util.cos_sim(original_embeds, cleaned_embeds).diagonal()
    # determine average similarity and plot distribution
    average_similarity = similarities.mean().item()
    print(f'Average semantic similarity: {np.round(average_similarity,3)}')
    # save summary/numeric data
    with open(os.path.join(dir, "summary.txt"), "w") as f:
        f.write(f"Average semantic similarity: {average_similarity:.4f}\n")
    # save to df
    df = pd.DataFrame({
        "expected_claim": expected_claims,
        "cleaned_claim": cleaned_claims,
        "similarity": similarities.cpu().numpy()
    })
    df.to_csv(os.path.join(dir, "similarities_data.csv"), index=False)
    # save plot
    plt.figure(figsize=(8, 6))
    plt.hist(similarities.cpu(), bins=20, color="skyblue", edgecolor="black")
    plt.title("FEVER (original) and Cleaned")
    plt.xlabel("cosine similarity")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(dir, "similarity_hist.png"))
    plt.close()


def main(n_claims=2000):
    fever_configs = ['v1.0', 'v2.0', 'wiki_pages']
    fever = load_dataset("fever", fever_configs[0], cache_dir="./data/hf_cache") # , download_mode="force_redownload"    
    original_claims = fever["train"].select(range(n_claims))["claim"]
    test_claim_extraction(original_claims, dir="../results/development/claim_extraction")
