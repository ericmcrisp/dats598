import faiss
import os
import multiprocessing
from app.features.factcheck_pipe import FactCheckPipe
from app.core.public_config import PublicConfig
from app.features.evidence_retrieval import EvidenceRetriever
from app.utils.configuration_syncing import sync
from app.utils.build_faiss_index import build_index

import random
from collections import defaultdict
import nltk
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
from nltk.corpus import brown
import matplotlib.pyplot as plt
from datasets import load_dataset
from datasets import concatenate_datasets
from sentence_transformers import util
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from app.core.config import settings

faiss.omp_set_num_threads(min(4, os.cpu_count()))

nltk.download('brown')


def sequential_claim_generator(claims, labels):
    for claim, label in zip(claims, labels):
        yield claim, label


def generate_text_blocks(claims, labels, brown_texts, n=5):
    claim_gen = sequential_claim_generator(claims, labels)
    claim_buffer = []  # to hold a second claim if needed
    while True:
        block = random.sample(brown_texts, n)
        claim_info_list = []

        # Decide randomly: 0, 1, or 2 claims
        num_claims = random.choice([0, 1, 2])

        for _ in range(num_claims):
            # Get next claim from buffer or generator
            if claim_buffer:
                claim_text, claim_label = claim_buffer.pop(0)
            else:
                try:
                    claim_text, claim_label = next(claim_gen)
                except StopIteration:
                    # All claims have been used, finish generation
                    if claim_info_list:
                        yield {
                            'text': ' '.join(block),
                            'number_of_claims': len(claim_info_list),
                            'claims': [c['claim'] for c in claim_info_list],
                            'labels': [c['label'] for c in claim_info_list]
                        }
                    return

            insert_idx = random.randint(0, len(block))
            block.insert(insert_idx, claim_text)
            claim_info_list.append({
                'claim': claim_text,
                'label': claim_label,
                'position': insert_idx
            })

        yield {
            'text': ' '.join(block),
            'number_of_claims': len(claim_info_list),
            'claims': [c['claim'] for c in claim_info_list],
            'labels': [c['label'] for c in claim_info_list]
        }


def test_user_def_claim_detection_and_extraction(pipe, data, resdir):
    # unpack inputs
    data_info, number_of_fluff_sentences = data
    # synthetically generate random text for testing
    claim_gen = sequential_claim_generator(data_info['fever_claims'], data_info['fever_labels'])
    block_gen = generate_text_blocks(data_info['fever_claims'],
                                     data_info['fever_labels'],
                                     data_info['brown_texts'])
    # track quantities for metrics
    all_labels = []
    all_preds = []
    all_sents = []
    gold_standard_claims = []
    generated_queries = []
    detected_claims_lst = []

    # iterate through blocks
    for block in tqdm(block_gen, desc="Testing claim detection"):
        text = block['text']
        # print(block['text'])
        # print('-'*50)
        print(block['claims'])
        print('-'*50)
        gold_standard_claims.extend(block['claims'])
        # apply pipeline to input
        pipe.clean(text)
        detected_claims = pipe.detect_claims()
    
        # for s in sentences:
        #     is_claim, confidence, claim_type = pipe.detector.is_factual_claim(s)
        #     print(f"Sentence: {s}")
        #     print(f"  is_claim: {is_claim}, confidence: {confidence}")
        # gather results
        # for claim in detected_claims:
        #     print(f'Claim text: {claim.text}')
        #     print(f'Sentence index: {claim.start_sentence_idx}')
        #     print(f'Context text: {claim.context_text}\n')
        #     is_gold = claim.text in gold_standard_claims
        #     print(f'Claim text: {claim.text} | Gold: {is_gold}')
        # if len(block['claims']) > 0:
        #     break

        labels = [1 if any(sent == gold for gold in gold_standard_claims) else 0 for sent in pipe.sentences]
        if pipe.cfg.CLAIM_MODE == 'simple':
            preds = [1 if any(sent == c.text for c in detected_claims) else 0 for sent in pipe.sentences]
        else:
            # deprecated because coref was dropped
            # preds = [1 if any(sent in c.resolved_text for c in detected_claims) else 0 for sent in pipe.sentences]
            preds = [1 if any(sent == c.text for c in detected_claims) else 0 for sent in pipe.sentences]

        # Collect for overall evaluation
        all_labels.extend(labels)
        all_preds.extend(preds)
        all_sents.extend(pipe.sentences)

        for claim in detected_claims:
            original = claim.text
            detected_claims_lst.append(original)
            if original in gold_standard_claims:
                if claim.queries:
                    generated_queries.append(claim.queries[0])
                else:
                    generated_queries.append('')
    # return
    # --- detection evaluation metrics ---
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary")

    # Print results
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1: {f1:.2f}")

    # determine extraction metrics
    if gold_standard_claims:
        gold_emb = pipe.evidence.vector_db.encoder.encode(gold_standard_claims, convert_to_tensor=True)
        ext_emb = pipe.evidence.vector_db.encoder.encode(generated_queries, convert_to_tensor=True)
        sims = util.cos_sim(gold_emb, ext_emb).diagonal().cpu().numpy()
        avg_sim = sims.mean()
        print("\n=== CLAIM EXTRACTION RESULTS ===")
        print(f"Average similarity: {avg_sim:.3f}")
        print(f"Median similarity : {float(np.median(sims)):.3f}")

    else:
        print("No extracted claims matched gold claims — cannot compute similarity.")
        sims = np.array([])

    if resdir:
        os.makedirs(os.path.join(resdir, 'claim_detection'), exist_ok=True)
        os.makedirs(os.path.join(resdir, 'claim_extraction'), exist_ok=True)
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "avg_extraction_similarity": float(avg_sim) if gold_standard_claims else None
        }
        with open(os.path.join(resdir, 'claim_detection', 'performance_metrics.json'), "w") as f:
            json.dump(metrics, f, indent=2)
        # create confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Claim Detection: Confusion Matrix")
        plt.savefig(os.path.join(resdir, "confusion_matrix.png"))
        plt.close()

        # hitting array length issues when writing the df
        df_gold_claims = pd.DataFrame({"gold_claim": gold_standard_claims})
        df_detected_claims = pd.DataFrame({"detected_claims": detected_claims_lst})

        df_gold_claims.to_csv(os.path.join(resdir, 'claim_extraction', 'gold_claims.csv'), index=False)        
        df_detected_claims.to_csv(os.path.join(resdir, 'claim_extraction', 'detected_claims.csv'), index=False)

        if len(sims) > 0:
            plt.hist(sims, bins=20, color="skyblue", edgecolor="black")
            plt.title("Claim Extraction Similarity Distribution")
            plt.xlabel("Cosine similarity")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig(os.path.join(resdir, "similarity_hist.png"))
            plt.close()

    return pipe


# def test_user_def_claim_extraction(pipe, data, resdir):
#     samples, labels = data
#     # similarity from query pairs (when query and claim exist)
#     # coverage of query for claim
#     # average similarity
#     # histogram of similarity scores
#     # % of extracted claims above threshold (e.g., 0.7)


#     return pipe


def compute_rr(returned, gold):
    for i, r in enumerate(returned):
        if r in gold:
            return 1.0 / (i + 1)
    return 0.0


def get_target_pages(claim):
    gold = set()
    for group in claim["evidence"]:
        for evidence in group:
            page = evidence[2]
            gold.add(page.replace(" ", "_"))
    return gold


# def test_user_def_evidence_retrieval(pipe, data, max_claims, resdir):
#     data_info, number_of_fluff_sentences = data
#     dataset = data_info['fever_train']
#     retriever = pipe.evidence
#     top_k = pipe.cfg.EVIDENCE_TOP_K

#     recalls = []
#     mrrs = []
#     rank1_hits = 0
#     score_stats = []

#     for item in tqdm(dataset, desc="Evaluating evidence retrieval"):
#         gold_pages = get_target_pages(item)

#         # Use your pipeline to extract claim
#         pipe.clean(claim_text)
#         pipe.segment_sentences()
#         detected = pipe.detect_claims()

#         if not detected:
#             continue

#         claim = detected[0]  # FEVER claims are single-sentence

#         # Retrieve evidence
#         evidence_list = retriever.retrieve_evidence_for_claim(claim)

#         # Extract evidence wiki titles returned by FAISS
#         returned_titles = [ev.source.title.replace(" ", "_") for ev in evidence_list]

#         # ---- Compute metrics ----
#         # recall@k
#         hit = any(t in gold_pages for t in returned_titles)
#         recalls.append(1 if hit else 0)

#         # rank-1 accuracy
#         if returned_titles and returned_titles[0] in gold_pages:
#             rank1_hits += 1

#         # MRR (mean reciprocal rank)
#         rr = compute_rr(returned_titles, gold_pages)
#         mrrs.append(rr)

#         # score statistics
#         if evidence_list:
#             score_stats.append(evidence_list[0].relevance_score)

#         # ---- aggregate metrics ----
#         results = {
#             "recall@k": sum(recalls) / len(recalls),
#             "rank1_accuracy": rank1_hits / len(recalls),
#             "mrr": sum(mrrs) / len(mrrs),
#             "avg_top_score": sum(score_stats) / len(score_stats),
#         }

#         print("\n=== Evidence Retrieval Results ===")
#         for k, v in results.items():
#             print(f"{k:20s}: {v:.4f}")

#         # Save results
#         if resdir:
#             os.makedirs(resdir, exist_ok=True)
#             with open(os.path.join(resdir, "evidence_metrics.json"), "w") as f:
#                 json.dump(results, f, indent=2)

#         return results


#     return pipe


def test_user_def_verification(pipe, data, resdir):
    samples, labels = data

    return pipe


def user_defined(model_common_name: str = "mini_L12",
                 claim_mode: str = 'simple'):
    # create custom config
    test_cfg = PublicConfig(
        embedding_model_name=model_common_name,
        claim_mode=claim_mode
    )
    # sync
    sync(test_cfg)
    return settings


def confirm_embeddings(models):
    EXTRACTED_PATH = 'data/wikipedia/extracted'
    MAX_ARTICLES = 1000000
    for name, model_name in models.items():
        user_cfg = user_defined(name)
        OUTPUT_PATH = f'app/data/vector_db/{name}'
        # Create a config override for this model
        print(f"\n=== Building FAISS index for model: {user_cfg.EMBEDDING_MODEL_COMMON_NAME} ===")
        print(user_cfg.FAISS_INDEX_PATH)
        build_index(EXTRACTED_PATH,
                    OUTPUT_PATH,
                    MAX_ARTICLES,
                    batch_size=2048,
                    cfg=user_cfg,
                    overwrite=True)
    return True


def main():
    # basic, large - top end performance for hugging face compatible, small - top performance hugging face compatible
    embedding_models = {'mini_L6': 'all-MiniLM-L6-v2'}
                        # 'e5small': 'intfloat/e5-small-v2',
                        # 'paraphase_L6': 'paraphrase-MiniLM-L6-v2',
                        # 'mini_L12': 'all-MiniLM-L12-v2'}
                        # 'Gemma3': 'tencent/KaLM-Embedding-Gemma3-12B-2511'}
    # make sure each embedding exists
    # if confirm_embeddings(embedding_models):
    #     print("All embedding models confirmed.")
    # else:
    #     print("Error confirming embedding models.")
    # return
    # load the datasets needed for testing
    fever_configs = ['v1.0', 'v2.0', 'wiki_pages']
    fever = load_dataset("fever", fever_configs[0], cache_dir="data/hf_cache")
    # dataset = concatenate_datasets([fever['train'], fever['dev'], fever['test']])
    # choose number of samples to use in testing
    LIMIT = min(50, len(fever['train']))
    training_set = fever['train'].shuffle(seed=42)[:LIMIT]
    # test_set = split_data['test']
    train_claims = training_set['claim']
    train_labels = list(training_set['label'])
    brown_sents = brown.sents()
    brown_texts = [' '.join(sent) for sent in brown_sents]
    dataset_info = {'fever_claims': train_claims,
                    'fever_labels': train_labels,
                    'fever_train': training_set,
                    'brown_sents': brown_sents,
                    'brown_texts': brown_texts}
    # analyze the process and subprocesses for each embedding
    for model in embedding_models.keys():
        print(f"Testing with embedding model: {model}")
        cfg = user_defined(model_common_name=model,
                           claim_mode='advanced')
        pipe = FactCheckPipe(cfg=cfg)
        data = [dataset_info, 3]
        test_user_def_claim_detection_and_extraction(pipe, data,
                                                     "results/prod")
        break
        # test_user_def_evidence_retrieval(pipe, data, "results/prod/evidence_retrieval")
        # test_user_def_verification(pipe, data, "results/prod/verification")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()

