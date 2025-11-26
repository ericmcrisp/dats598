import faiss
import os
import multiprocessing
from app.features.factcheck_pipe import FactCheckPipe
from app.core.public_config import PublicConfig
from app.features.evidence_retrieval import EvidenceRetriever
from app.utils.configuration_syncing import sync
from app.utils.build_faiss_index import build_index
from app.utils.build_resumable_index import build_index_resumable

import random
from collections import defaultdict, Counter
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
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.multiclass import unique_labels
from app.core.config import settings

faiss.omp_set_num_threads(min(4, os.cpu_count()))

nltk.download('brown')

# label mapping:
VERDICT_TO_ID = {
    "SUPPORTS": 0,
    "REFUTES": 1,
    "NOT_ENOUGH_INFO": 2
}

ID_TO_VERDICT = {v: k for k, v in VERDICT_TO_ID.items()}



def sequential_claim_generator(claims, labels, evidence):
    for claim, label, evidence in zip(claims, labels, evidence):
        yield claim, label, evidence


def generate_text_blocks(claims, labels, evidence, brown_texts, n=5):
    claim_gen = sequential_claim_generator(claims, labels, evidence)
    claim_buffer = []  # to hold a second claim if needed
    while True:
        block = random.sample(brown_texts, n)
        claim_info_list = []
        # randomly: 0, 1, or 2 claims
        num_claims = random.choice([0, 1, 2])
        for _ in range(num_claims):
            # get next claim from generator
            if claim_buffer:
                claim_text, claim_label, claim_ids = claim_buffer.pop(0)
            else:
                try:
                    claim_text, claim_label, claim_ids = next(claim_gen)
                except StopIteration:
                    # all claims in set have been used, finish generation
                    if claim_info_list:
                        yield {
                            'text': ' '.join(block),
                            'number_of_claims': len(claim_info_list),
                            'claims': [c['claim'] for c in claim_info_list],
                            'labels': [c['label'] for c in claim_info_list],
                            'ids': [c['ids'] for c in claim_info_list]

                        }
                    return

            insert_idx = random.randint(0, len(block))
            block.insert(insert_idx, claim_text)
            claim_info_list.append({
                'claim': claim_text,
                'label': claim_label,
                'position': insert_idx,
                'ids': claim_ids
            })

        yield {
            'text': ' '.join(block),
            'number_of_claims': len(claim_info_list),
            'claims': [c['claim'] for c in claim_info_list],
            'labels': [c['label'] for c in claim_info_list],
            'ids': [c['ids'] for c in claim_info_list]
        }


def test_process(pipe, data, resdir):
    # unpack inputs
    data_info, number_of_fluff_sentences = data
    FEVER_ID_TO_VERDICT = {}
    for i, label in enumerate(set(set(data_info['fever_labels']))):
        FEVER_ID_TO_VERDICT[label] = i
    fever_labels = [FEVER_ID_TO_VERDICT[l] for l in data_info['fever_labels']]
    # synthetically generate random text for testing
    claim_gen = sequential_claim_generator(data_info['fever_claims'], fever_labels, data_info['fever_evid_id'])
    block_gen = generate_text_blocks(data_info['fever_claims'],
                                     fever_labels,
                                     data_info['fever_evid_id'],
                                     data_info['brown_texts'])
    # track quantities for metrics
    all_labels = []
    all_preds = []
    all_sents = []
    gold_standard_claims = []
    gold_standard_ids = []
    generated_queries = []
    detected_claims_lst = []
    hits, mrrs, sims = [], [], []
    gold_standard_verdicts = []
    all_gold_verdicts = []
    all_pred_verdicts = []

    # iterate through blocks
    for block in tqdm(block_gen, desc="Testing claim detection"):
        text = block['text']
        # print(block['text'])
        # print('-'*50)
        print(block['claims'])
        print('-'*50)
        gold_standard_claims.extend(block['claims'])
        gold_standard_ids.extend(block['ids'])
        gold_standard_verdicts.extend(block['ids'])
        # apply pipeline to input
        pipe.clean(text)
        pipe.detect_claims()
    
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
            preds = [1 if any(sent == c.text for c in pipe.detected_claims) else 0 for sent in pipe.sentences]
        else:
            # deprecated because coref was dropped
            # preds = [1 if any(sent in c.resolved_text for c in detected_claims) else 0 for sent in pipe.sentences]
            preds = [1 if any(sent == c.text for c in pipe.detected_claims) else 0 for sent in pipe.sentences]

        # Collect for overall evaluation
        all_labels.extend(labels)
        all_preds.extend(preds)
        all_sents.extend(pipe.sentences)

        for claim in pipe.detected_claims:
            original = claim.text
            detected_claims_lst.append(original)
            generated_queries.append(claim.queries[0])
        # process the embeddings
        pipe.retrieve_evidence(pipe.detected_claims)
        
        for claim in pipe.claims_with_evidence:
            evidence_list = claim.evidence
            retrieved_ids = [e.source.title for e in evidence_list]
            # recall@K
            hit = any(g in retrieved_ids for g in gold_standard_ids)
            hits.append(1 if hit else 0)
            # mmr
            ranks = [i+1 for i, r in enumerate(retrieved_ids) if r in gold_standard_ids]
            mrrs.append(1 / min(ranks) if ranks else 0)
            # similarity
            sims.extend([e.similarity for e in evidence_list])
 
        claim_to_gold_idx = []
        for detected in detected_claims_lst:
            try:
                idx = gold_standard_claims.index(detected)
            except ValueError:
                idx = None
            claim_to_gold_idx.append(idx)
        # analyze the documents found
        verification_results = pipe.verify_claims(pipe.claims_with_evidence)
        # post process
        for v, gold_idx in zip(verification_results, claim_to_gold_idx):
            if gold_idx is not None:
                all_gold_verdicts.append(gold_standard_verdicts[gold_idx])
                all_pred_verdicts.append(FEVER_ID_TO_VERDICT[v.verdict])

        # gather output
        frontend_output = pipe.build_factcheck_response()

    classification_accuracy = sum([g==p for g, p in zip(all_gold_verdicts, all_pred_verdicts)]) / len(all_gold_verdicts)

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
        os.makedirs(os.path.join(resdir, 'classification'), exist_ok=True)
        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "avg_extraction_similarity": float(avg_sim) if gold_standard_claims else None,
            "median_extraction_similarity": float(np.median(sims)) if gold_standard_claims else None,
            "recall_at_k": float(np.mean(hits)) if hits else None,
            "mrr": float(np.mean(mrrs)) if mrrs else None,
            "avg_evidence_score": float(np.mean(sims)) if len(sims) > 0 else None,
            "num_gold_claims": int(len(gold_standard_claims)),
            "num_detected_claims": int(len(detected_claims_lst))
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

        rows = []
        for claim in pipe.claims_with_evidence:
            evidences = claim.evidence or []
            avg_relevance = float(np.mean([e.similarity for e in evidences])) if evidences else 0.0
            rows.append({
                "claim": claim.text,
                "retrieved_evidence_count": len(evidences),
                "avg_similarity": avg_relevance,
            })
        df_evidence = pd.DataFrame(rows)
        df_evidence.to_csv(os.path.join(resdir, 'claim_extraction', 'prod_evidence_stats.csv'), index=False)

        if len(sims) > 0:
            plt.hist(sims, bins=20, color="skyblue", edgecolor="black")
            plt.title("Claim Extraction Similarity Distribution")
            plt.xlabel("Cosine similarity")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig(os.path.join(resdir, "claim_extraction", "prod_similarity_hist.png"))
            plt.close()

        sim_stats = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "avg_extraction_similarity": float(avg_sim) if gold_standard_claims else None
        }
        with open(os.path.join(resdir, 'claim_extraction', 'similarity_stats.json'), 'w') as f:
            json.dump(sim_stats, f, indent=2)

        with open(os.path.join(resdir, 'classification', 'prod_classification_accuracy.json'), "w") as f:
            json.dump({"classification_accuracy": classification_accuracy}, f, indent=2)


        # if all_gold_verdicts:
        #     print("\n=== CLAIM VERIFICATION METRICS ===")
        #     print(classification_report(all_gold_verdicts, all_pred_verdicts, zero_division=0))

        #     cm = confusion_matrix(all_gold_verdicts, all_pred_verdicts, labels=["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"])
        #     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"])
        #     disp.plot(cmap=plt.cm.Blues)
        #     plt.title("Claim Verification Confusion Matrix")
        #     plt.savefig(os.path.join(resdir, "classification", "verification_confusion_matrix.png"))
        #     plt.close()

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
    EXTRACTED_PATH = os.path.abspath('data/wikipedia/extracted')
    # EXTRACTED_PATH = os.path.abspath('data/hf_cache/wiki-pages/')
    MAX_ARTICLES = None
    for name, model_name in models.items():
        user_cfg = user_defined(name)
        OUTPUT_PATH = f'app/data/wiki_page/vector_db/{name}'
        # Create a config override for this model
        print(f"\n=== Building FAISS index for model: {user_cfg.EMBEDDING_MODEL_COMMON_NAME} ===")
        print(user_cfg.FAISS_INDEX_PATH)
        # build_index_resumable(EXTRACTED_PATH,
        #                       OUTPUT_PATH,
        #                       MAX_ARTICLES,
        #                       batch_size=64,
        #                       cfg=user_cfg,
        #                       overwrite=True)
        build_index_resumable(EXTRACTED_PATH,
                              OUTPUT_PATH,
                              cfg=user_cfg,
                              batch_size=64)
    return True


def main():
    # basic, large - top end performance for hugging face compatible, small - top performance hugging face compatible
    embedding_models = {'mini_L6': 'all-MiniLM-L6-v2'}
                        # 'e5small': 'intfloat/e5-small-v2',
                        # 'paraphase_L6': 'paraphrase-MiniLM-L6-v2',
                        # 'mini_L12': 'all-MiniLM-L12-v2',
                        # 'Gemma3': 'tencent/KaLM-Embedding-Gemma3-12B-2511'}
    # make sure each embedding exists
    # if confirm_embeddings(embedding_models):
    #     print("All embedding models confirmed.")
    # else:
    #     print("Error confirming embedding models.")
    # return
    # load the datasets needed for testing
    # fever_configs = ['v1.0', 'v2.0', 'wiki_pages']
    # fever = load_dataset("fever", fever_configs[0], cache_dir="data/hf_cache")
    # # dataset = concatenate_datasets([fever['train'], fever['dev'], fever['test']])
    # # choose number of samples to use in testing
    # LIMIT = min(50, len(fever['train']))
    # training_set = fever['train'].shuffle(seed=42)[:LIMIT]
    # # print(training_set.keys())
    # train_claims = training_set['claim']
    # train_evidence = training_set['evidence_id']
    # train_labels = list(training_set['label'])
    # brown_sents = brown.sents()
    # brown_texts = [' '.join(sent) for sent in brown_sents]
    # dataset_info = {'fever_claims': train_claims,
    #                 'fever_labels': train_labels,
    #                 'fever_evid_id': train_evidence,
    #                 'fever_train': training_set,
    #                 'brown_sents': brown_sents,
    #                 'brown_texts': brown_texts}
    # # analyze the process and subprocesses for each embedding
    # for model in embedding_models.keys():
    #     print(f"Testing with embedding model: {model}")
    #     cfg = user_defined(model_common_name=model,
    #                        claim_mode='advanced')
    #     print(f"Vector database is located at: {cfg.VECTOR_DB_DIR}")
    #     pipe = FactCheckPipe(cfg=cfg)
    #     data = [dataset_info, 3]
    #     test_process(pipe, data, "results/prod")
        # test_user_def_evidence_retrieval(pipe, data, "results/prod/evidence_retrieval")
        # test_user_def_verification(pipe, data, "results/prod/verification")
    

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()

