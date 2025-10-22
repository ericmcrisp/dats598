""" 
Test the claim detection process.
"""

from claim_pipe import ClaimDetectionPipeline as CDP

import os
import json
import numpy as np
import pandas as pd
from nltk.corpus import brown
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def test_claim_detection(labels, text, dir=None):
    pipe = CDP()
    preds = []
    for sentence in text:
        is_claim, _, _ = pipe.detector.is_factual_claim(sentence)
        preds.append(is_claim)
    # analyze results
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    # print out results
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1: {f1:.2f}")
    # now save rhe data
    if dir:
        os.makedirs(dir, exist_ok=True)
        # save summary to file
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        with open(os.path.join(dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        # get confusion matrix
        cm = confusion_matrix(labels, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.savefig(os.path.join(dir, "confusion_matrix.png"))
        plt.close()

        # manual inspection of results
        df = pd.DataFrame({
            "text": text,
            "label": labels,
            "prediction": preds
        })
        df.to_csv(os.path.join(dir, "detection_data.csv"), index=False)


def main(n_claims=2000):
    fever_configs = ['v1.0', 'v2.0', 'wiki_pages']
    fever = load_dataset("fever", fever_configs[0], cache_dir="./data/hf_cache") # , download_mode="force_redownload"    
    positive_claims = fever["train"].shuffle(seed=42).select(range(n_claims))["claim"]
    # use nltk dataset for non-claims: use unrelated sentences
    non_claims = [" ".join(sent) for sent in brown.sents()[:n_claims]]
    data = positive_claims + non_claims
    labels = [1]*len(positive_claims) + [0]*len(non_claims)
    test_claim_detection(labels, data, dir="../results/development/claim_detection")
