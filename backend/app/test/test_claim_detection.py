""" 
Test the claim detection process.
"""

# from app.features.claim_pipe import ClaimDetectionPipeline as CDP
from app.features.factcheck_pipe import FactCheckPipe as fcp

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
    pipe = fcp()
    # --- Use the pipeline step-wise methods ---
    pipe.clean_text(text)
    pipe.segment_sentences()
    detected_claims = pipe.detect_claims()

    # Prepare predictions for evaluation
    preds = [1 if c else 0 for c in [True if claim else False for claim in detected_claims]]
    # Actually, we want 1 if sentence is detected as a claim, 0 otherwise
    # Since detect_claims returns Claim objects for detected claims only,
    # we need to map all sentences back to 1/0:
    preds = []
    for sentence in pipe.sentences:
        is_claim = any(sentence == c.text for c in detected_claims)
        preds.append(1 if is_claim else 0)

    # --- Evaluation metrics ---
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")

    # Print results
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1: {f1:.2f}")

    # --- Save metrics and outputs if dir specified ---
    if dir:
        os.makedirs(dir, exist_ok=True)

        # Save summary metrics
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        with open(os.path.join(dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        # Confusion matrix
        cm = confusion_matrix(labels, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.savefig(os.path.join(dir, "confusion_matrix.png"))
        plt.close()

        # Manual inspection DataFrame
        df = pd.DataFrame({
            "text": pipe.sentences,
            "label": labels,
            "prediction": preds
        })
        df.to_csv(os.path.join(dir, "detection_data.csv"), index=False)


def main(n_claims=2000):
    fever_configs = ['v1.0', 'v2.0', 'wiki_pages']
    fever = load_dataset("fever", fever_configs[0], cache_dir="./data/hf_cache")

    positive_claims = fever["train"].shuffle(seed=42).select(range(n_claims))["claim"]
    non_claims = [" ".join(sent) for sent in brown.sents()[:n_claims]]

    data = positive_claims + non_claims
    labels = [1]*len(positive_claims) + [0]*len(non_claims)

    test_claim_detection(labels, data, dir="../results/development/claim_detection")