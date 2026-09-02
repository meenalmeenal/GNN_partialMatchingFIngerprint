"""
Evaluate the trained model on a held-out test split:
- Rank-1 identification accuracy (closed-set)
- Verification ROC curve + Equal Error Rate (EER)
"""
import argparse
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from sklearn.metrics import roc_curve

from match import load_model, embed_graph_file, build_gallery, rank_matches


def evaluate_rank1(model, test_dir, gallery, device):
    correct, total = 0, 0
    for path in glob.glob(os.path.join(test_dir, "**", "*.pt"), recursive=True):
        true_subject = os.path.basename(os.path.dirname(path))
        query_emb = embed_graph_file(model, path, device)
        results = rank_matches(query_emb, gallery, top_k=1)
        if results and results[0][0] == true_subject:
            correct += 1
        total += 1
    return correct / total if total else 0.0


def evaluate_verification(model, test_dir, device):
    """Builds genuine/impostor score distributions and computes EER."""
    files_by_subject = {}
    for path in glob.glob(os.path.join(test_dir, "**", "*.pt"), recursive=True):
        subject = os.path.basename(os.path.dirname(path))
        files_by_subject.setdefault(subject, []).append(path)

    embeddings = {
        s: [embed_graph_file(model, p, device) for p in paths]
        for s, paths in files_by_subject.items()
    }

    scores, labels = [], []
    subjects = list(embeddings.keys())
    for s in subjects:
        embs = embeddings[s]
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                scores.append(torch.dot(embs[i], embs[j]).item())
                labels.append(1)  # genuine pair

    for i, s1 in enumerate(subjects):
        for s2 in subjects[i + 1:]:
            scores.append(torch.dot(embeddings[s1][0], embeddings[s2][0]).item())
            labels.append(0)  # impostor pair

    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    return eer, thresholds[eer_idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/best_model.pt")
    parser.add_argument("--gallery", required=True)
    parser.add_argument("--test", required=True, help="Held-out test graphs (partial prints)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(args.checkpoint, device)

    gallery = build_gallery(model, args.gallery, device)
    rank1 = evaluate_rank1(model, args.test, gallery, device)
    eer, threshold = evaluate_verification(model, args.test, device)

    print(f"Rank-1 accuracy: {rank1 * 100:.2f}%")
    print(f"EER: {eer * 100:.2f}%  (threshold={threshold:.4f})")


if __name__ == "__main__":
    main()
