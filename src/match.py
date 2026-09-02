"""
Embed a partial fingerprint's graph and retrieve the closest matches from a
gallery of pre-embedded full-print graphs (cosine similarity ranking).
"""
import argparse
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import yaml
from torch_geometric.data import Batch

from model import TripletGCN
from graph_build import minutiae_to_graph
from minutiae import minutiae_from_image


def load_model(checkpoint_path: str, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = TripletGCN(
        in_dim=cfg["data"]["node_feat_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        embedding_dim=cfg["model"]["embedding_dim"],
        num_layers=cfg["model"]["num_gcn_layers"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg


def embed_graph_file(model, path, device):
    data = torch.load(path, weights_only=False)
    batch = Batch.from_data_list([data]).to(device)
    with torch.no_grad():
        return model.embed(batch).squeeze(0)


def embed_image(model, image_path, k, device):
    minutiae = minutiae_from_image(image_path)
    graph = minutiae_to_graph(minutiae, k=k)
    batch = Batch.from_data_list([graph]).to(device)
    with torch.no_grad():
        return model.embed(batch).squeeze(0)


def build_gallery(model, gallery_dir, device):
    """gallery_dir/<subject_id>/<sample>.pt -> {subject_id: [embeddings]}"""
    gallery = {}
    for path in glob.glob(os.path.join(gallery_dir, "**", "*.pt"), recursive=True):
        subject = os.path.basename(os.path.dirname(path))
        emb = embed_graph_file(model, path, device)
        gallery.setdefault(subject, []).append(emb)
    return gallery


def rank_matches(query_emb, gallery: dict, top_k: int = 5):
    scores = []
    for subject, embs in gallery.items():
        best = max(torch.dot(query_emb, e).item() for e in embs)  # embeddings are L2-normalized
        scores.append((subject, best))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/best_model.pt")
    parser.add_argument("--partial", required=True, help="Path to partial print image, or .pt graph file")
    parser.add_argument("--gallery", required=True, help="Directory of gallery .pt graphs")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.checkpoint, device)

    if args.partial.endswith(".pt"):
        query_emb = embed_graph_file(model, args.partial, device)
    else:
        query_emb = embed_image(model, args.partial, cfg["data"]["k_neighbors"], device)

    gallery = build_gallery(model, args.gallery, device)
    results = rank_matches(query_emb, gallery, top_k=args.top_k)

    print("Top matches (subject_id, cosine_similarity):")
    for subject, score in results:
        print(f"  {subject}: {score:.4f}")


if __name__ == "__main__":
    main()
