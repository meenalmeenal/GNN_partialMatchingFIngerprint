import json
import os

nb = {
    "cells": [],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 5,
}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.splitlines(keepends=True)}


nb["cells"] = [
    md("# Stage 4 - Embedding diagnostics\n\n"
       "Runs the Stage 4 analysis interactively on the Stage 3 checkpoint (test split "
       "only). See `results/stage4_notes.md` for the write-up; this notebook lets you "
       "re-run pieces and poke at intermediate arrays."),
    code("import sys, os, json, torch\n"
         "sys.path.insert(0, os.path.abspath('../src'))\n"
         "from match import load_model\n"
         "from evaluate import collect, embed_paths\n"
         "import diagnostics as diag"),
    code("device = torch.device('cpu')\n"
         "model, cfg = load_model('../models/best_model.pt', device)\n"
         "sp = json.load(open('../models/splits.json'))\n"
         "graph_dir = cfg['data']['graph_dir']\n"
         "gallery, probes = collect(graph_dir, sp['test'], packed='../data/packed/test.pt')\n"
         "len(gallery), len(probes)"),
    code("gf_ids = list(gallery)\n"
         "fidx = {f: i for i, f in enumerate(gf_ids)}\n"
         "ge = embed_paths(model, [gallery[f] for f in gf_ids], device)\n"
         "gf = torch.tensor([fidx[f] for f in gf_ids])\n"
         "pe = embed_paths(model, [p for p, _, _ in probes], device)\n"
         "pf = torch.tensor([fidx[f] for _, f, _ in probes])\n"
         "tags = [t for _, _, t in probes]"),
    md("## Gallery inter-identity similarity\n"
       "How confusable are different fingers' embeddings? A heavy right tail here caps "
       "Rank-1 regardless of probe quality."),
    code("diag.gallery_intra_similarity(ge, '../results/stage4')"),
    md("## t-SNE of embeddings\n"
       "Star = gallery (Real print), dot = probe (altered). Color = finger identity."),
    code("diag.tsne_plot(ge, gf, gf_ids, pe, pf, tags, '../results/stage4')"),
    md("## Genuine score by alteration severity"),
    code("diag.score_hist_by_tag(pe, pf, ge, gf, tags, '../results/stage4')"),
    md("## Rank-1 decision margin: near-misses vs blowouts"),
    code("diag.rank_gap_analysis(pe, pf, ge, gf, '../results/stage4')"),
]

os.makedirs("notebooks", exist_ok=True)
json.dump(nb, open("notebooks/01_diagnostics.ipynb", "w"), indent=1)
print("wrote notebooks/01_diagnostics.ipynb")
