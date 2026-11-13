"""
Demo (Stage 7, minutiae + models upgraded in Stage 9e): give the system a partial fingerprint, get the closest enrolled prints back.

    streamlit run app/app.py          (from the repository root)

Everything shown is computed live: minutiae graph of the query -> (optionally cropped to a
partial print) -> learned matcher (relative-geometry GNN, per-minutia descriptors, MaxSim
score) and, for comparison, the classical Hough-alignment matcher, both against the whole
gallery.
"""
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import demo_core as D
from model_v2 import load_matcher

st.set_page_config(page_title="Partial Fingerprint Matching", page_icon="🧬", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2.2rem; max-width: 1300px;}
div[data-testid="stImage"] img {border-radius: 6px;}
.hit  {color:#1b7f4b; font-weight:600;}
.miss {color:#6b7280;}
.small{font-size:0.82rem; line-height:1.25;}
</style>""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- cached resources
@st.cache_resource(show_spinner="Loading dataset ...")
def get_dataset(key):
    return D.load_dataset(key)


@st.cache_resource(show_spinner="Loading model ...")
def get_model(label):
    return load_matcher(D.model_path(label))


@st.cache_resource(show_spinner="Encoding the gallery ...")
def get_gallery(ds_key, model_label):
    ds = get_dataset(ds_key)
    return D.Gallery(ds, get_model(model_label), D.tuned_classical_params() if ds.real else None)


@st.cache_data(show_spinner=False, max_entries=512)
def gallery_image(ds_key, finger, key):
    return get_dataset(ds_key).image(finger, key)


def fig_to_st(fig):
    st.pyplot(fig, clear_figure=True, use_container_width=True)


# ----------------------------------------------------------------------------- sidebar
choices = D.dataset_choices()
models = D.available_models()
with st.sidebar:
    st.header("Setup")
    ds_key = st.selectbox("Gallery / dataset", list(choices), format_func=choices.get,
                          help="The enrolled full prints that the query is searched against.")
    real = ds_key != "socofing"
    want = D.FVC_MODEL if real else D.SOCO_MODEL
    default_model = models.index(want) if want in models else 0
    model_label = st.selectbox("Learned model", models, index=default_model, key=f"model_{real}")
    source = st.radio("Query print", ["Sample from the dataset", "Upload an image"], key="source")

    st.header("Partial print")
    area = st.slider("Print area kept (%)", 10, 100, 50, 5,
                     help="The query is cut down to a square window covering this share of the "
                          "print's own extent; the enrolled templates stay complete.")
    wx = st.slider("Window position - horizontal", 0, 100, 50, 5, disabled=area == 100)
    wy = st.slider("Window position - vertical", 0, 100, 50, 5, disabled=area == 100)

    st.header("Results")
    topk = st.slider("Matches to show", 3, 10, 5)
    with_classical = st.checkbox("Compare with classical matcher", True)

    st.divider()
    if real:
        st.warning("**Real, independent impressions.** On held-out FVC fingers (50-print galleries, full prints) "
                   "the fine-tuned model reaches ~88 % Rank-1 (classical, tuned: 79 %; chance 2 %). Caveats: "
                   "it was fine-tuned on other fingers of the same sensors, and the minutiae detector (LEADER) "
                   "was itself trained on three of these databases (2002 DB3, 2004 DB1, DB3); on the others "
                   "the fine-tuned model reaches ~84 %.")
    else:
        st.info("SOCOFing queries are *edited copies* of the enrolled print, which makes this the optimistic "
                "setting: Rank-1 about 88 % on full prints, 75 % at half the print, 45 % at 10 % "
                "(897-print gallery). For independent impressions choose an FVC gallery.")

ds = get_dataset(ds_key)
gal = get_gallery(ds_key, model_label)

# ----------------------------------------------------------------------------- header
st.title("Partial fingerprint matching with graph neural networks")
st.caption(f"Gallery: **{ds.title}** - {len(gal.fingers)} enrolled prints - model: {model_label}")

# ----------------------------------------------------------------------------- choose the query
left, right = st.columns([5, 7], gap="large")

true_idx = None          # gallery index of the true finger, when known
with left:
    st.subheader("1. Query")
    if source == "Sample from the dataset":
        c1, c2 = st.columns([3, 2])
        if f"finger_{ds_key}" not in st.session_state:
            st.session_state[f"finger_{ds_key}"] = ds.fingers[7 % len(ds.fingers)]

        def _pick_random_finger(ds=ds, ds_key=ds_key):
            # runs as the button's on_click callback, i.e. BEFORE widgets are instantiated this run,
            # so it may still update the selectbox's session_state key (doing it after raises
            # StreamlitAPIException: cannot be modified after the widget ... is instantiated).
            st.session_state[f"finger_{ds_key}"] = ds.fingers[int(np.random.randint(len(ds.fingers)))]

        st.button("🎲 Random finger", on_click=_pick_random_finger)
        with c1:
            finger = st.selectbox("Finger", ds.fingers, key=f"finger_{ds_key}", format_func=ds.label)
        keys = ds.query_keys(finger)
        with c2:
            qkey = st.selectbox("Probe", keys, key=f"qkey_{ds_key}",
                                format_func=lambda k: str(k).replace("Altered-", "").replace("_", " ")
                                if isinstance(k, str) else f"impression {k}")
        q_full = ds.graph(finger, qkey)
        q_img = gallery_image(ds_key, finger, qkey)
        true_idx = gal.fingers.index(finger)
    else:
        up = st.file_uploader("Fingerprint image (png / jpg / bmp / tif)",
                              type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
        if up is None:
            st.info("Upload a fingerprint image, or switch to *Sample from the dataset*. The image is "
                    + ("resized to 160 x 160 and passed through the LEADER minutiae detector, like SOCOFing." if not real else
                       "rescaled with this database's ridge-spacing factor, then LEADER (assumes the same sensor).")
                    )
            st.stop()
        gray = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            st.error("Could not read that image.")
            st.stop()
        with st.spinner("Extracting minutiae ..."):
            q_full, q_img = ds.extract(gray)
        if q_full is None or q_full.num_nodes < 3:
            st.error("Fewer than 3 minutiae were found - try a larger or clearer print.")
            st.stop()
        known = st.selectbox("True identity (optional, for scoring)", [None] + list(gal.fingers),
                             format_func=lambda f: "unknown" if f is None else ds.label(f))
        true_idx = None if known is None else gal.fingers.index(known)

    q_graph, keep, rect = D.crop_window(q_full, area / 100, wx / 100, wy / 100)
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    D.draw_minutiae(ax, q_img, q_full, keep=keep if area < 100 else None, rect=rect)
    fig.tight_layout(pad=0.2)
    fig_to_st(fig)
    st.markdown(f"<span class='small'>{q_full.num_nodes} minutiae extracted, "
                f"<b>{q_graph.num_nodes}</b> in the partial query ({area} % of the area). "
                f"<span style='color:{D.END_COL}'>●</span> ridge ending &nbsp; "
                f"<span style='color:{D.BIF_COL}'>●</span> bifurcation &nbsp; "
                f"<span style='color:{D.WIN_COL}'>▢</span> kept window. "
                f"Grey = discarded.</span>", unsafe_allow_html=True)
    if q_graph.num_nodes < 6:
        st.warning("Very few minutiae in this window - the result is close to a guess. "
                   "Increase the area or move the window over a richer region.")

# ----------------------------------------------------------------------------- search
t0 = time.time()
s_l, enc_q = gal.search_learned(q_graph)
t_l = time.time() - t0
t0 = time.time()
s_c = gal.search_classical(q_graph) if with_classical else None
t_c = time.time() - t0
z_l = D.zscores(s_l)
order_l = np.argsort(-s_l)
if s_c is not None:
    z_c = D.zscores(s_c)
    order_c = np.argsort(-s_c)


def result_cards(order, scores, z, k):
    cols = st.columns(min(k, 5))
    for r in range(k):
        with cols[r % len(cols)]:
            gi = int(order[r])
            f = gal.fingers[gi]
            im = gallery_image(ds_key, f, ds.gallery_key)
            ok = true_idx is not None and gi == true_idx
            if im is not None:
                st.image(im, use_column_width=True, clamp=True)
            tag = "<br>✅ true match" if ok else ""
            cls = "hit" if ok else "miss"
            name = ds.label(f).replace("_", " ")
            st.markdown(f"<div class='{cls} small'><b>#{r + 1}</b> {name}<br>"
                        f"score {scores[gi]:.3f}<br>z {z[gi]:+.1f}{tag}</div>", unsafe_allow_html=True)


with right:
    st.subheader("2. Closest matches")
    m1, m2, m3 = st.columns(3)
    if true_idx is not None:
        rl = D.rank_of(s_l, true_idx)
        m1.metric("True finger - learned rank", f"#{rl}", help=f"of {len(gal.fingers)} enrolled prints")
        if s_c is not None:
            rc = D.rank_of(s_c, true_idx)
            m2.metric("True finger - classical rank", f"#{rc}")
    else:
        m1.metric("Top match z-score", f"{z_l[order_l[0]]:+.1f}",
                  help="how many standard deviations above the average gallery score; "
                       "higher = more distinctive")
    m3.metric("Search time", f"{(t_l + t_c) * 1000:.0f} ms",
              help=f"learned {t_l * 1000:.0f} ms" + (f", classical {t_c * 1000:.0f} ms" if s_c is not None else ""))

    th = D.reject_thresholds(ds_key, area)
    if th is not None:
        zt = float(z_l[order_l[0]])
        if zt >= th["z_far1"]:
            st.success(f"**Likely enrolled.** Top match z = {zt:+.1f} is above the strict threshold "
                       f"({th['z_far1']:+.1f}): about 1 in 100 prints that are *not* enrolled would score this high.")
        elif zt >= th["z_far10"]:
            st.warning(f"**Uncertain.** Top match z = {zt:+.1f} is between the lenient ({th['z_far10']:+.1f}, "
                       f"10 % of unknown prints pass) and strict ({th['z_far1']:+.1f}, 1 % pass) thresholds.")
        else:
            st.error(f"**Probably not enrolled.** Top match z = {zt:+.1f} is below {th['z_far10']:+.1f}, a level "
                     "that 90 % of prints not in the gallery stay under.")
        st.caption(f"Reject thresholds calibrated at {area} % print area with half of the fingers enrolled and the "
                   f"rest unknown; at these thresholds enrolled prints are correctly identified and accepted "
                   f"{th['dir_far10'] * 100:.0f} % (lenient) / {th['dir_far1'] * 100:.0f} % (strict) of the time.")

    st.markdown("**Learned matcher** (relative-geometry GNN, MaxSim over minutia descriptors)")
    result_cards(order_l, s_l, z_l, topk)
    if s_c is not None:
        st.markdown("**Classical matcher** (Hough alignment, no training" + (", tolerances tuned on FVC set B" if ds.real else "") + ")")
        result_cards(order_c, s_c, z_c, topk)

# ----------------------------------------------------------------------------- explanation
st.divider()
st.subheader("3. Why did it pick this one?")
ex_l, ex_r = st.columns([7, 5], gap="large")
with ex_l:
    pick = st.selectbox("Inspect a match", range(topk), format_func=lambda r: (
        f"#{r + 1} - {ds.label(gal.fingers[int(order_l[r])])}"
        + ("  (true match)" if true_idx is not None and int(order_l[r]) == true_idx else "")))
    gi = int(order_l[pick])
    gimg = gallery_image(ds_key, gal.fingers[gi], ds.gallery_key)
    qd, qm = enc_q
    gd, gm = gal.enc
    q_desc = qd[0][qm[0]].detach().numpy()
    g_desc = gd[gi][gm[gi]].detach().numpy()
    fig = plt.figure(figsize=(9, 4.6))
    mean_best = D.draw_correspondence(fig, q_img, q_graph, gimg, gal.graphs[gi], q_desc, g_desc,
                                      title=f"template {ds.label(gal.fingers[gi])}")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.01, wspace=0.05)
    fig_to_st(fig)
    st.caption("Lines join each query minutia to the template minutia whose learned descriptor is most "
               "similar (the 14 strongest links shown). The score is the average of these best "
               f"similarities over all {q_graph.num_nodes} query minutiae = {mean_best:.3f}. The descriptors "
               "depend only on relative geometry, so no alignment of the two prints is needed.")
with ex_r:
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.hist(z_l, bins=40, color="#9aa0a6")
    ax.axvline(z_l[order_l[0]], color=D.BIF_COL, lw=2, label="top match")
    if true_idx is not None:
        ax.axvline(z_l[true_idx], color="#1b7f4b", ls="--", lw=2, label="true finger")
    ax.set_xlabel("score (z, vs. all enrolled prints)")
    ax.set_ylabel("enrolled prints")
    ax.legend(frameon=False, fontsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig_to_st(fig)
    st.caption("Score of the query against every enrolled print. A confident identification sticks out "
               "far to the right of the crowd; a guess does not.")

with st.expander("How it works, and what to trust"):
    st.markdown("""
**Pipeline.** fingerprint image → minutiae (ridge endings / bifurcations) found by LEADER, a pretrained neural
detector (pyfing) → graph (minutiae = nodes, 5 nearest neighbours = edges) → graph neural network that sees only
*relative* geometry (distances and angles between neighbouring minutiae, so rotation and shift do not matter) →
one descriptor per minutia → the score of a query against a template is the mean, over query minutiae, of the best
cosine similarity to any template minutia. Because it is a mean over the *query's* minutiae, a partial query is
judged only on what it contains - this is what lets it degrade gracefully when the print shrinks.

**Honest limits.**
- SOCOFing queries are edited copies of the enrolled print. Full protocol on the 897-finger test gallery (windows
  centred on a minutia): Rank-1 88 / 85 / 75 / 61 / 45 % at 100 / 75 / 50 / 25 / 10 % of the print (classical:
  90 / 87 / 69 / 31 / 6 %). Windows placed uniformly at random (the sliders here) are harsher: expect less.
- On real FVC impressions the fine-tuned model reaches about 88 % Rank-1 against 50 enrolled fingers per database
  (44 % with 10 % of the print). The detector was trained on part of FVC, and the matcher fine-tuned on other
  fingers of the same sensors, so this does not show generalisation to new sensors.
- Uploaded images only work if they look like the training data (SOCOFing-style prints, or the sensor of the
  selected FVC database). Phone photos or latent prints are not supported.
- The partial prints are square windows over the extracted minutiae; real latent prints are irregular, smudged
  and rarely this clean. The "enrolled?" verdict is calibrated on these same benchmark conditions.
""")
