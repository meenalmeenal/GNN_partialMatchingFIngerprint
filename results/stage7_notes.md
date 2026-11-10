# Stage 7 — interactive partial-print matching demo

Goal: a working interface where a partial fingerprint goes in and the closest enrolled prints come
out, to be shown as a demo and then improved. Launch: `streamlit run app/app.py` (repo root).

## What it does
- **Gallery**: SOCOFing test split (900 held-out fingers, gallery = clean `Real` print) or one FVC2002/2004
  database (held-out fingers 51-100, gallery = impression 1).
- **Query**: a sample probe from the dataset (SOCOFing altered print / FVC impression 2-8), or an uploaded
  image (SOCOFing recipe: resize 160x160; FVC: the per-database ridge-spacing factor of Stage 8, i.e. it
  assumes the same sensor). Optional "true identity" to see the rank of the right finger.
- **Partial print**: area slider (10-100 %) and window-position sliders; the query is cut to a square window
  over the print's own minutiae extent (the family of crops used in Stages 8/9). Templates stay complete.
- **Search**: whole gallery scored live by the learned matcher (Stage 9 relative-geometry GNN + MaxSim; the
  Stage 8 FVC-fine-tuned version for FVC galleries) and, side by side, the classical Hough matcher.
  Roughly 0.3 s for 900 prints (0.03 s for 50).
- **Explanation**: minutiae overlay with the kept window; top-K cards with score, z-score and a marker
  on the true match; per-match view with the query-minutia -> best-template-minutia links behind the MaxSim
  score; score histogram over the gallery.
- **Honesty built in**: sidebar and an expander state the Stage 8 limits (real prints, extraction).

Code: `app/app.py` (UI), `src/demo_core.py` (datasets, gallery, search, cropping, drawing),
`src/stage7_check.py` (numbers below), `src/stage7_figures.py` (`results/stage7/examples.png`).

## Checks
- Extraction from an image reproduces the stored graph (SOCOFing 33/33 minutiae; FVC 29/29), so uploads
  go through the same path as the data the models were evaluated on.
- Streamlit AppTest run over SOCOFing/FVC galleries, 10/25/50/100 % area and both models: no exceptions.
  Rendering checked in the browser pane (`results/stage7/app_screenshot_*.jpg`). The upload widget itself was
  not exercised in a browser (only the extraction behind it).

## Numbers for exactly what the demo does (`results/stage7/demo_numbers.*`)
Rank-1 %, one random probe per query finger, random window position, whole gallery searched:

| setting | 100 % | 50 % | 25 % |
|---|---|---|---|
| SOCOFing, 900-print gallery (300 query fingers), learned | 71.7 | 15.7 | 10.7 |
| SOCOFing, 900-print gallery, classical | 81.0 | 15.3 | 4.0 |
| FVC, 6 real-sensor DBs, 50-print galleries (50 queries each), fine-tuned learned | 22.7 | 21.3 | 11.3 |
| FVC, same, classical | 18.0 | 14.3 | 7.0 |

Standard error is about 2.6 points on SOCOFing and about 1.5 on the 6-database FVC means (per-database
rows in the json are much noisier). Examples (`results/stage7/examples.png`) use a fixed seed, not picked for
success: SOCOFing 50 % ranks the true finger #521/900, SOCOFing 25 % #1, FVC2002 DB2 100 % #14/50, 50 % #2/50.

## Reading it
1. A learned-vs-classical gap exists only at small areas: at 25 % learned 10.7 vs classical 4.0 on SOCOFing and
   11.3 vs 7.0 on FVC; at 50 % SOCOFing they tie; on full SOCOFing prints classical is better (81 vs 72).
   This matches the Stage 9 direction (learned overtakes classical on small areas) but with a much smaller margin.
2. **These SOCOFing partial numbers are far below the Stage 9 sweep** (e.g. 15.7 % vs ~50 % at 50 % area). Two
   differences are visible in the code, effects not separated: the gallery is 900 prints here vs 200, and the
   demo window is placed uniformly at random over the minutiae extent, so it can land on a sparse region,
   whereas `partiality_sweep.crop_graph` centres each window on a randomly chosen minutia (more minutiae per
   probe). Do not quote the demo numbers as the Stage 9 result; use the demo numbers only to describe the
   demo's behaviour. Re-running the sweep at 900 fingers would settle how much is gallery size.
3. On real FVC prints the system is weak at every area (best 22.7 %, 50-print gallery, chance 2 %), as in Stage 8.
4. Partial query with few minutiae is a guess; the UI warns below 6 minutiae.

## Limits
- Models are trained on SOCOFing (one fine-tuned on FVC); uploads that do not look like SOCOFing or the
  selected FVC sensor (phone photos, latents) are not handled.
- Partial prints are square windows over extracted minutiae, not irregular latents.
- Gallery is a fixed benchmark split, not an enrolment interface.

## Next
Improve what the demo shows: a robust minutiae extractor (largest lever, Stage 8), re-running the partial sweep
at the 900-finger gallery size and with minutia-centred windows, and optionally enrolling user-supplied
galleries in the app.
