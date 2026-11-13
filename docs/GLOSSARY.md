# Glossary

**Fingerprint basics**
- **Ridge / valley**: the raised lines and the grooves between them in a fingerprint.
- **Minutia (plural minutiae)**: a point where a ridge ends (*ending*) or splits in two (*bifurcation*). A print has 20-100 of them.
  We describe each by position (x, y), direction and type.
- **Partial print**: only a part of the finger is available (for example at a crime scene). We simulate it by cutting out a square window
  covering 100 / 75 / 50 / 25 / 10 % of the print and keeping only the minutiae inside.
- **Impression**: one capture of a finger. Two impressions of the same finger are never identical (pressure, skin stretch, noise).

**Matching**
- **Gallery**: the enrolled, known prints (one full print per finger). **Probe / query**: the print we want to identify. We rank the gallery
  by similarity to the probe.
- **Identification (1:N)**: "who is this?" — rank all gallery prints. **Verification (1:1)**: "is this finger X?" — accept or reject one comparison.
- **Closed-set** identification: the true finger is guaranteed to be in the gallery. **Open-set**: it may not be, so the system must also be able to say "unknown".
- **Rank-1 / Rank-5**: share of probes whose true finger is ranked first / within the first five. Chance level = 1 / gallery size.
- **EER (equal error rate)**: the error rate at the threshold where false accepts equal false rejects (lower is better).
- **FAR (false accept rate)**: share of *unknown* prints wrongly accepted. **DIR @ FAR x %**: share of enrolled probes that are both identified correctly
  *and* accepted, at the threshold that lets only x % of unknown prints through (higher is better).
- **z-score of a match score**: how many standard deviations the top score is above the average score against the whole gallery. A confident
  identification stands out; this is what the demo's "enrolled / uncertain / not enrolled" verdict uses.

**Our methods**
- **Minutiae graph**: minutiae are the nodes; each is linked to its 5 nearest neighbours. Node features: x, y, sin(direction), cos(direction), type.
- **GNN (graph neural network)**: a network that passes messages between linked nodes. Our final one ("invariant relative-geometry GNN") only uses
  distances and angles *between* neighbouring minutiae, so rotating or shifting a print does not change its output.
- **MaxSim score**: each query minutia is matched to its most similar minutia in the gallery print; the score is the average of these best
  similarities. It judges a partial query only on what the query contains.
- **InfoNCE (contrastive loss)**: training objective that pulls two prints of the same finger together and pushes other fingers in the batch away.
- **Classical (Hough) matcher**: hand-designed baseline — vote for the rotation/shift that best aligns the two minutiae sets, then count the minutiae that coincide.
- **LEADER**: a pretrained neural network (package `pyfing`) that finds minutiae in an image. It replaced our hand-built extractor in Stage 9d.
- **Fine-tuning**: continuing to train a model on a small amount of data from the target domain (here: 40 FVC fingers per database).

**Data**
- **SOCOFing**: 6,000 fingers; each has one real print and synthetic damaged copies (*CR* central rotation, *Obl* obliteration, *Zcut* z-cut; each *Easy / Medium / Hard*).
  The damaged prints are edits of the *same* impression, so SOCOFing is an optimistic benchmark.
- **FVC2002 / FVC2004**: standard fingerprint competition databases; 8 databases x 100 fingers x 8 independent real impressions
  (databases 1-3 real sensors, database 4 synthetic). Fingers 1-100 = *set A* (evaluation), 101-110 = *set B* (allowed for tuning).
- **Held-out fingers**: fingers never used for training or model selection. All reported test numbers use them.
