"""
Reorganize the raw SOCOFing dataset into a  <identity>/<sample>  image tree that
graph_build.py expects.

Raw SOCOFing layout:
    SOCOFing/Real/1__M_Left_index_finger.BMP
    SOCOFing/Altered/Altered-Easy/1__M_Left_index_finger_CR.BMP
    SOCOFing/Altered/Altered-Medium/...
    SOCOFing/Altered/Altered-Hard/...

Filename grammar:
    {id}__{gender}_{hand}_{finger}_finger[_{alteration}].BMP

Identity (same finger of same person) is  {id}_{hand}_{finger}, e.g. "1_Left_index".
Each identity folder collects the Real print plus its altered variants as samples,
so every identity has >=2 samples (needed for positive pairs).

Usage:
    python src/prepare_socofing.py --input data/SOCOFing --output data/SOCOFing_organized
    # then:
    python src/graph_build.py --input data/SOCOFing_organized --output data/graphs
"""
import argparse
import glob
import os
import re
import shutil

FNAME_RE = re.compile(
    r"^(?P<id>\d+)__(?P<gender>[MF])_(?P<hand>Left|Right)_(?P<finger>\w+?)_finger"
    r"(?:_(?P<alt>[A-Za-z]+))?\.(?P<ext>[A-Za-z]+)$"
)


def parse(fname: str):
    m = FNAME_RE.match(fname)
    if not m:
        return None
    d = m.groupdict()
    identity = f"{d['id']}_{d['hand']}_{d['finger']}"
    return identity, d["alt"] or "real", d["ext"]


def organize(input_dir: str, output_dir: str, link: bool = False):
    paths = glob.glob(os.path.join(input_dir, "**", "*.*"), recursive=True)
    paths = [p for p in paths if p.lower().endswith((".bmp", ".png", ".jpg", ".tif"))]
    if not paths:
        raise SystemExit(f"No fingerprint images found under {input_dir}")

    counts, skipped = {}, 0
    for src in paths:
        parsed = parse(os.path.basename(src))
        if parsed is None:
            skipped += 1
            continue
        identity, tag, ext = parsed
        # bucket = Real / Altered-Easy / ... -> keeps variant filenames unique
        bucket = os.path.basename(os.path.dirname(src))
        dst_dir = os.path.join(output_dir, identity)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, f"{bucket}_{tag}.{ext}")
        if os.path.exists(dst):
            dst = os.path.join(dst_dir, f"{bucket}_{tag}_{counts.get(identity, 0)}.{ext}")
        if link:
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        else:
            shutil.copy2(src, dst)
        counts[identity] = counts.get(identity, 0) + 1

    multi = sum(1 for c in counts.values() if c >= 2)
    print(f"Organized {sum(counts.values())} images into {len(counts)} identities "
          f"({multi} with >=2 samples). Skipped {skipped} unparseable filenames.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Raw SOCOFing root directory")
    ap.add_argument("--output", required=True, help="Destination organized image tree")
    ap.add_argument("--link", action="store_true", help="Hard-link instead of copy (saves disk)")
    args = ap.parse_args()
    organize(args.input, args.output, link=args.link)
