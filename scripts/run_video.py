"""Measure one video: field geometry + the HUD cross-check.

    python scripts/run_video.py 1          # Clip_1
    python scripts/run_video.py 1 --prep   # also redo stages 0-2
"""
import argparse, json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from blitzfield import pipeline
from blitzfield.config import get_video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="processing order 1-7, or slug")
    ap.add_argument("--prep", action="store_true", help="run stages 0-2 first")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    spec = get_video(a.video)
    print(f"[{spec.order}] {spec.filename}")
    if a.prep:
        print("  stages 0-2:", json.dumps(pipeline.prepare(spec), indent=2))
    s = pipeline.analyse(spec, stride=a.stride, limit=a.limit)
    print("  measurement:", json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
