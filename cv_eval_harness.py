"""CV proctoring evaluation harness — precision/recall of cv_service (review score).

Feeds labeled webcam CLIPS through the teammate's cv_service (/session/start ->
/frame per frame -> /session/end), thresholds the peak review_score into a binary
cheating decision, and reports precision/recall — the SAME framework as the VLM
eval, so the two modalities are comparable.

The CV output is an "experimental review indicator, NOT a calibrated cheating
probability" (see cv_service.py). We only threshold it here to MEASURE the
detector; keep that framing in the paper.

Clips folder: video files named  <C|N><n>_description.mp4
  C1_phone.mp4     -> cheating       (C = cheating)
  N1_normal.mp4    -> not_cheating   (N = not)
Tip: start each clip with ~2-3 s of the person sitting normally so cv_service can
auto-calibrate its neutral head baseline before the behavior begins.

Setup:  pip install requests opencv-python
Run:
  # cv_service local:
  python cv_eval_harness.py --clips ./cv_clips
  # cv_service on Kaggle (tunnel):
  python cv_eval_harness.py --clips ./cv_clips --cv-url https://xxxx.trycloudflare.com --threshold 0.5
"""
import argparse
import base64
import glob
import os
import re
import time

import cv2
import requests

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")
FRAME_SIZE = (320, 240)   # match the extension's capture resolution


def label_from_name(fname):
    m = re.match(r"([CcNn])\d", os.path.basename(fname))
    if not m:
        return None
    return "cheating" if m.group(1).upper() == "C" else "not_cheating"


def _stream(cv_url, sid, path, stride, collect, debug, st, tag=""):
    """POST a video's frames to /frame. collect=True updates peak/diagnostics."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    est = int((cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) // max(stride, 1))
    idx = 0; local = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                h, w = frame.shape[:2]
                if w > 960:
                    frame = cv2.resize(frame, (960, int(h * 960 / w)))
                ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok2:
                    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
                    d = None
                    for attempt in range(3):   # tolerate transient tunnel resets
                        try:
                            r = requests.post(cv_url + "/frame",
                                              json={"studentId": sid, "image": b64, "timestamp": st["t"]},
                                              timeout=120)
                            d = r.json(); break
                        except Exception as e:
                            if attempt == 2:
                                print("   frame err (gave up after 3):", repr(e)[:120])
                            else:
                                time.sleep(1.0)
                    if d is not None:
                        st["sent"] += 1; local += 1
                        st["t"] += stride / fps
                        phase = d.get("calibration_phase"); st["last_phase"] = phase
                        if phase is not None: st["phases"].add(phase)
                        cues = d.get("active_cues", []) or []; st["cues"].update(cues)
                        dets = d.get("detections", []) or []; st["max_dets"] = max(st["max_dets"], len(dets))
                        sc = float(d.get("review_score", 0.0))
                        if collect and sc > st["peak"]:
                            st["peak"] = sc; st["level"] = d.get("review_level", st["level"])
                        if debug and collect:
                            names = [x.get("label") or x.get("class") or x.get("name") for x in dets] if dets else []
                            print(f"    f{st['sent']:03d} phase={phase} score={sc:.3f} cues={cues} dets={names}")
                        elif local % 3 == 0:
                            print(f"\r    {tag}: {local}/{est} frames  phase={phase}  peak={st['peak']:.2f}   ",
                                  end="", flush=True)
            idx += 1
    finally:
        cap.release()
        if not debug and local:
            print()  # newline after the progress line


def run_clip(cv_url, path, stride, debug=False, calib=None):
    """Optionally feed a calibration clip first (to finish gaze calibration), then
    score the scenario clip. Peak is taken over the SCENARIO frames only."""
    sid = os.path.splitext(os.path.basename(path))[0]
    requests.post(cv_url + "/session/start", json={"studentId": sid}, timeout=60)
    st = {"t": 0.0, "sent": 0, "peak": 0.0, "level": "NONE",
          "last_phase": None, "phases": set(), "cues": set(), "max_dets": 0}
    try:
        if calib:
            _stream(cv_url, sid, calib, stride, collect=False, debug=False, st=st, tag="calib")
            if st["last_phase"] != "READY":
                print(f"   WARNING: calibration NOT ready after calib clip "
                      f"(phase={st['last_phase']}) -> scores will be 0. Use a longer, "
                      f"dead-still, straight-at-camera calib clip.")
            else:
                print("   calibration READY after calib clip")
            st["max_dets"] = 0; st["cues"] = set()   # reset diagnostics to scenario-only
        _stream(cv_url, sid, path, stride, collect=True, debug=debug, st=st, tag=sid)
    finally:
        requests.post(cv_url + "/session/end", json={"studentId": sid}, timeout=30)
    return {"peak": st["peak"], "level": st["level"], "nframes": st["sent"], "sent": st["sent"],
            "last_phase": st["last_phase"], "phases": sorted(st["phases"]),
            "cues_seen": sorted(st["cues"]), "max_dets": st["max_dets"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True, help="folder of labeled video clips")
    ap.add_argument("--cv-url", default=os.environ.get("CV_URL", "http://localhost:8789"))
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="flag cheating if peak review_score >= this (0.25=MODERATE, 0.5=HIGH, 0.75=VERY_HIGH)")
    ap.add_argument("--stride", type=int, default=5, help="process every Nth frame")
    ap.add_argument("--debug", action="store_true", help="print per-frame phase/score/cues/detections")
    ap.add_argument("--calib", default=None,
                    help="a ~10s 'sit still, look at camera' clip fed first each session to finish gaze calibration")
    args = ap.parse_args()

    if args.calib and not os.path.isfile(args.calib):
        print("ERROR: --calib file not found:", args.calib); return

    # health check
    try:
        h = requests.get(args.cv_url + "/health", timeout=30).json()
        print("cv_service:", h)
    except Exception as e:
        print("cannot reach cv_service at", args.cv_url, "->", e)
        return

    clips = sorted(f for f in glob.glob(os.path.join(args.clips, "*"))
                   if f.lower().endswith(VIDEO_EXTS))
    rows = []
    for path in clips:
        gt = label_from_name(path)
        if gt is None:
            print("skip (bad name, need C#/N# prefix):", os.path.basename(path)); continue
        t0 = time.time()
        if args.debug: print("DEBUG", os.path.basename(path))
        info = run_clip(args.cv_url, path, args.stride, debug=args.debug, calib=args.calib)
        peak, level, nframes = info["peak"], info["level"], info["nframes"]
        pred = "cheating" if peak >= args.threshold else "not_cheating"
        print(f"  {os.path.basename(path):26s} gt={gt:12s} peak={peak:.3f} -> {pred}"
              f"  | phase={info['last_phase']} phases={info['phases']} "
              f"cues={info['cues_seen']} maxdet={info['max_dets']} sent={info['sent']}")
        rows.append({"clip": os.path.basename(path), "gt": gt, "pred": pred,
                     "peak_score": round(peak, 3), "review_level": level,
                     "frames": nframes, "sec": round(time.time() - t0, 1)})

    # metrics
    tp = sum(r["gt"] == "cheating" and r["pred"] == "cheating" for r in rows)
    fp = sum(r["gt"] == "not_cheating" and r["pred"] == "cheating" for r in rows)
    fn = sum(r["gt"] == "cheating" and r["pred"] == "not_cheating" for r in rows)
    tn = sum(r["gt"] == "not_cheating" and r["pred"] == "not_cheating" for r in rows)
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0

    print("\n=== CV cheating-decision (threshold =", args.threshold, ") ===")
    print(f"n={len(rows)}  TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"Precision={P:.3f}  Recall={R:.3f}  F1={F:.3f}")

    # save
    try:
        import csv
        with open("cv_eval_results.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print("per-clip -> cv_eval_results.csv")
    except Exception:
        pass


if __name__ == "__main__":
    main()
