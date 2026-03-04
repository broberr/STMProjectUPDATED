# src/run_experiment.py
import os
import csv
import random
import re
import math
import argparse
from typing import Dict, List, Tuple, Optional, Any

from src.models.llava_captioner import LlavaCaptioner
from src.models.qwen2vl_captioner import Qwen2VLCaptioner
from src.models.smolvlm_captioner import SmolVLMCaptioner
from src.models.minicpm_captioner import MiniCPMCaptioner
from src.models.internvl_captioner import InternVLCaptioner

from sklearn.metrics import f1_score

from src.config import Config
from src.frames import extract_frames_with_timestamps
from src.memory import ShortTermMemory
from src.utils_gpu import free_vram
from src.llm_reason import ActivityClassifierLLM

# If you add InternVL later, import it here:
try:
    from src.models.internvl_captioner import InternVLCaptioner
    HAS_INTERNVL = True
except Exception:
    HAS_INTERNVL = False


ALLOWED_LABELS = ["meeting", "phone_use", "eating_drinking", "video_call", "using_computer"]


# -------------------------
# Helpers
# -------------------------
def norm_label(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    return x.strip().lower().replace(" ", "_").replace("-", "_")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def clean_caption(cap: str) -> str:
    if not cap:
        return ""
    # Remove common chat markers
    cap = cap.replace("ASSISTANT:", "").replace("USER:", "").strip()

    # If the prompt got echoed, cut everything before the last blank line.
    # This works because your prompt is multi-line and the model output usually comes after a blank line.
    if "\n\n" in cap:
        parts = [p.strip() for p in cap.split("\n\n") if p.strip()]
        # keep only the LAST chunk (most likely the real caption)
        cap = parts[-1]

    return cap.strip()

def load_dataset(videos_dir: str, labels_csv: str) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    with open(labels_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row["video"].strip()
            lab = norm_label(row["label"])
            if not lab or lab not in ALLOWED_LABELS:
                continue
            path = os.path.join(videos_dir, vid)
            if os.path.exists(path):
                items.append((path, lab))
    return items


def build_observations(per_frame: List[Tuple[float, str]], mode: str, mem_cfg: dict) -> str:
    """
    mode:
      - "none": raw timestamped captions
      - "stm": ShortTermMemory window aggregation
    """
    if mode == "none":
        return "\n".join([f"[{t:.1f}s] {cap}" for (t, cap) in per_frame])

    stm = ShortTermMemory(
        window_seconds=float(mem_cfg["window_seconds"]),
        compress=bool(mem_cfg["compress"]),
        summary_max_chars=int(mem_cfg["summary_max_chars"]),
    )
    for t, cap in per_frame:
        stm.add(t, cap)
    return stm.as_text()



def score_labels_from_text(text: str) -> Dict[str, int]:
    t = text.lower()

    patterns = {
        "video_call": [
            "video call","facetime", "call controls", "webcam", "microphone", "headset",
            "speaking", "talking", "gesturing", "raised hand", "waving",
            "engaged in a conversation",
        ],
        "phone_use": [
            "phone", "smartphone", "holding cell phone", "cell phone",
            "scrolling", "texting", "typing on phone", "holding phone",
            "calling", "answering calls", "answering", "pressing on screen",
        ],
        "eating_drinking": [
            "drinking", "sip", "sipping", "cup", "mug", "coffee", "tea", "eating", "bite",
        ],
        "meeting": [
            "meeting", "in-person", "with another person", "talking to someone", "two people",
        ],
        "using_computer": [
            "laptop", "computer", "typing", "touchpad", "keyboard",
            "attention towards computer", "looking at the laptop",
            "working on", "browsing", "reading", "writing",
            "using a laptop", "using the laptop"
        ],
    }


    strong_video_call = ["zoom", "teams", "google meet", "facetime", "call controls"]
    strong_phone_use = [
        "phone", "smartphone", "holding phone", "holding a phone", "cell phone",
        "scrolling", "texting", "typing on phone", "calling", "answering",
    ]

    counts = {k: 0 for k in patterns.keys()}

    for label, cues in patterns.items():
        for cue in cues:
            counts[label] += t.count(cue)


    return {lab: int(counts.get(lab, 0)) for lab in ALLOWED_LABELS}


def pick_closest_label(scores: Dict[str, int]) -> str:
    best_label, best_score = max(scores.items(), key=lambda kv: kv[1])
    if best_score == 0:
        return "using_computer"

    specificity_order = ["video_call", "phone_use", "eating_drinking", "meeting", "using_computer"]
    tied = [k for k, v in scores.items() if v == best_score]
    for lab in specificity_order:
        if lab in tied:
            return lab
    return best_label



def predict_main_label_llm(classifier: ActivityClassifierLLM, observations_text: str, debug: bool = False) -> Tuple[Optional[str], str]:
    prompt = f"""
You are a strict classifier.

Choose ONE label from this exact list:
{ALLOWED_LABELS}

IMPORTANT:
If you are not confident, output:
label=uncertain

Rules:
- Output EXACTLY one line: label=<one_label_or_uncertain>
- Use underscores, not spaces.
- Do NOT output explanations.

OBSERVATIONS:
{observations_text}
""".strip()

    out = classifier.freeform(prompt)

    if debug:
        print("\n=== RAW CLASSIFIER OUTPUT ===\n")
        print(out)
        print("\n=== END RAW OUTPUT ===\n")


    m = re.search(r"(?im)^\s*label\s*=\s*([a-z_]+)\s*$", out)
    if not m:
        m2 = re.search(r"label\s*=\s*([a-z_]+)", out.lower())
        if not m2:
            return None, out
        label = norm_label(m2.group(1))
    else:
        label = norm_label(m.group(1))

    if label in ALLOWED_LABELS or label == "uncertain":
        return label, out
    return None, out


def decide_final_label(llm_pred: Optional[str], scores: Dict[str, int], closest: str, margin: int = 3) -> str:
    if llm_pred is None or llm_pred == "uncertain":
        return closest
    if llm_pred not in scores:
        return closest
    if scores[closest] - scores[llm_pred] >= margin:
        return closest
    return llm_pred

def print_evidence(scores: Dict[str, int], closest: str):
    print("=== EVIDENCE SCORES ===\n")
    print(f"meeting         = {{{scores.get('meeting', 0)}}}")
    print(f"phone_use       = {{{scores.get('phone_use', 0)}}}")
    print(f"eating_drinking = {{{scores.get('eating_drinking', 0)}}}")
    print(f"video_call      = {{{scores.get('video_call', 0)}}}")
    print(f"using_computer  = {{{scores.get('using_computer', 0)}}}\n")
    print(f"closest={closest}\n")
# -------------------------
# McNemar exact
# -------------------------
def binom_cdf(k: int, n: int, p: float = 0.5) -> float:
    if n <= 0:
        return 1.0
    k = max(0, min(k, n))
    s = 0.0
    for i in range(0, k + 1):
        s += math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
    return s
def mcnemar_exact_p(n01: int, n10: int) -> float:
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    p_one = binom_cdf(k, n, 0.5)
    return min(1.0, 2.0 * p_one)


# -------------------------
# Plotting
# -------------------------
def try_plot(out_png: str, fn):
    try:
        fn()
    except Exception as e:
        print(f"[WARN] Skipping plot {out_png}: {e}")


def plot_mode_macro(out_png: str, summary_rows: List[Dict[str, Any]]):
    import matplotlib.pyplot as plt
    modes = [r["mode"] for r in summary_rows]
    vals = [float(r["macro_f1"]) for r in summary_rows]
    plt.figure()
    plt.bar(modes, vals)
    plt.ylim(0, 1.05)
    plt.ylabel("Macro F1")
    plt.title("STM vs NONE — Overall Macro F1")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


# -------------------------
# VLMs
# -------------------------
def make_vlm(model_type: str, model_id: str, device: str, dtype: str, load_in_4bit: bool, max_new_tokens: int):
    model_type = model_type.lower().strip()
    if model_type == "llava":
        return LlavaCaptioner(model_id=model_id, device=device, dtype=dtype, load_in_4bit=load_in_4bit, max_new_tokens=max_new_tokens)
    if model_type == "qwen2vl":
        return Qwen2VLCaptioner(model_id=model_id, device=device, dtype=dtype, load_in_4bit=load_in_4bit, max_new_tokens=max_new_tokens)
    if model_type == "smolvlm":
        return SmolVLMCaptioner(
            model_id=model_id,
            device=device,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            max_new_tokens=max_new_tokens
        )
    if model_type == "minicpm":
        return MiniCPMCaptioner(model_id=model_id, device=device, dtype=dtype, load_in_4bit=load_in_4bit, max_new_tokens=max_new_tokens)
    if model_type == "internvl":
        if not HAS_INTERNVL:
            raise RuntimeError("InternVL adapter not found. Add src/models/internvl_captioner.py first.")
        return InternVLCaptioner(model_id=model_id, device=device, dtype=dtype, load_in_4bit=load_in_4bit, max_new_tokens=max_new_tokens)

    raise ValueError(f"Unknown VLM type: {model_type}")


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=["stm", "none"], choices=["stm", "none"])
    parser.add_argument("--video", default=None, help="Optional: run only one video filename (e.g. phone1.mp4)")
    parser.add_argument("--models", nargs="*", default=None, help="Optional: run only these VLM model names (e.g. llava15_7b qwen2vl_2b)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--margin", type=int, default=3)
    args = parser.parse_args()

    cfg = Config.load("config.yaml")
    data_cfg = cfg["data"]
    vlm_cfg = cfg["vlm"]
    llm_cfg = cfg["llm"]
    mem_cfg = cfg["memory"]
    exp_cfg = cfg["experiment"]

    seed = args.seed if args.seed is not None else int(cfg.raw.get("seed", 42))
    random.seed(seed)

    outputs_dir = exp_cfg.get("output_dir", "outputs")
    ensure_dir(outputs_dir)

    items = load_dataset(data_cfg["videos_dir"], data_cfg["labels_csv"])
    if not items:
        print("No dataset items found.")
        return

    if args.video:
        target = args.video.strip().lower()
        items = [(p, y) for (p, y) in items if os.path.basename(p).lower() == target]
        if not items:
            print(f"No matching video found in dataset for: {args.video}")
            return


    vlm_models = vlm_cfg.get("models", [])
    if args.models:
        wanted = {m.strip().lower() for m in args.models}
        vlm_models = [m for m in vlm_models if m.get("name", "").lower() in wanted]
        if not vlm_models:
            print("No VLM models matched --models.")
            return

    modes = args.modes


    combined_summary_csv = os.path.join(outputs_dir, "multi_vlm_ablation_summary.csv")
    combined_per_video_csv = os.path.join(outputs_dir, "multi_vlm_ablation_results.csv")

    combined_summary_rows: List[Dict[str, Any]] = []
    combined_per_video_rows: List[Dict[str, Any]] = []


    classifier = ActivityClassifierLLM(
        model_id=llm_cfg["model_id"],
        device=cfg["device"],
        dtype=cfg["dtype"],
        load_in_4bit=llm_cfg["load_in_4bit"],
        max_new_tokens=int(llm_cfg["max_new_tokens"]),
        temperature=float(llm_cfg["temperature"]),
        system_prompt=llm_cfg.get("system_prompt", ""),
    )

    for model_entry in vlm_models:
        model_name = model_entry["name"]
        model_type = model_entry["type"]
        model_id = model_entry["model_id"]

        print("\n====================================================")
        print(f"VLM MODEL: {model_name} ({model_type})")
        print("====================================================\n")

        model_out_dir = os.path.join(outputs_dir, model_name)
        ensure_dir(model_out_dir)


        paired: Dict[str, Dict[str, Dict[str, Any]]] = {}

        per_video_rows: List[Dict[str, Any]] = []
        summary_rows: List[Dict[str, Any]] = []

        for mode in modes:
            y_true: List[str] = []
            y_pred: List[str] = []
            per_video_f1s: List[float] = []

            print(f"\n------------------ MODE: {mode.upper()} ------------------\n")

            for video_path, true_label in items:
                vid_name = os.path.basename(video_path)
                print(f"\n===== VIDEO: {vid_name} =====\n")


                try:
                    captioner = make_vlm(
                        model_type=model_type,
                        model_id=model_id,
                        device=cfg["device"],
                        dtype=cfg["dtype"],
                        load_in_4bit=bool(vlm_cfg["load_in_4bit"]),
                        max_new_tokens=int(vlm_cfg["max_new_tokens"]),
                    )
                except Exception as e:
                    print(f"[WARN] Skipping VLM {model_name}: {e}")
                    break

                frames = extract_frames_with_timestamps(
                    video_path,
                    stride_seconds=float(data_cfg["frame_stride_seconds"]),
                    max_frames=int(data_cfg["max_frames_per_video"]),
                )
                frames.sort(key=lambda x: x.timestamp_s)

                per_frame: List[Tuple[float, str]] = []
                for fr in frames:
                    t = float(fr.timestamp_s)
                    vlm_prompt = vlm_cfg.get("prompt", "").strip()
                    if not vlm_prompt:
                        vlm_prompt = "Describe what the person is doing in this frame in 1–2 sentences."
                    cap = captioner.caption(fr.bgr, vlm_prompt)
                    cap = clean_caption(cap)
                    per_frame.append((t, cap))


                if args.debug:
                    print("\n=== PER-FRAME CAPTIONS ===\n")
                    for t, cap in per_frame:
                        print(f"[{t:06.2f}s] {cap}")


                observations = build_observations(per_frame, mode, mem_cfg)


                if args.debug:
                    print("\n=== OBSERVATIONS ===\n")
                    print(observations[:4000], "...\n" if len(observations) > 4000 else "\n")
                scores = score_labels_from_text(observations)
                closest = pick_closest_label(scores)

                llm_pred, raw_out = predict_main_label_llm(classifier, observations, debug=args.debug)
                pred_label = decide_final_label(llm_pred, scores, closest, margin=args.margin)

                if args.debug:
                    print_evidence(scores, closest)
                    print(f"llm_pred={llm_pred}")
                    print(f"final_pred={pred_label}\n")

                pv_f1 = float(f1_score([true_label], [pred_label], average="macro"))
                correct = 1 if pred_label == true_label else 0

                per_video_rows.append({
                    "vlm_model": model_name,
                    "mode": mode,
                    "video": vid_name,
                    "true": true_label,
                    "pred": pred_label,
                    "correct": correct,
                    "per_video_f1": pv_f1,
                    "closest": closest,
                    "score_meeting": scores.get("meeting", 0),
                    "score_phone_use": scores.get("phone_use", 0),
                    "score_eating_drinking": scores.get("eating_drinking", 0),
                    "score_video_call": scores.get("video_call", 0),
                    "score_using_computer": scores.get("using_computer", 0),
                    "llm_pred_raw": (llm_pred if llm_pred else ""),
                    "raw_classifier_output": raw_out.replace("\n", "\\n")[:2000],
                })

                paired.setdefault(vid_name, {})
                paired[vid_name][mode] = {"true": true_label, "pred": pred_label, "correct": correct}

                y_true.append(true_label)
                y_pred.append(pred_label)
                per_video_f1s.append(pv_f1)

                del captioner
                free_vram()

            if not y_true:
                continue

            macro = float(f1_score(y_true, y_pred, average="macro"))
            mean_pv = sum(per_video_f1s) / len(per_video_f1s)
            var_pv = sum((x - mean_pv) ** 2 for x in per_video_f1s) / len(per_video_f1s)
            std_pv = var_pv ** 0.5

            summary_rows.append({
                "vlm_model": model_name,
                "mode": mode,
                "n": len(y_true),
                "macro_f1": macro,
                "mean_per_video_f1": mean_pv,
                "std_per_video_f1": std_pv,
            })


        stats_lines: List[str] = []
        if "stm" in modes and "none" in modes:
            n01 = 0
            n10 = 0
            for vid, d in paired.items():
                if "stm" not in d or "none" not in d:
                    continue
                stm_c = int(d["stm"]["correct"])
                non_c = int(d["none"]["correct"])
                if stm_c == 0 and non_c == 1:
                    n01 += 1
                elif stm_c == 1 and non_c == 0:
                    n10 += 1
            p = mcnemar_exact_p(n01, n10)
            stats_lines.append(f"McNemar exact test (STM vs NONE): n01={n01}, n10={n10}, p={p:.6f}")


        per_video_csv = os.path.join(model_out_dir, "ablation_results.csv")
        summary_csv = os.path.join(model_out_dir, "ablation_summary.csv")
        stats_txt = os.path.join(model_out_dir, "ablation_stats.txt")

        if per_video_rows:
            with open(per_video_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(per_video_rows[0].keys()))
                w.writeheader()
                w.writerows(per_video_rows)

        if summary_rows:
            with open(summary_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
                w.writeheader()
                w.writerows(summary_rows)

        with open(stats_txt, "w", encoding="utf-8") as f:
            for line in stats_lines:
                f.write(line + "\n")


        if summary_rows:
            out_png = os.path.join(model_out_dir, "macro_f1_by_mode.png")
            try_plot(out_png, lambda: plot_mode_macro(out_png, summary_rows))


        combined_summary_rows.extend(summary_rows)
        combined_per_video_rows.extend(per_video_rows)

        print(f"\nSaved per-model outputs to: {model_out_dir}")


    if combined_per_video_rows:
        with open(combined_per_video_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(combined_per_video_rows[0].keys()))
            w.writeheader()
            w.writerows(combined_per_video_rows)

    if combined_summary_rows:
        with open(combined_summary_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(combined_summary_rows[0].keys()))
            w.writeheader()
            w.writerows(combined_summary_rows)

    print(f"\nSaved combined results: {combined_per_video_csv}")
    print(f"Saved combined summary: {combined_summary_csv}")


    del classifier
    free_vram()


if __name__ == "__main__":
    main()