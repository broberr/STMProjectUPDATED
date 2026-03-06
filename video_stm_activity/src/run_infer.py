# src/run_infer.py
import os
import csv
import re
import argparse
from typing import Optional, Dict, List, Tuple

from sklearn.metrics import f1_score

from src.config import Config
from src.frames import extract_frames_with_timestamps
from src.memory import ShortTermMemory
from src.utils_gpu import free_vram
from src.llm_reason import ActivityClassifierLLM

from src.models.llava_captioner import LlavaCaptioner
from src.models.qwen2vl_captioner import Qwen2VLCaptioner
from src.models.smolvlm_captioner import SmolVLMCaptioner
from src.models.minicpm_captioner import MiniCPMCaptioner


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


def load_video_labels(csv_path: str) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["video"].strip()] = norm_label(row["label"])
    return labels


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
            "video call", "on a call", "in a call", "zoom", "teams", "google meet", "meet",
            "facetime", "call controls", "webcam", "microphone", "headset",
            "speaking", "talking", "gesturing", "raised hand", "waving",
            "looking at computer while gesturing", "laptop use + gesture + talking", "desk + gesture + talking",
            "unmuting", "muting microphone", "turning camera on", "turning camera off",
            "sharing screen", "screen share", "viewing presentation", "watching slides",
            "waiting in waiting room", "joining call", "leaving meeting", "ending call",
            "looking at camera", "looking at screen", "looking away from camera",
            "sitting at desk on call", "standing during video call", "walking with laptop on call",
            "virtual background", "blurred background", "sitting on couch on call",
            "nodding on video call", "smiling at camera", "laughing during call",
            "taking notes during call", "participating in meeting", "listening intently",
            "poor connection", "buffering on call", "frozen screen on video call",
            "bad lighting on webcam", "glasses glare on call", "echo on microphone"
        ],
        "phone_use": [
            "phone", "smartphone", "cell phone", "holding phone", "holding a phone",
            "scrolling", "texting", "typing on phone", "calling", "answering",
            "pressing on screen",
            "holding phone to ear", "holding phone in hand", "holding phone with two hands",
            "resting phone on lap", "phone on table", "phone propped up", "on speakerphone",
            "phone on shoulder", "cradling phone", "phone in pocket",
            "swiping on phone", "tapping on phone", "pinching to zoom", "taking photo with phone",
            "recording video on phone", "reading on phone", "checking notifications",
            "unlocking phone", "face id", "fingerprint scanner", "phone alarm going off",
            "phone conversation", "talking on phone", "listening on phone", "ending phone call",
            "dialing number", "receiving call", "rejecting call", "missed call",
            "using phone while walking", "using phone in bed", "using phone at table",
            "phone screen light on face", "blue light from phone", "phone battery low"
        ],
        "eating_drinking": [
            "drinking", "sip", "sipping", "cup", "mug", "coffee", "tea", "eating", "bite",
            "drinking from cup", "holding a cup", "holding a mug",
            "taking a sip", "gulping", "stirring coffee", "blowing on hot drink",
            "glass of water", "water bottle", "wine glass", "beer bottle", "soda can",
            "straw", "drinking through straw", "refilling cup", "pouring drink",
            "taking a bite", "chewing", "swallowing", "using fork", "using spoon", "using chopsticks",
            "cutting food with knife", "stirring food", "serving food", "passing food",
            "sandwich", "burger", "salad", "soup", "pasta", "rice bowl", "snack", "chips",
            "fruit", "apple", "banana", "granola bar", "yogurt", "cereal", "breakfast",
            "lunch", "dinner", "takeout container", "delivery food",
            "eating at desk", "eating during meeting", "lunch break", "coffee break",
            "holding fork and knife", "napkin on lap", "wiping mouth", "table setting",
            "placemat", "plate", "bowl", "utensils", "crumbs on desk"
        ],
        "meeting": [
            "meeting", "in-person", "with another person", "talking to someone", "two people",
            "group of people", "colleagues", "coworkers", "team meeting", "staff meeting",
            "client meeting", "one-on-one", "conversation", "discussion", "collaboration",
            "conference room", "meeting table", "whiteboard", "projector screen",
            "sitting around table", "standing at whiteboard", "presenting to group",
            "audience listening", "taking minutes", "facilitator", "moderator",
            "brainstorming session", "sharing ideas", "giving feedback", "taking notes",
            "asking question", "answering question", "raising hand in person",
            "shaking hands", "introducing oneself", "name tags", "business cards",
            "laptop open in meeting", "notebook and pen", "handouts", "agenda",
            "flip chart", "markers", "sticky notes", "printed documents",
            "morning meeting", "stand-up meeting", "briefing", "debriefing",
            "formal meeting", "casual catch-up", "networking event", "conference", "using computer",
            "looking at computers", "multiple laptops", "multiple computers"
        ],
        "using_computer": [
            "laptop", "computer", "keyboard", "typing", "touchpad",
            "working on", "browsing", "reading", "writing",
            "desktop pc", "macbook", "windows laptop", "external monitor", "dual screens",
            "wireless mouse", "usb drive", "headset", "webcam", "tablet",
            "clicking", "scrolling", "dragging and dropping", "copy pasting",
            "saving file", "printing document", "downloading", "uploading",
            "searching online", "googling", "researching", "editing document",
            "highlighting text", "deleting files", "organizing folders",
            "coding", "programming", "writing email", "checking email",
            "filling spreadsheet", "data entry", "creating presentation",
            "photo editing", "video editing", "graphic design", "illustrating",
            "hunched over keyboard", "leaning close to screen", "adjusting monitor height",
            "ergonomic setup", "standing desk", "wrist rest", "typing posture",
            "wearing headphones while typing", "glare on screen", "brightness adjustment",
            "plugging in charger", "battery low warning", "blue light from screen",
            "fingers on home row", "fast typer", "hunt and peck typing"
        ]
    }

    strong_video_call = [
        "zoom", "teams", "google meet", "facetime", "call controls", "gesturing", "raised hand",
        "waving", "laptop use + gesture + talking", "desk + gesture +talking", "gesture while at desk",
        "talking while using laptop", "talking while gesturing",
        "nodding in agreement on webcam", "surprised expression on video call", "person shrugging on screen",
        "facepalming during meeting", "thumbs up to camera", "applause on video call", "shaking head on webcam",
        "confused look during conference",
        "typing while looking at camera", "looking down at phone on call", "eating lunch on webcam",
        "drinking coffee during meeting", "holding baby on conference call", "dog interrupting zoom",
        "talking with muted microphone", "frantically trying to unmute", "background noise distraction",
        "turning away from screen to talk",
        "presenting slideshow on teams", "screen sharing document", "ceo giving update on webcam",
        "professional lighting for zoom", "home office setup", "waiting for host to start",
        "joining from coworking space",
    ]
    strong_phone_use = ["phone", "smartphone", "cell phone", "holding phone", "scrolling", "texting", "calling", "answering"]

    counts = {k: 0 for k in patterns.keys()}
    for label, cues in patterns.items():
        for cue in cues:
            counts[label] += t.count(cue)

    for cue in strong_video_call:
        if cue in t:
            counts["video_call"] += 2

    for cue in strong_phone_use:
        if cue in t:
            counts["phone_use"] += 1

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


def print_evidence(scores: Dict[str, int], closest: str):
    print("=== EVIDENCE SCORES ===\n")
    print(f"meeting         = {{{scores.get('meeting', 0)}}}")
    print(f"phone_use       = {{{scores.get('phone_use', 0)}}}")
    print(f"eating_drinking = {{{scores.get('eating_drinking', 0)}}}")
    print(f"video_call      = {{{scores.get('video_call', 0)}}}")
    print(f"using_computer  = {{{scores.get('using_computer', 0)}}}\n")
    print(f"closest={closest}\n")



def predict_main_label_llm(
    classifier: ActivityClassifierLLM,
    observations_text: str,
    debug: bool = False
) -> Tuple[Optional[str], str]:
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



def make_vlm(model_type: str, model_id: str, device: str, dtype: str, load_in_4bit: bool, max_new_tokens: int):
    model_type = model_type.lower().strip()
    if model_type == "llava":
        return LlavaCaptioner(model_id=model_id, device=device, dtype=dtype, load_in_4bit=load_in_4bit, max_new_tokens=max_new_tokens)
    if model_type == "qwen2vl":
        return Qwen2VLCaptioner(model_id=model_id, device=device, dtype=dtype, load_in_4bit=load_in_4bit, max_new_tokens=max_new_tokens)
    if model_type == "smolvlm":
        return SmolVLMCaptioner(model_id=model_id, device=device, dtype=dtype, load_in_4bit=load_in_4bit, max_new_tokens=max_new_tokens)
    if model_type == "minicpm":
        return MiniCPMCaptioner(model_id=model_id, device=device, dtype=dtype, load_in_4bit=load_in_4bit, max_new_tokens=max_new_tokens)
    if model_type == "internvl":
        if not HAS_INTERNVL:
            raise RuntimeError("InternVL adapter not found. Add src/models/internvl_captioner.py first.")
        return InternVLCaptioner(model_id=model_id, device=device, dtype=dtype, load_in_4bit=load_in_4bit, max_new_tokens=max_new_tokens)

    raise ValueError(f"Unknown VLM type: {model_type}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--mode", choices=["stm", "none"], default="stm")
    parser.add_argument("--model", required=True, help="Model name from config.yaml vlm.models[].name (e.g. llava15_7b)")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--margin", type=int, default=3)
    args = parser.parse_args()

    cfg = Config.load("config.yaml")
    data_cfg = cfg["data"]
    vlm_cfg = cfg["vlm"]
    llm_cfg = cfg["llm"]
    mem_cfg = cfg["memory"]


    wanted = args.model.strip().lower()
    model_entry = None
    for m in vlm_cfg.get("models", []):
        if m.get("name", "").lower() == wanted:
            model_entry = m
            break
    if model_entry is None:
        raise ValueError(f"Could not find model '{args.model}' in config.yaml under vlm.models")

    model_name = model_entry["name"]
    model_type = model_entry["type"]
    model_id = model_entry["model_id"]

    print("\n====================================================")
    print(f"VLM MODEL: {model_name} ({model_type})")
    print("====================================================\n")
    print(f"MODE: {args.mode.upper()}\n")


    labels_map = load_video_labels(data_cfg["labels_csv"])
    vid_name = os.path.basename(args.video)
    true_label = labels_map.get(vid_name, None)


    classifier = ActivityClassifierLLM(
        model_id=llm_cfg["model_id"],
        device=cfg["device"],
        dtype=cfg["dtype"],
        load_in_4bit=llm_cfg["load_in_4bit"],
        max_new_tokens=int(llm_cfg["max_new_tokens"]),
        temperature=float(llm_cfg["temperature"]),
        system_prompt=llm_cfg.get("system_prompt", ""),
    )


    captioner = make_vlm(
        model_type=model_type,
        model_id=model_id,
        device=cfg["device"],
        dtype=cfg["dtype"],
        load_in_4bit=bool(vlm_cfg["load_in_4bit"]),
        max_new_tokens=int(vlm_cfg["max_new_tokens"]),
    )

    frames = extract_frames_with_timestamps(
        args.video,
        stride_seconds=float(data_cfg["frame_stride_seconds"]),
        max_frames=int(data_cfg["max_frames_per_video"]),
    )
    frames.sort(key=lambda x: x.timestamp_s)

    per_frame: List[Tuple[float, str]] = []
    for fr in frames:
        t = float(fr.timestamp_s)
        cap = captioner.caption(fr.bgr, vlm_cfg["prompt"])
        cap = cap.replace("ASSISTANT:", "").replace("USER:", "").strip()
        per_frame.append((t, cap))

    if args.debug:
        print("\n=== PER-FRAME CAPTIONS ===\n")
        for t, cap in per_frame:
            print(f"[{t:06.2f}s] {cap}")

    observations = build_observations(per_frame, args.mode, mem_cfg)

    if args.debug:
        print("\n=== OBSERVATIONS ===\n")
        print(observations[:4000], "...\n" if len(observations) > 4000 else "\n")

    scores = score_labels_from_text(observations)
    closest = pick_closest_label(scores)

    llm_pred, raw_out = predict_main_label_llm(classifier, observations, debug=args.debug)
    pred_label = decide_final_label(llm_pred, scores, closest, margin=args.margin)

    if args.debug:
        print_evidence(scores, closest)

    print("=== MAIN ACTION LABEL ===\n")
    print(f"pred={pred_label}")
    if true_label is None:
        print("true=<missing label in labels csv>")
    else:
        print(f"true={true_label}")
        f1 = f1_score([true_label], [pred_label], average="macro")
        print(f"F1_macro (single video) = {float(f1):.4f}")

    # cleanup
    del captioner
    del classifier
    free_vram()


if __name__ == "__main__":
    main()