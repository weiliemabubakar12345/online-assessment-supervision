"""Cascaded two-model VLM analysis service for the exam monitor.

Implements the architecture in Fig. "cascade-flow": a small, cheap model
(SmolVLM-500M) screens every screenshot; only screenshots it cannot clear
confidently are escalated to the large, dominant model (Qwen2.5-VL), which
receives the small model's assessment as an advisory second opinion
(single-round consultation, NOT an iterative debate).

    screenshot -> SmolVLM-500M -> [confident AND plainly just the exam page?]
                                     |- yes -> not cheating          (fast path)
                                     '- no  -> Qwen2.5-VL + 2nd opinion -> verdict

The gate asks "is this plainly nothing but the exam page", NOT "did the small
model decide it is not cheating". Those differ whenever the small model returns
an unusable answer or misjudges exam_relevance, and reading either as innocence
made the cascade skip exactly the screenshots it exists to catch.

Why cascade and not a debate loop: an iterative exchange multiplies
T_inference by the number of rounds, and the 500M model is far too weak to
change a 7B model's mind through argument. The cascade instead SAVES
inference on the majority of benign screenshots, which is what the
end-to-end latency metric (T_total = T_capture + T_network + T_inference)
actually rewards.

The same service also runs the two single-model baselines, so an accuracy or
latency difference cannot come from a different code path, prompt or parser:

    MODE=cascade     small screens, large decides on escalation   (default)
    MODE=large_only  large model on every screenshot              (baseline A)
    MODE=small_only  small model on every screenshot              (baseline B)

Endpoints:
  GET  /health                        -> models, devices, mode, threshold
  GET  /config                        -> {lang, exam_context, mode, conf_threshold}
  POST /config  {"lang": "en"}        -> switch to a built-in exam context
                {"exam_context": "..."} -> set a custom one
                {"mode": "large_only"}  -> switch routing mode
                {"conf_threshold": 0.6} -> switch the escalation gate
  POST /analyze {"image": "<dataURL|base64>"}
       -> {category, identity, exam_relevance, summary, is_cheating,
           latency_ms, path, escalated, small, large}
  GET  /stats                         -> escalation rate + per-path latency
  POST /stats/reset                   -> clear the counters between runs

Setup:
  pip install flask torch transformers qwen-vl-utils accelerate pillow

Run (alongside the single-model service on 8788):
  REQUIRE_CUDA=1 PORT=8791 python vlm_service_cascade.py

VRAM: Qwen2.5-VL-7B in fp16 is ~15 GB of weights and does NOT fit in one T4's
usable ~15 GB once activations are added. Placement options:

  LARGE_DEVICE=auto            shard the large model across every visible GPU
                               (the only way to run 7B in fp16 on T4 x2)
  LARGE_MAX_MEMORY=0:9GiB,1:9GiB   cap the shard per GPU, so the small model
                               and cv_service.py still have room
  LARGE_LOAD_4BIT=1            ~5 GB instead, fits one GPU (needs bitsandbytes,
                               and changes the numerics you are reporting)
  LARGE_MODEL_ID=Qwen/Qwen2.5-VL-3B-Instruct   what the single-model deployment
                               already uses; ~7 GB, comfortable on one T4

Defaults (no env set): large on cuda:0, small on cuda:1 when a second GPU
exists, otherwise both on the same device.
"""
import base64
import io
import json
import os
import re
import threading
import time
from datetime import datetime

# Reduce fragmentation-driven OOM (must be set before CUDA context init).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from flask import Flask, request, jsonify
from PIL import Image
import torch
import transformers
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# transformers 5.x dropped AutoModelForVision2Seq in favour of
# AutoModelForImageTextToText (Kaggle images already ship 5.x); keep the old
# name working so the same file runs on an older local install too.
try:
    from transformers import AutoModelForImageTextToText as AutoVLM
except ImportError:                                     # transformers < 4.46
    from transformers import AutoModelForVision2Seq as AutoVLM

# Same story for the dtype kwarg: renamed torch_dtype -> dtype in 5.x. The old
# name is still accepted there "for BC", but that BC will not last forever.
_DTYPE_KW = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"

# ---------------------------------------------------------------- config ----

LARGE_MODEL_ID = os.environ.get("LARGE_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")
SMALL_MODEL_ID = os.environ.get("SMALL_MODEL_ID", "HuggingFaceTB/SmolVLM-500M-Instruct")

PORT = int(os.environ.get("PORT", 8791))
HOST = os.environ.get("HOST", "127.0.0.1")

MODE = os.environ.get("MODE", "cascade")            # cascade|large_only|small_only
VALID_MODES = {"cascade", "large_only", "small_only"}

# Gate for the fast path. The small model must report a usable answer of
# exactly "exam_page" AND be confident above this threshold; anything else
# escalates. Raising it escalates more often -> safer, slower. This is the knob
# to sweep when reporting the accuracy/latency trade-off.
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", 0.60))

# Visual-token caps. Keep identical to the benchmark notebook, or the accuracy
# you report is not the accuracy you deployed.
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1024 * 28 * 28
SMALL_LONGEST_EDGE = int(os.environ.get("SMALL_LONGEST_EDGE", 4 * 384))

MAX_NEW_TOKENS = 192
SMALL_MAX_NEW_TOKENS = 160

# Categories that can constitute cheating. exam_page is excluded: an official
# exam-platform page is never cheating regardless of relevance.
CHEAT_CATEGORIES = {"ai_assistant", "search_engine",
                    "reference_material", "other_application"}

# Every label the prompt allows. Anything outside this set means the model did
# not actually answer -- most often a small model echoing the prompt's JSON
# template verbatim ("<one label above>"), which is syntactically valid JSON and
# therefore sails through the parser. Treated as an invalid answer, never as
# "nothing suspicious".
VALID_CATEGORIES = CHEAT_CATEGORIES | {"exam_page"}
VALID_RELEVANCE = {"direct", "partial", "none"}

EXAM_CONTEXTS = {
    "en": ("An online English level test (International English Test), 60 multiple-choice "
           "questions on English grammar and vocabulary. Current question (Q3, grammar): "
           "'____ 20 chairs in the office.' Options: 'This is', 'There is', 'They are', "
           "'There are'."),
    "ko": ("An online Korean language level test (90 Day Korean Level Test), multiple-choice "
           "questions on basic Korean vocabulary and expressions. Current question (Q2): "
           "'How do you say \"thank you\" in Korean?' Options: 죄송합니다, 사랑해요, 감사합니다, "
           "안녕하세요."),
    "zh": ("An online Chinese proficiency practice test (Dong Chinese, HSK level 1), matching "
           "a sentence to the correct picture. Current question: '这是我先生的衣服。' — choose "
           "the image that matches the text."),
    "ja": ("An online Japanese proficiency practice test (JLPTCheck, JLPT style), "
           "fill-in-the-blank multiple choice. Current question: "
           "'きょうは５００＿およぎました。' Options: ど, ばん, メートル, グラム."),
}

LANG = os.environ.get("EXAM_LANG", "en")
EXAM_CONTEXT = EXAM_CONTEXTS.get(LANG, EXAM_CONTEXTS["en"])

if MODE not in VALID_MODES:
    raise SystemExit("MODE must be one of %s, got %r" % (sorted(VALID_MODES), MODE))

if os.environ.get("REQUIRE_CUDA") == "1":
    assert torch.cuda.is_available(), (
        "REQUIRE_CUDA=1 but no GPU is visible. On Kaggle: "
        "Settings > Accelerator > GPU T4 x2, then restart the session.")

# Serialize generate() across both models: a burst of screenshots must never
# run two forward passes at once, which would multiply peak VRAM.
_GEN_LOCK = threading.Lock()
_STATS_LOCK = threading.Lock()

# ---------------------------------------------------------------- prompts ---

# NOTE: contains literal { } for the JSON schema -> fill with .replace(),
# never .format() / f-string.
PROMPT_TEMPLATE = """You are a screen-analysis component in an online-exam monitoring system.

EXAM CONTEXT (what the student is supposed to be working on):
{EXAM_CONTEXT}

Given ONE screenshot of a student's screen, produce TWO independent judgments.

1) category - classify the PRIMARY non-exam content into exactly ONE label:
- exam_page: the online exam or quiz itself, or an official page of the exam platform (instructions, an allowed calculator)
- ai_assistant: an AI chatbot (ChatGPT, Claude, Gemini, Copilot, Perplexity, Doubao, Kimi, ERNIE/Wenxin, DeepSeek, etc.)
- search_engine: a search engine or its results page (Google, Bing, DuckDuckGo, Baidu, Naver, Yahoo! Japan)
- reference_material: encyclopedia, dictionary, docs, tutorial, PDF, or Q&A site (Wikipedia, Namu Wiki, Baidu Baike, Zdic, MDN, StackOverflow, Zhihu)
- other_application: any other website or app (news, video, shopping, maps, social, messaging like WeChat/LINE, music, a notes or word-processor window, etc.)

MULTI-WINDOW RULE: if the exam is visible at the same time as another resource
(split screen, two windows side by side, a popup over the exam), classify the
OTHER resource, not the exam. Choose exam_page ONLY when the exam or an official
exam-platform page is the only substantive content on screen.

2) exam_relevance - how related the visible content is to the EXAM CONTEXT above:
- direct: the exam's question, its wording, or its answer is visibly being looked up, typed, or answered
- partial: the same subject or topic as the exam, but not the specific question
- none: unrelated to the exam content - a different subject, entertainment, personal matters, or a neutral/empty page

Judge exam_relevance ONLY against the EXAM CONTEXT above. A page can be academic
and still be "none" if it has nothing to do with this exam. If category is
exam_page, set exam_relevance to "direct".

The screen may be in ANY language (including Chinese, Japanese, Korean). Classify by
meaning and visual layout, NOT by language. Always write "summary" in English.

Output ONLY valid JSON (no markdown):
{"category":"<one label above>","identity":"<site/app name if legible else 'unknown'>","exam_relevance":"<direct|partial|none>","summary":"<one English sentence describing what is shown>"}"""

# Appended to the large model's prompt on escalation. Advisory, never binding:
# the dominant model must be free to overrule a wrong screening verdict,
# otherwise the cascade would inherit the small model's errors.
SECOND_OPINION_BLOCK = """

SECOND OPINION (from a smaller screening model that flagged this screenshot for review):
  category: {SO_CATEGORY}
  exam_relevance: {SO_RELEVANCE}
  identity: {SO_IDENTITY}
  summary: {SO_SUMMARY}
  confidence: {SO_CONF}

Treat this as advisory only. It comes from a weaker model and may be wrong.
Judge the screenshot yourself and overrule it whenever your own reading differs."""


def build_prompt():
    return PROMPT_TEMPLATE.replace("{EXAM_CONTEXT}", EXAM_CONTEXT)


def build_escalation_prompt(small_out, small_conf):
    block = (SECOND_OPINION_BLOCK
             .replace("{SO_CATEGORY}", str(small_out.get("category", "unclear")))
             .replace("{SO_RELEVANCE}", str(small_out.get("exam_relevance", "none")))
             .replace("{SO_IDENTITY}", str(small_out.get("identity", "unknown")))
             .replace("{SO_SUMMARY}", str(small_out.get("summary", ""))[:200])
             .replace("{SO_CONF}", "%.2f" % small_conf))
    return build_prompt() + block


# ------------------------------------------------------------- utilities ----

def dataurl_to_pil(s):
    if isinstance(s, str) and s.strip().startswith("data:") and "," in s:
        s = s.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB")


def parse_json(raw):
    """Return (dict, parsed_ok). A parse failure is a hard escalation trigger."""
    s = re.sub(r"```$", "", re.sub(r"^```(?:json)?", "", raw.strip())).strip()
    for cand in (s, None):
        if cand is None:
            m = re.search(r"\{.*\}", s, re.DOTALL)
            if not m:
                break
            cand = m.group(0)
        try:
            d = json.loads(cand)
            if isinstance(d, dict):
                return d, True
        except Exception:
            continue
    return ({"category": "unclear", "identity": "unknown",
             "exam_relevance": "none", "summary": raw[:160]}, False)


def decide(out):
    """Apply the cheating rule. Returns (is_cheating, warning_or_None).

    A missing exam_relevance is treated as "none" so the service keeps serving,
    but it is surfaced -- silently defaulting to "none" makes the system flag
    nothing at all, which is indistinguishable from a well-behaved student.
    """
    cat = str(out.get("category") or "").strip().lower()
    rel_raw = out.get("exam_relevance")
    warning = None
    if rel_raw is None:
        warning = "model did not return exam_relevance; treated as 'none'"
        rel = "none"
    else:
        rel = str(rel_raw).strip().lower()
        if rel not in {"direct", "partial", "none"}:
            warning = "unexpected exam_relevance %r; treated as 'none'" % (rel_raw,)
            rel = "none"
    return (cat in CHEAT_CATEGORIES and rel != "none"), warning


def output_valid(out):
    """Did the model actually answer? Returns (ok, reason_or_None).

    Separate from decide() on purpose. decide() answers "is this cheating",
    which for any unrecognised label is False -- and a False from garbage must
    never be read as evidence of innocence. This tells the gate whether the
    answer is usable at all.
    """
    cat = str(out.get("category") or "").strip().lower()
    rel = str(out.get("exam_relevance") or "").strip().lower()
    if cat not in VALID_CATEGORIES:
        return False, "category %r is not one of the allowed labels" % (
            out.get("category"),)
    if rel not in VALID_RELEVANCE:
        return False, "exam_relevance %r is not one of the allowed values" % (
            out.get("exam_relevance"),)
    return True, None


def sequence_confidence(model, sequences, scores):
    """Mean per-token probability of the greedy generation, in [0, 1].

    Used as the small model's confidence. Not a calibrated probability of
    being correct -- it is a proxy -- but it is monotonic enough to gate on,
    and it is computed identically for every screenshot, which is what the
    escalation-rate comparison needs.
    """
    try:
        tr = model.compute_transition_scores(sequences, scores, normalize_logits=True)
        tr = tr.float()
        mask = torch.isfinite(tr)
        if mask.sum() == 0:
            return 0.0
        return float(torch.exp(tr[mask].mean()).clamp(0.0, 1.0))
    except Exception:
        return 0.0


# ------------------------------------------------------------- model defs ---

DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
N_GPU = torch.cuda.device_count() if torch.cuda.is_available() else 0

# Default placement: large on cuda:0, small on the second GPU when there is
# one (Kaggle T4 x2), otherwise both on the same device.
#
# LARGE_DEVICE=auto shards the large model across every visible GPU via
# accelerate. Needed for Qwen2.5-VL-7B in fp16 (~15 GB of weights), which does
# NOT fit in one T4's usable ~15 GB once activations are added -- pinning it to
# a single device OOMs during the first forward pass, not at load time.
LARGE_DEVICE = os.environ.get("LARGE_DEVICE") or ("cuda:0" if N_GPU else "cpu")
LARGE_SHARDED = LARGE_DEVICE.strip().lower() == "auto"
SMALL_DEVICE = os.environ.get("SMALL_DEVICE") or (
    "cuda:1" if N_GPU > 1 else ("cuda:0" if N_GPU else "cpu"))


class QwenVLM:
    """The dominant model. float16 + sdpa: a T4 is Turing (sm75), which has
    neither native bf16 nor flash-attention-2."""

    def __init__(self):
        print("Loading LARGE", LARGE_MODEL_ID, "->", LARGE_DEVICE, "...", flush=True)
        kw = {_DTYPE_KW: DTYPE, "attn_implementation": "sdpa"}
        if os.environ.get("LARGE_LOAD_4BIT") == "1":
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=DTYPE,
                bnb_4bit_quant_type="nf4")
            kw["device_map"] = "auto" if LARGE_SHARDED else {"": LARGE_DEVICE}
        elif torch.cuda.is_available():
            kw["device_map"] = "auto" if LARGE_SHARDED else {"": LARGE_DEVICE}
            if LARGE_SHARDED and os.environ.get("LARGE_MAX_MEMORY"):
                # e.g. LARGE_MAX_MEMORY="0:9GiB,1:9GiB" -- leaves room on both
                # GPUs for the small model and the CV service, instead of
                # letting accelerate fill cuda:0 to the brim first.
                kw["max_memory"] = dict(
                    (int(k), v) for k, v in
                    (part.split(":", 1) for part in
                     os.environ["LARGE_MAX_MEMORY"].split(",")))
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            LARGE_MODEL_ID, **kw)
        if not torch.cuda.is_available():
            self.model.to("cpu")
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            LARGE_MODEL_ID, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)

    def run(self, img, prompt):
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(msgs)
        inp = self.processor(text=[text], images=imgs, videos=vids,
                             padding=True, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            gen = self.model.generate(**inp, max_new_tokens=MAX_NEW_TOKENS,
                                      do_sample=False, return_dict_in_generate=True,
                                      output_scores=True)
        trimmed = [o[len(i):] for i, o in zip(inp.input_ids, gen.sequences)]
        raw = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
        conf = sequence_confidence(self.model, gen.sequences, gen.scores)
        return raw, conf


class SmolVLM:
    """The non-dominant screening model. Idefics3-style API: no
    qwen_vl_utils, images are passed to the processor directly."""

    def __init__(self):
        print("Loading SMALL", SMALL_MODEL_ID, "->", SMALL_DEVICE, "...", flush=True)
        try:
            self.model = AutoVLM.from_pretrained(
                SMALL_MODEL_ID, attn_implementation="sdpa", **{_DTYPE_KW: DTYPE})
        except Exception as e:
            print("  sdpa unavailable (%s); falling back to eager" % type(e).__name__)
            self.model = AutoVLM.from_pretrained(SMALL_MODEL_ID, **{_DTYPE_KW: DTYPE})
        self.model.to(SMALL_DEVICE).eval()
        self.processor = AutoProcessor.from_pretrained(
            SMALL_MODEL_ID, size={"longest_edge": SMALL_LONGEST_EDGE})

    def run(self, img, prompt):
        msgs = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(msgs, add_generation_prompt=True)
        inp = self.processor(text=text, images=[img], return_tensors="pt")
        inp = {k: (v.to(SMALL_DEVICE) if hasattr(v, "to") else v) for k, v in inp.items()}
        with torch.inference_mode():
            gen = self.model.generate(**inp, max_new_tokens=SMALL_MAX_NEW_TOKENS,
                                      do_sample=False, return_dict_in_generate=True,
                                      output_scores=True)
        in_len = inp["input_ids"].shape[1]
        raw = self.processor.batch_decode(
            gen.sequences[:, in_len:], skip_special_tokens=True)[0].strip()
        conf = sequence_confidence(self.model, gen.sequences, gen.scores)
        return raw, conf


LARGE = QwenVLM() if MODE in {"cascade", "large_only"} else None
SMALL = SmolVLM() if MODE in {"cascade", "small_only"} else None
print("Ready | mode=%s | conf_threshold=%.2f | exam context=%s"
      % (MODE, CONF_THRESHOLD, LANG), flush=True)

# ------------------------------------------------------------------ stats ---

STATS = {"n": 0, "n_fast": 0, "n_escalated": 0,
         "sum_small_ms": 0.0, "sum_large_ms": 0.0,
         "sum_total_ms": 0.0, "sum_fast_ms": 0.0, "sum_esc_ms": 0.0,
         "n_small_flag": 0, "n_final_flag": 0, "n_overruled": 0,
         "n_parse_fail_small": 0}


def _record(path, t_small, t_large, t_total, small_flag, final_flag,
            overruled, parse_fail_small):
    with _STATS_LOCK:
        STATS["n"] += 1
        STATS["sum_small_ms"] += t_small or 0.0
        STATS["sum_large_ms"] += t_large or 0.0
        STATS["sum_total_ms"] += t_total
        if path == "fast":
            STATS["n_fast"] += 1
            STATS["sum_fast_ms"] += t_total
        elif path == "escalated":
            STATS["n_escalated"] += 1
            STATS["sum_esc_ms"] += t_total
        STATS["n_small_flag"] += int(bool(small_flag))
        STATS["n_final_flag"] += int(bool(final_flag))
        STATS["n_overruled"] += int(bool(overruled))
        STATS["n_parse_fail_small"] += int(bool(parse_fail_small))


# ---------------------------------------------------------------- analyze ---

def _err(msg):
    return {"category": "error", "identity": "unknown", "exam_relevance": "none",
            "summary": msg, "is_cheating": False, "latency_ms": None}


def _run_guarded(runner, img, prompt, tag):
    """Run one model under the global lock.

    Returns (raw, conf, queue_ms, generate_ms, error). The two timings are kept
    apart on purpose: when several screenshots arrive at once they serialise on
    _GEN_LOCK, and folding that wait into "inference time" makes a queue backlog
    look like a slow model. Report generate_ms as inference; queue_ms belongs to
    throughput, not to the model.
    """
    t_enter = time.perf_counter()
    with _GEN_LOCK:
        t_start = time.perf_counter()
        queue_ms = (t_start - t_enter) * 1000.0
        try:
            raw, conf = runner.run(img, prompt)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return None, 0.0, queue_ms, None, (
                "%s: GPU out of memory; lower MAX_PIXELS, the capture "
                "resolution, or use LARGE_LOAD_4BIT=1" % tag)
        except Exception as e:
            return None, 0.0, queue_ms, None, "%s inference failed: %s: %s" % (
                tag, type(e).__name__, e)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return raw, conf, queue_ms, (time.perf_counter() - t_start) * 1000.0, None


def analyze(img):
    t_wall = time.perf_counter()
    t_small = t_large = None
    small_queue = small_gen = large_queue = large_gen = None
    small_out = None
    small_conf = 0.0
    small_flag = None
    parse_ok_small = True
    small_valid = True
    small_invalid_reason = None
    small_warning = None

    # -- stage 1: screening by the non-dominant model ------------------------
    if MODE in {"cascade", "small_only"}:
        raw, small_conf, small_queue, small_gen, err = _run_guarded(
            SMALL, img, build_prompt(), "small")
        if err:
            return _err(err)
        t_small = small_queue + small_gen
        small_out, parse_ok_small = parse_json(raw)
        small_out.setdefault("identity", "unknown")
        small_out.setdefault("summary", "")
        small_flag, small_warning = decide(small_out)
        small_valid, small_invalid_reason = output_valid(small_out)

    if MODE == "small_only":
        result = dict(small_out)
        is_cheating, warning = decide(result)
        total_ms = (time.perf_counter() - t_wall) * 1000.0
        result.update({"is_cheating": is_cheating, "latency_ms": round(total_ms),
                       "path": "small_only", "escalated": False,
                       "small": {"category": small_out.get("category"),
                                 "exam_relevance": small_out.get("exam_relevance"),
                                 "is_cheating": small_flag,
                                 "schema_valid": small_valid,
                                 "confidence": round(small_conf, 3),
                                 "queue_ms": round(small_queue),
                                 "generate_ms": round(small_gen),
                                 "latency_ms": round(t_small)}})
        if warning:
            result["warning"] = warning
        _record("small_only", t_small, None, total_ms, small_flag, is_cheating,
                False, (not parse_ok_small) or (not small_valid))
        _log(result)
        return result

    # -- gate: clear the fast path only when the screen is plainly the exam ---
    #
    # The gate deliberately does NOT key off the small model's cheating verdict.
    # That verdict is False in three very different situations: the screen
    # really is just the exam; the model echoed the prompt template instead of
    # answering; or the model saw ChatGPT but judged exam_relevance as "none".
    # Only the first is evidence of innocence. Keying off "not cheating" made
    # the cascade skip precisely the screenshots it exists to catch.
    #
    # So the small model is trusted for exactly one claim -- "there is nothing
    # here but the exam page" -- and everything else escalates. Relevance is the
    # hard judgment; it belongs to the dominant model.
    if MODE == "cascade":
        clean = (str(small_out.get("category") or "").strip().lower()
                 == "exam_page") and parse_ok_small and small_valid
        confident = small_conf >= CONF_THRESHOLD
        if clean and confident:
            total_ms = (time.perf_counter() - t_wall) * 1000.0
            result = {
                "category": small_out.get("category", "exam_page"),
                "identity": small_out.get("identity", "unknown"),
                "exam_relevance": small_out.get("exam_relevance", "none"),
                "summary": small_out.get("summary", ""),
                "is_cheating": False,
                "latency_ms": round(total_ms),
                "path": "fast",
                "escalated": False,
                "gate": {"clean": clean, "confident": confident,
                         "confidence": round(small_conf, 3),
                         "threshold": CONF_THRESHOLD,
                         "rule": "category == exam_page"},
                "small": {"category": small_out.get("category"),
                          "exam_relevance": small_out.get("exam_relevance"),
                          "is_cheating": False,
                          "schema_valid": small_valid,
                          "confidence": round(small_conf, 3),
                          "queue_ms": round(small_queue),
                          "generate_ms": round(small_gen),
                          "latency_ms": round(t_small)},
                "large": None,
            }
            if small_warning:
                result["warning"] = small_warning
            _record("fast", t_small, None, total_ms, False, False, False,
                    (not parse_ok_small) or (not small_valid))
            _log(result)
            return result

    # -- stage 2: the dominant model decides ---------------------------------
    prompt = (build_escalation_prompt(small_out, small_conf)
              if MODE == "cascade" and small_out else build_prompt())
    raw, large_conf, large_queue, large_gen, err = _run_guarded(LARGE, img, prompt, "large")
    if err:
        return _err(err)
    t_large = large_queue + large_gen
    large_out, parse_ok_large = parse_json(raw)
    large_out.setdefault("identity", "unknown")
    large_out.setdefault("summary", "")
    is_cheating, warning = decide(large_out)

    total_ms = (time.perf_counter() - t_wall) * 1000.0
    overruled = (MODE == "cascade" and small_flag is not None
                 and bool(small_flag) != bool(is_cheating))

    result = dict(large_out)
    result.update({
        "is_cheating": is_cheating,
        "latency_ms": round(total_ms),
        "path": "escalated" if MODE == "cascade" else "large_only",
        "escalated": MODE == "cascade",
        "large": {"category": large_out.get("category"),
                  "exam_relevance": large_out.get("exam_relevance"),
                  "is_cheating": is_cheating,
                  "confidence": round(large_conf, 3),
                  "parsed_ok": parse_ok_large,
                  "queue_ms": round(large_queue),
                  "generate_ms": round(large_gen),
                  "latency_ms": round(t_large)},
    })
    if MODE == "cascade":
        small_cat = str(small_out.get("category") or "").strip().lower()
        result["gate"] = {"clean": small_cat == "exam_page" and parse_ok_small
                                   and small_valid,
                          "confident": small_conf >= CONF_THRESHOLD,
                          "confidence": round(small_conf, 3),
                          "threshold": CONF_THRESHOLD,
                          "rule": "category == exam_page",
                          "escalation_reason": (
                              small_invalid_reason if not small_valid
                              else "json parse failed" if not parse_ok_small
                              else "confidence %.3f < %.2f" % (small_conf, CONF_THRESHOLD)
                              if small_conf < CONF_THRESHOLD
                              else "category %r is not exam_page" % small_cat)}
        result["small"] = {"category": small_out.get("category"),
                           "exam_relevance": small_out.get("exam_relevance"),
                           "is_cheating": small_flag,
                           "parsed_ok": parse_ok_small,
                           "schema_valid": small_valid,
                           "confidence": round(small_conf, 3),
                           "queue_ms": round(small_queue),
                           "generate_ms": round(small_gen),
                           "latency_ms": round(t_small)}
        result["overruled_small"] = overruled
    if warning:
        result["warning"] = warning

    _record(result["path"], t_small, t_large, total_ms, small_flag, is_cheating,
            overruled,
            ((not parse_ok_small) or (not small_valid)) if small_out else False)
    _log(result)
    return result


def _log(r):
    small = r.get("small") or {}
    gate = r.get("gate") or {}
    bits = []
    if small.get("category") is not None:
        bits.append("small=%s/%s" % (small.get("category"), small.get("exam_relevance")))
    if small.get("schema_valid") is False:
        bits.append("SMALL-UNUSABLE")
    if gate.get("escalation_reason"):
        bits.append("why=%s" % gate["escalation_reason"])
    if r.get("overruled_small"):
        bits.append("OVERRULED")
    if r.get("warning"):
        bits.append("!! " + r["warning"])
    print("[%s] %-10s %-18s rel=%-8s cheat=%-5s conf=%-5s %5dms  %s" % (
        datetime.now().strftime("%H:%M:%S"),
        r.get("path"), r.get("category"), r.get("exam_relevance"),
        r.get("is_cheating"),
        ("%.2f" % small["confidence"]) if small.get("confidence") is not None else "-",
        r.get("latency_ms") or 0, "  ".join(bits)), flush=True)


# --------------------------------------------------------------- endpoints --

app = Flask(__name__)


@app.route("/health")
def health():
    return jsonify({
        "status": "ok", "mode": MODE, "conf_threshold": CONF_THRESHOLD,
        "large_model": LARGE_MODEL_ID if LARGE else None,
        "small_model": SMALL_MODEL_ID if SMALL else None,
        "large_device": LARGE_DEVICE if LARGE else None,
        "small_device": SMALL_DEVICE if SMALL else None,
        "n_gpu": N_GPU, "lang": LANG, "exam_context": EXAM_CONTEXT})


@app.route("/config", methods=["GET", "POST"])
def config():
    global EXAM_CONTEXT, LANG, MODE, CONF_THRESHOLD
    if request.method == "POST":
        data = request.get_json(force=True, silent=True) or {}
        touched = False

        lang, ctx = data.get("lang"), data.get("exam_context")
        if lang:
            if lang not in EXAM_CONTEXTS:
                return jsonify({"error": "unknown lang %r; known: %s"
                                % (lang, sorted(EXAM_CONTEXTS))}), 400
            LANG, EXAM_CONTEXT, touched = lang, EXAM_CONTEXTS[lang], True
        elif isinstance(ctx, str) and ctx.strip():
            LANG, EXAM_CONTEXT, touched = "custom", ctx.strip(), True

        if "mode" in data:
            m = str(data["mode"])
            if m not in VALID_MODES:
                return jsonify({"error": "mode must be one of %s" % sorted(VALID_MODES)}), 400
            if m in {"cascade", "large_only"} and LARGE is None:
                return jsonify({"error": "large model not loaded; restart with MODE=%s" % m}), 409
            if m in {"cascade", "small_only"} and SMALL is None:
                return jsonify({"error": "small model not loaded; restart with MODE=%s" % m}), 409
            MODE, touched = m, True

        if "conf_threshold" in data:
            try:
                v = float(data["conf_threshold"])
            except (TypeError, ValueError):
                return jsonify({"error": "conf_threshold must be a number in [0,1]"}), 400
            if not 0.0 <= v <= 1.0:
                return jsonify({"error": "conf_threshold must be in [0,1]"}), 400
            CONF_THRESHOLD, touched = v, True

        if not touched:
            return jsonify({"error": "send lang, exam_context, mode or conf_threshold"}), 400
        print("[config] mode=%s conf_threshold=%.2f lang=%s"
              % (MODE, CONF_THRESHOLD, LANG), flush=True)
    return jsonify({"lang": LANG, "exam_context": EXAM_CONTEXT,
                    "mode": MODE, "conf_threshold": CONF_THRESHOLD})


@app.route("/analyze", methods=["POST"])
def analyze_route():
    data = request.get_json(force=True, silent=True) or {}
    if "image" not in data:
        return jsonify({"error": "missing 'image'"}), 400
    try:
        img = dataurl_to_pil(data["image"])
    except Exception as e:
        return jsonify({"error": "bad image: " + str(e)}), 400
    return jsonify(analyze(img))


@app.route("/stats")
def stats():
    with _STATS_LOCK:
        s = dict(STATS)
    n = max(s["n"], 1)
    return jsonify({
        "mode": MODE, "conf_threshold": CONF_THRESHOLD,
        "n_screenshots": s["n"],
        "n_fast_path": s["n_fast"],
        "n_escalated": s["n_escalated"],
        "escalation_rate": round(s["n_escalated"] / n, 4),
        "mean_total_ms": round(s["sum_total_ms"] / n, 1),
        "mean_fast_path_ms": round(s["sum_fast_ms"] / max(s["n_fast"], 1), 1),
        "mean_escalated_ms": round(s["sum_esc_ms"] / max(s["n_escalated"], 1), 1),
        "mean_small_ms": round(s["sum_small_ms"] / n, 1),
        "mean_large_ms": round(s["sum_large_ms"] / max(s["n_escalated"], 1), 1),
        "n_small_flagged": s["n_small_flag"],
        "n_final_flagged": s["n_final_flag"],
        "n_large_overruled_small": s["n_overruled"],
        "n_small_unusable_answers": s["n_parse_fail_small"]})


@app.route("/stats/reset", methods=["POST"])
def stats_reset():
    with _STATS_LOCK:
        for k in STATS:
            STATS[k] = 0 if isinstance(STATS[k], int) else 0.0
    return jsonify({"status": "reset"})


if __name__ == "__main__":
    print("Serving on http://%s:%d  (mode=%s)" % (HOST, PORT, MODE))
    app.run(host=HOST, port=PORT)
