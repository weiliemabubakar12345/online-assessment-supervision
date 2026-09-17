"""Local VLM analysis service for the exam monitor (Qwen2.5-VL-3B).

The Node proctor server (server.js) POSTs each screenshot here; this service runs
the VLM to identify what is on screen and returns a JSON conclusion that the
dashboard displays.

Endpoints:
  GET  /health                        -> {status, device, model, lang, exam_context}
  GET  /config                        -> {lang, exam_context}
  POST /config  {"lang": "en"}        -> switch to a built-in exam context
                {"exam_context": "..."} -> set a custom one
  POST /analyze {"image": "<dataURL|base64>"}
       -> {category, identity, exam_relevance, summary, is_cheating, latency_ms}

Setup:
  pip install flask torch transformers qwen-vl-utils accelerate pillow

Run:
  REQUIRE_CUDA=1 python vlm_service.py      # serves http://localhost:8788

On Kaggle set REQUIRE_CUDA=1 so a session that silently started with
"No Accelerator" fails loudly instead of crawling on CPU for hours.
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
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
PORT = int(os.environ.get("PORT", 8788))
HOST = os.environ.get("HOST", "127.0.0.1")

# Cap the visual tokens per screenshot. Attention memory grows ~O(n^2) in the
# token count and screenshots are large, so an uncapped image is the main cause
# of CUDA OOM on a 16 GB GPU. IMPORTANT: use the SAME value in the benchmark
# notebook, or the accuracy you report is not the accuracy you deployed.
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1024 * 28 * 28

# Longer than the original 128: the schema now carries a fourth field, and a
# truncated generation yields invalid JSON that falls silently into the
# "unclear" branch -- which looks exactly like a model failure.
MAX_NEW_TOKENS = 192

# Serialize generate() so a burst of screenshots never runs two forward passes
# concurrently (which would multiply peak VRAM).
_GEN_LOCK = threading.Lock()

# Categories that can constitute cheating. exam_page is excluded: an official
# exam-platform page is never cheating regardless of relevance.
CHEAT_CATEGORIES = {"ai_assistant", "search_engine",
                    "reference_material", "other_application"}

# Anchor questions per language, read off the actual exam pages used in the
# evaluation. Switch with: POST /config {"lang": "ko"}
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

if os.environ.get("REQUIRE_CUDA") == "1":
    assert torch.cuda.is_available(), (
        "REQUIRE_CUDA=1 but no GPU is visible. On Kaggle: "
        "Settings > Accelerator > GPU T4 x2, then restart the session."
    )

print("Loading", MODEL_ID, "...")
if torch.cuda.is_available():
    # float16, not bfloat16: T4 is Turing (sm75) and has no native bf16.
    # sdpa, not flash_attention_2: FA2 does not support Turing either.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto",
        attn_implementation="sdpa")
    DEVICE = "cuda"
    print("GPU:", torch.cuda.get_device_name(0))
else:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32)
    model.to("cpu")
    DEVICE = "cpu"
    print("WARNING: no CUDA -> running on CPU (~30-60 s/image).")

processor = AutoProcessor.from_pretrained(
    MODEL_ID, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
print("VLM ready on", DEVICE, "| max visual tokens/img ~", MAX_PIXELS // (28 * 28))
print("Exam context:", LANG)

# NOTE: this template contains literal { } for the JSON schema, so it must be
# filled with .replace(), never .format() / f-string.
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


def build_prompt():
    # .replace(), not .format() -- the template contains literal JSON braces.
    return PROMPT_TEMPLATE.replace("{EXAM_CONTEXT}", EXAM_CONTEXT)


def dataurl_to_pil(s):
    if isinstance(s, str) and s.strip().startswith("data:") and "," in s:
        s = s.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB")


def parse_json(raw):
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
                return d
        except Exception:
            continue
    return {"category": "unclear", "identity": "unknown",
            "exam_relevance": "none", "summary": raw[:160]}


def decide(out):
    """Apply the cheating rule. Returns (is_cheating, warning_or_None).

    A missing exam_relevance is treated as "none" so the service keeps serving,
    but it is surfaced in the response and logged -- silently defaulting to
    "none" makes the system flag nothing at all, which is indistinguishable
    from a well-behaved student.
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


def analyze(img):
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": build_prompt()}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs, vids = process_vision_info(msgs)
    inp = processor(text=[text], images=imgs, videos=vids,
                    padding=True, return_tensors="pt").to(model.device)

    t0 = time.perf_counter()
    with _GEN_LOCK:
        try:
            with torch.inference_mode():
                out = model.generate(**inp, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            trimmed = [o[len(i):] for i, o in zip(inp.input_ids, out)]
            raw = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
            result = parse_json(raw)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return {"category": "error", "identity": "unknown", "exam_relevance": "none",
                    "is_cheating": False, "latency_ms": None,
                    "summary": "GPU out of memory; lower MAX_PIXELS or the capture resolution"}
        except Exception as e:
            return {"category": "error", "identity": "unknown", "exam_relevance": "none",
                    "is_cheating": False, "latency_ms": None,
                    "summary": "inference failed: %s: %s" % (type(e).__name__, e)}
        finally:
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    latency_ms = round((time.perf_counter() - t0) * 1000)
    result.setdefault("identity", "unknown")
    result.setdefault("summary", "")
    is_cheating, warning = decide(result)
    result["is_cheating"] = is_cheating
    result["latency_ms"] = latency_ms
    if warning:
        result["warning"] = warning

    print("[%s] %-18s rel=%-8s cheat=%-5s %4dms  %s%s" % (
        datetime.now().strftime("%H:%M:%S"),
        result.get("category"), result.get("exam_relevance"), is_cheating,
        latency_ms, result.get("identity"),
        ("  !! " + warning) if warning else ""), flush=True)
    return result


app = Flask(__name__)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "device": DEVICE, "model": MODEL_ID,
                    "lang": LANG, "exam_context": EXAM_CONTEXT})


@app.route("/config", methods=["GET", "POST"])
def config():
    global EXAM_CONTEXT, LANG
    if request.method == "POST":
        data = request.get_json(force=True, silent=True) or {}
        lang = data.get("lang")
        ctx = data.get("exam_context")
        if lang:
            if lang not in EXAM_CONTEXTS:
                return jsonify({"error": "unknown lang %r; known: %s"
                                % (lang, sorted(EXAM_CONTEXTS))}), 400
            LANG, EXAM_CONTEXT = lang, EXAM_CONTEXTS[lang]
        elif isinstance(ctx, str) and ctx.strip():
            LANG, EXAM_CONTEXT = "custom", ctx.strip()
        else:
            return jsonify({"error": "send {\"lang\":\"en\"} or {\"exam_context\":\"...\"}"}), 400
        print("[config] exam context -> %s (%d chars)" % (LANG, len(EXAM_CONTEXT)), flush=True)
    return jsonify({"lang": LANG, "exam_context": EXAM_CONTEXT})


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


if __name__ == "__main__":
    print("Serving on http://%s:%d" % (HOST, PORT))
    app.run(host=HOST, port=PORT)
