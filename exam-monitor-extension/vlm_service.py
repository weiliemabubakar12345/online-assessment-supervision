"""Local VLM analysis service for the exam monitor (Qwen2.5-VL-3B).

The Node proctor server (server.js) POSTs each screenshot here; this service runs
the VLM to identify what is on screen and returns a JSON conclusion that the
dashboard displays.

Endpoints:
  GET  /health                          -> {status, device, model}
  POST /analyze  { "image": "<dataURL|base64>" }
       -> { "category", "identity", "summary" }

Setup:
  pip install flask torch transformers qwen-vl-utils accelerate pillow
Run:
  python vlm_service.py            # serves http://localhost:8788

Note: on a machine WITHOUT CUDA (e.g. Intel Arc) this falls back to CPU and is
slow (~30-60 s/image). For a smooth live demo, run it on a CUDA GPU machine and
point server.js's VLM_URL at it.
"""
import base64
import io
import json
import os
import re
import threading

# Reduce fragmentation-driven OOM (must be set before CUDA context init).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from flask import Flask, request, jsonify
from PIL import Image
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
PORT = 8788

# Cap the number of visual tokens per screenshot. Attention memory grows ~O(n^2)
# in the token count, and screenshots are large, so an uncapped image is the main
# cause of CUDA OOM on a 16 GB GPU. 1024 * 28 * 28 keeps each image <= ~1024
# visual tokens while staying legible for UI text. IMPORTANT: use the SAME value
# in the benchmark notebook so reported accuracy matches deployment behaviour.
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1024 * 28 * 28

# Serialize generate() so a burst of screenshots never runs two forward passes
# concurrently (which would multiply peak VRAM). Cheap insurance even though the
# Flask dev server is single-threaded by default.
_GEN_LOCK = threading.Lock()

print("Loading", MODEL_ID, "...")
if torch.cuda.is_available():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto")
    DEVICE = "cuda"
else:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32)
    model.to("cpu")
    DEVICE = "cpu"
    print("WARNING: no CUDA -> running on CPU (slow). Fine for a small demo.")
processor = AutoProcessor.from_pretrained(
    MODEL_ID, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
print("VLM ready on", DEVICE, "| max visual tokens/img ~", MAX_PIXELS // (28 * 28))

PROMPT = """You are a screen-analysis component in an online-exam monitoring system.
Given ONE screenshot of a student's screen, identify what is open and classify the
PRIMARY content into exactly ONE category:
- exam_page: the online exam or quiz itself
- ai_assistant: an AI chatbot (ChatGPT, Claude, Gemini, Copilot, Perplexity, Doubao, Kimi, ERNIE/Wenxin, DeepSeek, etc.)
- search_engine: a search engine or its results page (Google, Bing, DuckDuckGo, Baidu, Yahoo! Japan)
- reference_material: encyclopedia, docs, tutorial, PDF, or Q&A site (Wikipedia, Baidu Baike, MDN, StackOverflow, Zhihu)
- other_application: any other website or app (news, video, shopping, maps, social, messaging like WeChat/LINE, music, etc.)

The screen may be in ANY language (including Chinese, Japanese, Korean). Classify by
meaning and visual layout, NOT by language. Always write "summary" in English.

Output ONLY valid JSON (no markdown):
{"category":"<one label above>","identity":"<site/app name if legible else 'unknown'>","summary":"<one English sentence describing what is shown>"}"""


def dataurl_to_pil(s):
    if isinstance(s, str) and s.strip().startswith("data:") and "," in s:
        s = s.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB")


def parse_json(raw):
    s = re.sub(r"```$", "", re.sub(r"^```(?:json)?", "", raw.strip())).strip()
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {"category": "unclear", "identity": "unknown", "summary": raw[:160]}


def analyze(img):
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": PROMPT}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs, vids = process_vision_info(msgs)
    inp = processor(text=[text], images=imgs, videos=vids,
                    padding=True, return_tensors="pt").to(model.device)
    with _GEN_LOCK:  # one forward pass at a time -> bounded peak VRAM
        try:
            with torch.inference_mode():
                out = model.generate(**inp, max_new_tokens=128, do_sample=False)
            trimmed = [o[len(i):] for i, o in zip(inp.input_ids, out)]
            raw = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
            return parse_json(raw)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return {"category": "error", "identity": "unknown",
                    "summary": "GPU out of memory; lower MAX_PIXELS or the capture resolution"}
        finally:
            if DEVICE == "cuda":
                torch.cuda.empty_cache()  # release cached blocks between requests


app = Flask(__name__)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "device": DEVICE, "model": MODEL_ID})


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
    print("Serving on http://localhost:" + str(PORT))
    app.run(host="127.0.0.1", port=PORT)
