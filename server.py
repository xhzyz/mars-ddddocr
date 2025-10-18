# server.py — 双模型并发识别（上半算术 + 下半4位码）高并发优化版（CPU多核池）
import os
import time
import base64
import asyncio
import re
from io import BytesIO
from typing import Optional, Tuple, Dict, List
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field
from PIL import Image
import ddddocr

# ========== 环境配置 ==========
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/models"))
TOP_ONNX  = os.getenv("TOP_ONNX",  str(MODEL_DIR / "top/model.onnx"))
TOP_CHRS  = os.getenv("TOP_CHRS",  str(MODEL_DIR / "top/charsets.json"))
BOT_ONNX  = os.getenv("BOT_ONNX",  str(MODEL_DIR / "bot/model.onnx"))
BOT_CHRS  = os.getenv("BOT_CHRS",  str(MODEL_DIR / "bot/charsets.json"))

DEFAULT_TOP_RATIO = float(os.getenv("TOP_RATIO", "0.5"))
HTTP_TIMEOUT      = float(os.getenv("HTTP_TIMEOUT", "6.0"))
MAX_CONCURRENCY   = int(os.getenv("MAX_CONCURRENCY", "64"))   # 全局限流
OCR_POOL_SIZE     = int(os.getenv("OCR_POOL_SIZE", os.cpu_count() or 4))  # 自动按核数

HTTP_LIMITS = httpx.Limits(
    max_keepalive_connections=64,
    max_connections=128,
)

# ========== 全局状态 ==========
state: Dict[str, object] = {
    "client": None,
    "ocr_top_pool": [],
    "ocr_bot_pool": [],
    "sem": asyncio.Semaphore(MAX_CONCURRENCY),
}

# ========== 启动生命周期 ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"🔥 初始化 OCR 实例池，共 {OCR_POOL_SIZE} 组")
    for _ in range(OCR_POOL_SIZE):
        state["ocr_top_pool"].append(ddddocr.DdddOcr(det=False, ocr=False, show_ad=False,
                                                     import_onnx_path=TOP_ONNX, charsets_path=TOP_CHRS))
        state["ocr_bot_pool"].append(ddddocr.DdddOcr(det=False, ocr=False, show_ad=False,
                                                     import_onnx_path=BOT_ONNX, charsets_path=BOT_CHRS))

    # 预热一次，避免首次推理慢
    for ocr in state["ocr_top_pool"] + state["ocr_bot_pool"]:
        _ = ocr.classification(Image.new("RGB", (6, 6)))

    state["client"] = httpx.AsyncClient(http2=True, timeout=HTTP_TIMEOUT, limits=HTTP_LIMITS)
    try:
        yield
    finally:
        if state["client"]:
            await state["client"].aclose()

app = FastAPI(default_response_class=ORJSONResponse, lifespan=lifespan)

# ========== 工具函数 ==========
def _b64_or_url_to_bytes(s: str) -> Tuple[Optional[str], Optional[bytes]]:
    s = (s or "").strip()
    if s.startswith(("http://", "https://")):
        return s, None
    if "base64," in s:
        s = s.split("base64,")[-1]
    pad = "=" * (-len(s) % 4)
    try:
        return None, base64.b64decode(s + pad)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image base64: {e}")

async def _fetch_if_url(url: Optional[str], raw: Optional[bytes]) -> bytes:
    if raw is not None:
        return raw
    assert url and state["client"], "HTTP client not ready"
    r = await state["client"].get(url)
    r.raise_for_status()
    return r.content

def _split_top_bottom(img: Image.Image, top_ratio: float):
    w, h = img.size
    cut = max(1, min(h - 1, int(h * top_ratio)))
    top = img.crop((0, 0, w, cut))
    bot = img.crop((0, cut, w, h))
    meta = {"w": w, "h": h, "cut": cut, "top_box": [0, 0, w, cut], "bottom_box": [0, cut, w, h]}
    return top, bot, meta

def _to_png_bytes(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# ========== OCR 线程池封装 ==========
async def _ocr_bytes(is_top: bool, data: bytes) -> str:
    async with state["sem"]:
        pool = state["ocr_top_pool"] if is_top else state["ocr_bot_pool"]
        ocr = pool[hash(asyncio.current_task()) % len(pool)]
        return await asyncio.to_thread(ocr.classification, data)

# ========== 数据模型 ==========
class SolveBody(BaseModel):
    image: str
    top_ratio: float = Field(DEFAULT_TOP_RATIO, ge=0.2, le=0.8)

class Health(BaseModel):
    ok: bool

# ========== 路由 ==========
@app.get("/healthz", response_model=Health)
async def healthz():
    return {"ok": True}

@app.post("/solve")
async def solve(body: SolveBody):
    t0 = time.perf_counter()
    url, raw = _b64_or_url_to_bytes(body.image)
    img_bytes = await _fetch_if_url(url, raw)

    try:
        full = Image.open(BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot decode image: {e}")

    top_img, bot_img, meta = _split_top_bottom(full, body.top_ratio)
    tb, bb = _to_png_bytes(top_img), _to_png_bytes(bot_img)

    # ⚡ 真正并发 OCR
    top_task = asyncio.create_task(_ocr_bytes(True, tb))
    bot_task = asyncio.create_task(_ocr_bytes(False, bb))
    top_text, bot_text = await asyncio.gather(top_task, bot_task)

    top_text = (top_text or "").strip()
    try:
        expr = re.sub(r"[^0-9+\-]", "", top_text)
        if re.fullmatch(r"-?\d+([+\-]-?\d+)?", expr):
            answer = eval(expr)
            if not (-99 <= answer <= 99):
                answer = None
        else:
            answer = None
    except Exception:
        answer = None

    return {
        "answer": answer,
        "answer_text": top_text,
        "code4": (bot_text or "").strip(),
        "boxes": meta,
        "timings_sec": {"total": round(time.perf_counter() - t0, 4)},
    }

# 启动命令（推荐4-8核服务器）：
#   uvicorn server:app --host 0.0.0.0 --port 7777 --workers 2
