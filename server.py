# server.py  — 双模型并发识别（上半算术 + 下半4位码），高并发优化版
import os
import time
import base64
import asyncio
from io import BytesIO
from typing import Optional, Tuple, Dict
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field
from PIL import Image
import ddddocr
from pathlib import Path

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/models"))

TOP_ONNX  = os.getenv("TOP_ONNX",  str(MODEL_DIR / "top/model.onnx"))
TOP_CHRS  = os.getenv("TOP_CHRS",  str(MODEL_DIR / "top/charsets.json"))
BOT_ONNX  = os.getenv("BOT_ONNX",  str(MODEL_DIR / "bot/model.onnx"))
BOT_CHRS  = os.getenv("BOT_CHRS",  str(MODEL_DIR / "bot/charsets.json"))


DEFAULT_TOP_RATIO = float(os.getenv("TOP_RATIO", "0.5"))         # 上半高度占比
MAX_CONCURRENCY   = int(os.getenv("MAX_CONCURRENCY", "32"))      # 并发限流
HTTP_TIMEOUT      = float(os.getenv("HTTP_TIMEOUT", "6.0"))

HTTP_LIMITS = httpx.Limits(  # 连接池上限（高并发更稳）
    max_keepalive_connections=int(os.getenv("MAX_KEEPALIVE", "64")),
    max_connections=int(os.getenv("MAX_CONNECTIONS", "128")),
)

# ========= FastAPI 应用（lifespan 更规范） =========
state: Dict[str, object] = {
    "client": None,
    "ocr_top": None,
    "ocr_bot": None,
    "sem": asyncio.Semaphore(MAX_CONCURRENCY),
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1) 预加载两个 OCR 模型
    state["ocr_top"] = ddddocr.DdddOcr(
        det=False, ocr=False, show_ad=False,
        import_onnx_path=TOP_ONNX, charsets_path=TOP_CHRS
    )
    state["ocr_bot"] = ddddocr.DdddOcr(
        det=False, ocr=False, show_ad=False,
        import_onnx_path=BOT_ONNX, charsets_path=BOT_CHRS
    )
    # 2) 预热（避免首个请求慢）
    for ocr in (state["ocr_top"], state["ocr_bot"]):
        _ = ocr.classification(Image.new("RGB", (6, 6)))  # 传PIL也可，内部会转换

    # 3) 异步 HTTP 客户端（如果传 URL 会用它抓图）
    state["client"] = httpx.AsyncClient(
        http2=True, timeout=HTTP_TIMEOUT, limits=HTTP_LIMITS
    )
    try:
        yield
    finally:
        if state["client"]:
            await state["client"].aclose()

app = FastAPI(default_response_class=ORJSONResponse, lifespan=lifespan)

# ========= 工具函数 =========
def _b64_or_url_to_bytes(s: str) -> Tuple[Optional[str], Optional[bytes]]:
    """如果是URL返回(url, None)；如果是base64返回(None, bytes)。否则抛错。"""
    s = (s or "").strip()
    if s.startswith(("http://", "https://")):
        return s, None
    if "base64," in s:
        s = s.split("base64,")[-1]
    pad = "=" * (-len(s) % 4)  # 补齐padding
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

async def _ocr_bytes(ocr: ddddocr.DdddOcr, data: bytes) -> str:
    # ddddocr.classification 是同步的；用线程池避免阻塞事件循环（高并发更稳）
    return await asyncio.to_thread(ocr.classification, data)

def _to_png_bytes(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# ========= 请求/响应模型 =========
class SolveBody(BaseModel):
    image: str = Field(..., description="完整验证码（base64 或 URL）")
    top_ratio: float = Field(DEFAULT_TOP_RATIO, ge=0.2, le=0.8, description="上半高度占比")

class Health(BaseModel):
    ok: bool

# ========= 路由 =========
@app.get("/healthz", response_model=Health)
async def healthz():
    return {"ok": True}

@app.post("/solve")
async def solve(body: SolveBody):
    t0 = time.perf_counter()
    url, raw = _b64_or_url_to_bytes(body.image)
    img_bytes = await _fetch_if_url(url, raw)

    # 解码 + 拆图
    try:
        full = Image.open(BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot decode image: {e}")
    top_img, bot_img, meta = _split_top_bottom(full, body.top_ratio)

    # 转 Bytes（PNG）
    tb = _to_png_bytes(top_img)
    bb = _to_png_bytes(bot_img)

    # 并发识别
    async with state["sem"]:
        top_task = asyncio.create_task(_ocr_bytes(state["ocr_top"], tb))
        bot_task = asyncio.create_task(_ocr_bytes(state["ocr_bot"], bb))
        top_text, bot_text = await asyncio.gather(top_task, bot_task)

    top_text = (top_text or "").strip()
    try:
        answer = int(top_text)   # 算术模型已直接输出结果数字（可带负号）
    except Exception:
        answer = None

    return {
        "answer": answer,
        "answer_text": top_text,
        "code4": (bot_text or "").strip(),
        "boxes": meta,
        "timings_sec": {"total": round(time.perf_counter() - t0, 4)},
    }

# 运行： uvicorn server:app --host 0.0.0.0 --port 7777  （生产建议多worker，如：--workers 2）
