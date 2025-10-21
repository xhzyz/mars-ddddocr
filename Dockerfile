# ==== base ====
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 常用运行库
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 安装依赖
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY server.py /app/server.py

# 复制模型并“重命名”为固定名
# 顶部
COPY models/top/calc_top_0.99609375_266_4000_2025-10-18-20-31-19.onnx /app/models/top/model.onnx
COPY models/top/charsets.json                                         /app/models/top/charsets.json
# 底部
COPY models/bot/bot90.onnx         /app/models/bot/model.onnx
COPY models/bot/charsets.json                                         /app/models/bot/charsets.json

EXPOSE 7777
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7777", "--workers", "2"]
