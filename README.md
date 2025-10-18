# mars-ddddocr
mars-ddddocr

## 部署
### docker (推荐)
```shell
docker run -d -p 7777:7777 --restart=always --name mars-ddddocr xhzyz/mars-ddddocr:latest
```
### 本地
```shell
git clone https://github.com/xhzyz/mars-ddddocr.git
pip install -r requirements.txt
python server.py
```

## 测试
```shell
curl -s http://127.0.0.1:7777/healthz
```

## 接口
```shell
http://ip:7777/solve
```
