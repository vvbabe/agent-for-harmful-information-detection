# 信息安全实验数据获取模块（协作详版）

本项目提供“网络威胁流量 / 有害信息识别”实验的数据获取与预处理流水线：
- 开源数据集下载与统一转换（CSV/JSON/JSONL/ZIP/TAR/PCAP → JSONL）
- 社交媒体公开帖子抓取（微博/小红书）+ 文本清洗
- 统一日志与单元测试，便于协作扩展

---

## 目录结构
- `configs/datasets.yaml` 数据集清单（名称 → 下载链接列表）
- `datasets_downloader.py` 多线程下载、断点续传、解压与统一 JSONL 转换
- `crawler_weibo.py` 微博公开搜索抓取（requests 或可选 selenium）
- `crawler_xiaohongshu.py` 小红书公开搜索抓取（requests 或可选 selenium）
- `preprocess.py` 文本清洗（去 HTML/emoji/冗余空格），CLI 可用
- `utils/logger.py` 统一日志（控制台 + `logs/security_data.log`）
- `tests/` 关键逻辑 pytest 用例
- `notebooks/example_keyword_stats.ipynb` JSONL 关键词统计演示
- `datasets/` 下载与输出目录（按数据集名分子目录）

## 环境安装
- Python 3.10+
- 推荐使用虚拟环境：
```bash
python -m venv .venv
source .venv/bin/activate  # Windows 用 .venv\Scripts\activate
pip install -r requirements.txt
```

## 配置数据集（`configs/datasets.yaml`）
示例：
```yaml
HateSpeech-Davidson:
  urls:
    - https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv
  type: csv

SpamAssassin-Public:
  urls:
    - https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2
    - https://spamassassin.apache.org/old/publiccorpus/20030228_spam.tar.bz2
  type: tar

# 支持本地文件，便于离线验证
# demo:
#   urls:
#     - file:///ABSOLUTE/PATH/TO/demo.csv
#   type: csv
```
注意：
- Kaggle/部分官网页仅提供说明页，请替换为直链或使用本地 file://
- PCAP→JSONL 目前为占位（记录文件路径），如需真实特征请扩展转换逻辑

## 使用方法
### 1) 下载与转换为 JSONL
```bash
python datasets_downloader.py --config configs/datasets.yaml --workers 4
# 仅处理指定数据集
python datasets_downloader.py --config configs/datasets.yaml --dataset HateSpeech-Davidson
```
输出：`datasets/<dataset_name>/<dataset_name>.jsonl`

JSONL 示例（文本类）：
```json
{"id":"1","text":"example","label":0}
```

### 2) 爬取公开社交媒体
两种驱动：
- 默认 `requests`：轻量，可能受动态渲染/反爬限制
- 可选 `selenium`：undetected_chromedriver，适合动态站点

微博：
```bash
# requests
python crawler_weibo.py --keyword "诈骗" --pages 5 --output weibo_data.jsonl --delay 3 --append
# selenium（若需：--chrome-binary 指定 Chrome 路径；--proxy 设置代理）
python crawler_weibo.py --keyword "诈骗" --pages 5 --output weibo_data.jsonl --driver selenium --delay 3 --append --chrome-binary "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --proxy http://127.0.0.1:7890
```
小红书：
```bash
python crawler_xiaohongshu.py --keyword "辱骂" --pages 3 --output xiaohongshu_data.jsonl --driver selenium --delay 3 --append
```
可选参数：
- `--driver {requests|selenium}` 默认 requests
- `--pages N` 页数
- `--delay SECONDS` 页间延迟（建议 2~5s+）
- `--append` 追加写入（长跑/断点续抓）
- `--proxy`（微博支持）HTTP/HTTPS 代理
- `--chrome-binary`（微博支持）指定 Chrome/Chromium 二进制

Selenium 依赖：
```bash
pip install selenium undetected-chromedriver
```
首次运行会自动匹配 chromedriver；macOS 默认无头模式。

### 3) 文本清洗
```bash
python preprocess.py --input weibo_data.jsonl --output weibo_data.clean.jsonl --text-field text
```

## 批量抓取建议
- 合规前提下分批（按关键词/时间窗口）+ `--append` 合并
- 增加延迟与随机抖动，避免过速访问
- 监控 `logs/security_data.log`，关注错误与重试
- 可配置代理提升稳定性

## 合规与隐私
- 仅抓取公开可见内容，遵循 robots.txt 与平台协议
- 不采集敏感个人信息，仅保存匿名/哈希化 ID
- 控制频率，避免对站点造成负担

## 测试
```bash
pytest -q
```

## 故障排查
- 返回 0 条：可能结构变更/反爬，尝试 selenium、代理与更大延迟
- selenium 失败：提供 `--chrome-binary` 或安装本地 Chrome；检查 undetected-chromedriver 版本
- 无法直链：先手动下载到本地，用 file:// 引入
- 编码异常：使用清洗器或先转码后再处理