import json
from preprocess import clean_text, strip_html, strip_emoji, normalize_ws, process_jsonl

def test_clean_text_basic():
    raw = "<p>你好😀</p>  这 是\n测试"
    out = clean_text(raw)
    assert "<" not in out and "]" not in out
    assert "😀" not in out
    assert "\n" not in out
    assert out.startswith("你好") and "测试" in out


def test_process_jsonl(tmp_path):
    src = tmp_path / "in.jsonl"
    dst = tmp_path / "out.jsonl"
    with open(src, "w", encoding="utf-8") as f:
        f.write(json.dumps({"text": "<b>Hi😀</b>\n"}) + "\n")
        f.write(json.dumps({"text": "  多  空  格  ", "x": 1}) + "\n")
    n = process_jsonl(str(src), str(dst))
    assert n == 2
    lines = [json.loads(l) for l in open(dst, encoding="utf-8")]
    assert lines[0]["text"] == "Hi"
    assert lines[1]["text"] == "多 空 格"
