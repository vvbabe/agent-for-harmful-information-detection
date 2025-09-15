import json
import os
from pathlib import Path

from datasets_downloader import convert_to_jsonl, normalize_record


def test_normalize_record():
    obj = {"ID": 5, "comment_text": "hello", "target": 1}
    out = normalize_record(obj, 0)
    assert out["id"] == "5"
    assert out["text"] == "hello"
    assert out["label"] == 1


def test_convert_csv_jsonl(tmp_path):
    csv_path = tmp_path / "a.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("id,text,label\n1,hi,0\n2,hey,1\n")
    out = tmp_path / "out.jsonl"
    n = convert_to_jsonl([csv_path], out)
    assert n == 2
    lines = [json.loads(l) for l in open(out, encoding="utf-8")]
    assert lines[0] == {"id": "1", "text": "hi", "label": "0"}


def test_convert_json_jsonl(tmp_path):
    json_path = tmp_path / "b.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([{"id": 1, "text": "x"}, {"id": 2, "text": "y"}], f)
    out = tmp_path / "out2.jsonl"
    n = convert_to_jsonl([json_path], out)
    assert n == 2
    lines = [json.loads(l) for l in open(out, encoding="utf-8")]
    assert lines[1]["text"] == "y"
