# 检索分数与精度调优记录

## 为什么原始分数只有 20% 左右

页面之前显示的“20%”来自 CLIP 向量余弦相似度，例如 `0.22` 被直接乘以 100。

这个值不是概率，也不是“只有 20% 相关”。CLIP 的图文向量经过归一化后，相关图片的余弦相似度常见就在 `0.20 - 0.35` 区间。真正有意义的是它在候选图片中的相对排序。

因此现在页面分成两类分数：

- 排序置信：给用户看的相对分数，体现当前结果在本次检索中的强弱。
- 原始 CLIP：给开发和报告看的技术分数，保留真实余弦相似度。

## 本轮精度优化

新增了三类辅助信号：

- 标签重排：COCO category 标签导入 SQLite，命中 dog/person/bus/food 等标签时加分。
- OCR 预留：新增 OCR 文本表和写入接口，后续字幕/文字图片可参与检索。
- 分数校准：新增 `display_score`，避免把 CLIP 原始相似度误读为概率。

## 标签导入

```powershell
.\.venv\Scripts\python.exe tools\import_coco_tags.py --data-dir data --instances data\datasets\coco\annotations\instances_val2017.json
```

当前导入结果：

- 有标签图片：496
- 标签关联：1485
- 标签种类：78

## 评测对比

评测命令：

```powershell
.\.venv\Scripts\python.exe tools\evaluate_coco_retrieval.py --limit 100 --top-k 10 --out data\eval\coco_retrieval_report_100_tagged.json
```

| 版本 | Recall@1 | Recall@5 | Recall@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: |
| CLIP + prompt | 0.63 | 0.94 | 1.00 | 0.7599 |
| CLIP + prompt + tag rerank | 0.68 | 0.96 | 1.00 | 0.7927 |

## OCR 接口

后续 C 模块可以把 OCR 结果写入：

```http
POST /api/ocr-text
Content-Type: application/json
```

```json
{
  "asset_id": "000000012667",
  "text": "SALE 50% OFF",
  "engine": "paddleocr",
  "confidence": 0.93
}
```

也可以用 CSV 脚本批量导入：

```powershell
.\.venv\Scripts\python.exe tools\import_ocr_text.py --csv data\ocr_rows.csv --engine paddleocr
```

CSV 字段：

```text
asset_id,text,confidence
000000012667,SALE 50% OFF,0.93
```
