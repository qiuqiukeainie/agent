# D 数据与系统工程升级说明

本轮已经把 D 的最小底座从“文件 + JSON”升级为“SQLite 元数据 + 可替换向量索引 + 搜索日志 + 评测脚本”。

## 已实现能力

- SQLite 元数据表：`data/assets.db`
- 搜索日志表：记录 query、Agent plan、Top results、索引后端、最高分
- SHA-256 重复文件检测
- 素材统计接口：`GET /api/stats`
- 重复文件接口：`GET /api/duplicates`
- 搜索日志接口：`GET /api/search-logs`
- 从 `uploads` 重建索引：`POST /api/rebuild-index`
- 同步元数据：`POST /api/reindex-metadata`
- 可选 Faiss 向量索引层：
  - 已安装 `faiss` 时使用 `IndexFlatIP`
  - 未安装时自动回退到 NumPy

## 当前素材库状态

- 图片素材：526
- SQLite 元数据：526
- 重复文件：21
- 当前索引后端：`numpy`
- 当前模型：`clip-vit-base-patch32`

## 检索评测

评测脚本：

```powershell
.\.venv\Scripts\python.exe tools\evaluate_coco_retrieval.py --limit 100 --top-k 10 --out data\eval\coco_retrieval_report_100.json
```

100 条 COCO caption 评测结果：

| 指标 | 结果 |
| --- | --- |
| Recall@1 | 0.63 |
| Recall@5 | 0.94 |
| Recall@10 | 1.00 |
| MRR@10 | 0.7599 |

报告文件：

```text
data/eval/coco_retrieval_report_100.json
```

## 为什么这一步重要

之前系统的问题是：图片文件存在，但 JSON 索引和页面显示不同步。

现在 D 层负责把三件事统一起来：

- 磁盘文件：`data/uploads`
- 元数据：`data/assets.db`
- 向量索引：`data/index.json` + 内存向量索引

后续 B/C 模块产生的人脸、场景、OCR、ASR 标签，都可以进入 SQLite，再参与检索重排。

## 下一步

- 在支持 Faiss 的 Python 环境中安装 `faiss-cpu`，让后端切到 `faiss-flatip`。
- 增加 `tags`、`persons`、`asset_tags` 表。
- 增加去重确认接口和 Web 页面。
- 增加评测查询管理表，保存多轮实验结果。
