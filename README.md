# 多模态素材语义管理系统

本项目当前完成了一个可演示的多模态素材语义检索与管理原型。

## 当前能力

- 图片素材上传、导入、浏览
- CLIP 文本-图像跨模态检索
- Agent 查询拆解与多 prompt 调度
- SQLite 元数据管理
- 可选 Faiss / NumPy 向量召回
- COCO 标签导入与标签辅助重排
- OCR 文本落库接口预留
- 视频上传、抽帧、CLIP 平均向量检索
- 搜索日志与 COCO Recall@K 评测

## 启动

```powershell
cd F:\agent
.\.venv\Scripts\python.exe app.py
```

打开：

```text
http://127.0.0.1:8000
```

如果页面没有刷新出最新内容，使用 `Ctrl + F5` 强制刷新。

## 当前基线

```text
素材数：526
视频样例：2
模型：CLIP ViT-B/32
索引后端：NumPy fallback
标签关联：1485
OCR 文本：0
Recall@1：0.68
Recall@5：0.96
Recall@10：1.00
MRR@10：0.7927
```

## 核心接口

- `GET /api/status`
- `GET /api/assets`
- `GET /api/stats`
- `GET /api/search?q=...`
- `GET /api/search-logs`
- `GET /api/duplicates`
- `POST /api/ocr-text`
- `POST /api/rebuild-index`
- `POST /api/reindex-metadata`

## 团队协作规范

详细接口、命名、数据库、角色分工、调试流程、提交规范见：

[docs/team-interface-spec.md](docs/team-interface-spec.md)

这是组员之间协作和联调的统一基准。

## 常用脚本

导入 COCO 图片子集：

```powershell
.\.venv\Scripts\python.exe tools\import_coco_subset.py --images "F:\agent\data\datasets\coco\val2017" --limit 500 --data-dir "F:\agent\data"
```

导入 COCO 标签：

```powershell
.\.venv\Scripts\python.exe tools\import_coco_tags.py --data-dir data --instances data\datasets\coco\annotations\instances_val2017.json
```

运行检索评测：

```powershell
.\.venv\Scripts\python.exe tools\evaluate_coco_retrieval.py --limit 100 --top-k 10
```

批量导入 OCR 文本：

```powershell
.\.venv\Scripts\python.exe tools\import_ocr_text.py --csv data\ocr_rows.csv --engine paddleocr
```

生成并导入本地视频样例：

```powershell
.\.venv\Scripts\python.exe tools\create_video_samples.py
.\.venv\Scripts\python.exe tools\import_video_dataset.py --videos data\video_samples --limit 10 --data-dir data
```

视频与中文检索升级说明：

[docs/video-and-chinese-upgrade.md](docs/video-and-chinese-upgrade.md)
