# Asset Tags API

本文档用于 A/B/C/D/E 成员对齐“标签写入”接口。A 负责统一 HTTP 接口和落库，B/C/D 优先交付可被 Python 调用的模块，不建议各自启动独立端口。

## 1. 接口用途

`POST /api/asset-tags` 用于把视觉模型、场景理解、人脸聚类、人工修正等模块生成的标签写入统一素材库。

标签写入后会进入 SQLite 的 `tags` 和 `asset_tags` 表，并参与后续语义检索重排。

## 2. 请求地址

```http
POST http://127.0.0.1:8000/api/asset-tags
Content-Type: application/json
```

## 3. 最小请求格式

```json
{
  "asset_id": "000000012667",
  "tags": ["person", "building", "outdoor"],
  "source": "vlm_blip",
  "confidence": 0.91
}
```

字段说明：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `asset_id` | string | 是 | 素材唯一 ID，通常等于文件名去掉扩展名 |
| `tags` | array | 是 | 标签列表，支持字符串或对象 |
| `source` | string | 否 | 标签来源，默认 `external` |
| `confidence` | number | 否 | 置信度，范围 0 到 1，默认 1.0 |

## 4. 结构化标签请求

B 的视觉模块推荐输出这种格式：

```json
{
  "asset_id": "000000012667",
  "tags": [
    {"name": "person", "confidence": 0.91, "source": "vlm_blip"},
    {"name": "building", "confidence": 0.88, "source": "vlm_blip"},
    {"name": "group_photo", "confidence": 0.76, "source": "face_cluster"}
  ]
}
```

接口会按照 `source + confidence` 自动分组写入。

## 5. 成功响应

```json
{
  "ok": true,
  "results": [
    {
      "asset_id": "000000012667",
      "source": "vlm_blip",
      "confidence": 0.91,
      "tags": ["building", "person"],
      "written_count": 2
    }
  ]
}
```

## 6. 调试命令

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/asset-tags" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"asset_id":"000000012667","tags":[{"name":"person","confidence":0.91,"source":"vlm_blip"},{"name":"building","confidence":0.88,"source":"vlm_blip"}]}'
```

如果不确定 `asset_id`，先打开：

```text
http://127.0.0.1:8000/api/assets
```

## 7. 团队边界约定

- A 维护统一 HTTP API、Agent 检索链路、融合重排和接口文档。
- B 的视觉算法、人脸识别、场景理解应优先做成 Python 模块，例如 `extract_tags(image_path) -> dict`。
- C 的 OCR/ASR 也优先做成 Python 模块，例如 `extract_ocr(asset_path) -> dict`。
- D 负责 Pipeline 时，可以调用 B/C 的 Python 模块，再统一写入 A 提供的 HTTP API 或 MetadataStore。
- E 的 Web 页面只调用 HTTP API，不直接读取 Python 内部变量。
- 除非团队明确需要微服务架构，否则 B/C 不单独开端口，避免端口冲突、启动顺序混乱和接口版本失控。

## 8. 命名规则

- 标签名统一小写英文或稳定英文短语：`person`、`building`、`group_photo`、`red_object`。
- 多词标签使用下划线，不使用空格：`traffic_light`，不要写 `traffic light`。
- `source` 必须能定位来源模块：`vlm_blip`、`insightface`、`face_cluster`、`paddleocr`、`whisper`、`manual`。
- `confidence` 统一为 0 到 1 的浮点数。
- 不确定标签可以先写入，但置信度应低于 0.6。
