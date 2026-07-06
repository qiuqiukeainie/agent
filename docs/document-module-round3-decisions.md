# B 文档模块第三轮边界情况与收尾细节答复

本文档用于回答 B 关于文档模块第三轮问题，重点明确 PDF section 规则、短页合并、`chunk_type` 枚举、TXT 编码、标签上下文构造、领域词表、demo 查询、`evidence` 字段和第一版交付范围。

总体口径：

```text
第一版不要追求复杂版式完美解析。
优先保证输出结构稳定、规则可解释、A/D 容易入库和调试。
```

## A. PDF 的 section 检测

PDF 第一版采用方案 B：

```text
空行切分 + 最小长度合并
```

具体规则：

- 连续非空行组成一个段落组。
- 遇到空行切分新的段落组。
- 如果段落组过短，合并到相邻 section。
- 不做复杂标题视觉检测。

推荐阈值：

```text
min_section_chars = 100
```

处理逻辑：

```text
1. 先按空行切成 paragraph groups。
2. 如果 group < 100 字，优先合并到前一个 group。
3. 如果没有前一个 group，则暂存并尝试和下一个 group 合并。
4. 合并后仍然很短也可以保留，但标记为 short_section。
```

不建议第一版只用方案 A，因为 PDF 一页可能会被切得太碎。

## B. PDF 短页合并阈值

“很短”的定义采用绝对字数，而不是比例。

推荐：

```text
short_page_chars = 120
```

规则：

- 如果一页文本少于 120 字，可以和下一页合并。
- 可以连续合并多页，直到达到 300 字左右。
- 最多连续合并 3 页，避免把目录、封面、插图页和正文混得太长。

建议字段：

```json
{
  "page_start": 2,
  "page_end": 4
}
```

如果连续几页都只有几十字：

```text
继续合并，但最多合并 3 页。
仍然很短就保留为 heading_only 或 short_text。
```

## C. chunk_type 枚举

第一版需要区分，但不要过细。

采用以下枚举：

| chunk_type | 触发条件 |
|---|---|
| `text` | 默认普通文本 chunk |
| `code` | Markdown 代码块 |
| `heading_only` | 只有标题、没有正文的段落或幻灯片 |
| `table` | 表格内容被提取并转成文本 |

第一版必须填写 `chunk_type`，不要用 `null`。

默认：

```text
chunk_type = "text"
```

PPTX 只有标题的过渡页：

```text
chunk_type = "heading_only"
```

Markdown 代码块：

```text
chunk_type = "code"
```

DOCX 表格：

```text
chunk_type = "table"
```

## D. TXT 编码检测

第一版采用方案 B：

```text
按 UTF-8 / UTF-8-SIG / GBK / GB2312 / UTF-16 顺序尝试，不引入 chardet。
```

建议顺序：

```python
encodings = ["utf-8-sig", "utf-8", "gbk", "gb2312", "utf-16"]
```

原因：

- 少一个依赖。
- 足够覆盖常见中文 TXT。
- 解析失败时能给出明确 warning。

可以预留：

```text
如果后续遇到大量编码异常，再把 chardet 放入 optional requirements。
```

## E. generate_tags 的 context 构造

采用方案 B：

```text
提供独立函数 build_tag_context(parse_result, chunks) -> str。
generate_tags 只接收构造好的 context。
```

原因：

- 解耦。
- 方便 A/D 以后自己构造 context。
- 方便测试不同 context 策略对标签质量的影响。

推荐接口：

```python
def build_tag_context(parse_result: DocumentParseResult, chunks: list[DocumentChunk], max_chars: int = 4000) -> str:
    ...
```

context 内容顺序：

```text
1. 文档标题
2. 文件名和格式
3. section_title 列表
4. 前 3 个 chunk 的 summary + 少量 text
5. 各章节 summary
6. 高频候选关键词
```

## F. 领域词表维护

B 先建立初始版本，不需要 A 提供完整词表。

建议放在：

```text
document_module/resources/term_dict.json
```

不要硬编码在 `tagger.py` 里。

原因：

- 后续扩展方便。
- A/D 可以直接补词。
- README 中能说明词表如何维护。

建议结构：

```json
{
  "tech": ["OCR", "ASR", "CLIP", "Agent", "Faiss", "embedding"],
  "domain": ["人工智能", "教育", "文学", "软件工程", "多媒体"],
  "genre": ["说明文", "报告", "散文", "论文", "接口文档", "演示稿"],
  "usage": ["答辩材料", "学习资料", "项目文档", "接口规范"]
}
```

第一版词表规模：

```text
50-150 个词即可。
```

不要为了凑数量写一两百个低质量词。

## G. Demo 示例查询

采用默认硬编码 + 命令行覆盖。

默认查询 3-5 个：

```text
描写春天的文章
计算机教育
OCR 接口怎么调用
项目分工
多模态检索
```

支持命令行覆盖：

```text
python demo_document_module.py --input samples --output demo_report.md --query "OCR 接口怎么调用"
```

每个查询建议展示两个流程：

```text
1. 文档切片搜索结果
2. 基于 top chunks 的文档问答结果
```

如果问答还只是 Heuristic，也要明确标注：

```text
qa_mode = heuristic
```

## H. DocumentTag 的 evidence 字段

`evidence` 表示文档中实际支持该标签的证据。

优先级：

```text
原文短语 > 原文句子片段 > 共现关键词
```

建议：

- 每个标签最多 3 条 evidence。
- 每条 evidence 不超过 40 字。
- 如果找不到合适证据，允许 `[]`，但 confidence 应降低。

示例：

```json
{
  "name": "计算机教育",
  "type": "topic",
  "confidence": 0.82,
  "evidence": ["信息技术课程", "编程教学", "课堂实践"]
}
```

不要让 `evidence` 变成 LLM 自己编的解释，它必须来自文档文本或可追溯关键词。

## I. 第一版文件支持范围

理解正确。第一版交付范围如下：

必须完成：

```text
MD
TXT
DOCX
```

尽最大努力完成：

```text
PDF：使用 pypdf，支持可提取文本的 PDF
PPTX：使用 python-pptx，支持普通文本幻灯片
```

明确不做：

```text
图片型 PDF OCR
复杂表格结构还原
公式提取
SmartArt 结构解析
图片内容理解
```

如果 PDF/PPTX 解析效果有限，返回 warnings，不要直接崩溃。

## 最终收尾口径

B 第一版实现要满足：

```text
结构稳定
轻依赖
离线可跑
可生成 demo_report.md
有清晰 warnings/errors
输出 chunk_type / embedding_text / evidence
方便 A/D 接入向量检索和前端展示
```

不要把第一版目标定成“完美文档智能体”。第一版目标是：

```text
把文档稳定转成可检索、可定位、可解释的结构化 chunk。
```

