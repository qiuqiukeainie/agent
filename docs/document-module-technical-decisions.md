# B 文档模块技术选型与接口细化答复

本文档用于回答 B 对“文档理解、切片检索与文档问答模块”的技术细节问题。总体原则是：

```text
文档模块不要强绑定某个 LLM / Dify / 外部 API。
先做成可离线运行的 Python 模块，LLM 能力通过可替换接口接入。
```

也就是说：

```text
解析、切片、基础标签、chunk 输出：必须离线可跑
摘要、主题标签、文档问答：设计 LLM 接口，但允许 fallback
```

## 一、技术选型与外部依赖

### 1. LLM 选型

目前主项目中没有固定接入某个 LLM。因此文档模块不要写死 OpenAI、Dify 或某个本地模型。

建议做一个抽象接口：

```python
class LLMClient:
    def summarize(self, text: str) -> str:
        ...

    def generate_tags(self, context: str) -> list[dict]:
        ...

    def answer(self, question: str, contexts: list[dict]) -> dict:
        ...
```

默认实现建议分两类：

```text
HeuristicLLMClient：不用大模型，基于规则、关键词、摘要截断，保证离线可跑
ExternalLLMClient：以后接 Dify、OpenAI 兼容接口、本地模型
```

第一版统一使用一个 `LLMClient` 即可，不强制区分摘要模型、标签模型和问答模型。

后续可以预留配置：

```text
summary_model：较小模型
tag_model：较小模型
qa_model：较强模型
```

但第一版不要把复杂度拉太高。

是否要求纯离线：

```text
第一版必须支持纯离线 fallback。
外部 LLM 是增强项，不是硬依赖。
```

### 2. 文档解析库选择

建议第一版选型：

| 格式 | 推荐库 | 说明 |
|---|---|---|
| PDF | `pdfplumber` 或 `pypdf` | 不优先使用 PyMuPDF，避免 AGPL 协议风险 |
| DOCX | `python-docx` | 结构化读取段落、标题、表格较方便 |
| PPTX | `python-pptx` | 可读取幻灯片文本，复杂 SmartArt 不作为第一版重点 |
| MD/TXT | 标准库 | 直接按文本读取 |

关于 PyMuPDF：

```text
考虑 AGPL 协议风险，第一版不作为默认依赖。
如果后续只是课堂演示可以选用，但文档里要说明可替换。
```

扫描件 PDF：

```text
第一版不要求 B 做 OCR。
图片型 PDF 解析不到文字时，返回空页/警告，由 A/D 或 C 的 OCR 管线后续处理。
```

DOCX/PPTX 中的图片、表格、公式：

```text
第一版只提取文字。
表格按行拼成文本。
公式、图片先忽略或记录 placeholder。
复杂版式不是第一阶段重点。
```

## 二、切片策略细节

### 3. 切片边界策略

切片优先级：

```text
标题边界 > 段落边界 > 句子边界 > 固定长度兜底
```

不要优先硬切在句子中间。

chunk 长度建议：

```text
目标长度：500 中文字左右
允许范围：300-800 中文字
极端兜底：最长不超过 1200 字
```

是否 overlap：

```text
需要 overlap。
建议 80-120 中文字，默认 100 字。
```

代码块处理：

```text
MD 代码块尽量单独作为 chunk。
如果代码块超过 1200 字，不截断内容，但标记 chunk_type="code"，允许超长。
```

### 4. 标题/章节信息继承

`section_title` 建议保留完整路径。

例如：

```text
第二章 > 2.2 方法
```

不要只填最近一级标题。这样搜索结果更容易解释，也方便前端展示。

chunk 字段建议：

```json
{
  "section_title": "第二章 > 2.2 方法",
  "heading_path": ["第二章", "2.2 方法"]
}
```

### 5. 空文档 / 极短文档

极短文档：

```text
不拒绝。
100 字也生成一个 chunk。
```

空页 PDF：

```text
跳过空 text 页。
但 parse_result 里记录 skipped_pages。
```

整篇空文档：

```text
返回 result.status = "empty_text"
不要直接崩溃。
```

## 三、标签生成

### 6. 标签生成输入

第一版标签生成不要直接喂全文，成本高，也容易被长文档噪声影响。

推荐输入：

```text
文档标题
+ 所有 section_title
+ 前 3 个 chunk 的 text/summary
+ 每个章节的 chunk summary
+ 高频候选关键词
```

也就是构造一个 `tag_context`：

```python
tag_context = build_tag_context(parse_result, chunks)
```

上下文建议控制在 3000-5000 中文字以内。

### 7. confidence 计算

第一版不要完全相信 LLM 自评分。

建议：

```text
confidence = 0.5 * LLM 自评分
           + 0.3 * 证据覆盖度
           + 0.2 * 出现频率/章节覆盖
```

如果不用 LLM，则用启发式：

```text
主题词出现越多、覆盖章节越多、出现在标题里，confidence 越高。
```

统一范围：

```text
0.0 - 1.0
低于 0.45 的标签不输出。
```

### 8. 标签去重与合并

需要后处理。

必须做：

```text
去空白
统一大小写
去重复
去停用词
限制长度
同义合并
```

建议黑名单：

```text
本文
我们
介绍
内容
研究
一个
这个
进行
说明
相关
```

“计算机教育”和“信息技术课程”不一定算重复，可以同时保留，但类型要区分：

```text
计算机教育：topic
信息技术课程：content/domain
```

## 四、语义匹配与查询扩展

### 9. 查询扩展词生成

第一版不要实时调用 LLM 扩展查询，避免搜索变慢。

推荐：

```text
离线为文档生成主题标签、摘要、关键词、扩展词。
搜索时直接使用这些字段。
```

实时查询扩展可以由 A 的 Search Orchestrator 做轻量规则扩展。

示例：

```text
春天 -> 花开 / 柳树 / 温暖 / 生机 / 自然
计算机教育 -> 信息技术 / 编程教学 / 课程 / 课堂 / 教育信息化
```

### 10. 多路召回融合策略

B 不需要最终决定全局权重，但需要输出可用于融合的字段。

B 输出：

```text
chunk_vector_text
summary
tags
keywords
embedding_text
```

A/D 融合：

```text
chunk 向量分：主路
标签命中：boost
摘要命中：boost
关键词命中：轻 boost
```

建议默认权重：

```text
0.70 chunk_vector
0.15 tag_match
0.10 summary_match
0.05 keyword_match
```

权重写成配置，不要写死。

## 五、文档问答

### 11. `answer_from_chunks` 的上下文构造

`answer_from_chunks(question, chunks)` 里的 chunks 默认是 A/D 已经检索排好序的 top-k。

B 内部仍需做二次限制：

```text
最多使用 top 5 chunks
总上下文不超过 6000 中文字
```

如果 chunks 来自多篇文档，prompt 里必须标清来源：

```text
[1] filename=xxx.md, page=2, section=接口说明
content...
```

### 12. 回答引用格式

如果答案使用了多个 chunk，`sources` 应列出所有用到的 chunk，最多 5 个。

如果找不到答案：

```json
{
  "answer": "未在当前素材库文档中找到可靠答案。",
  "sources": []
}
```

不要让 LLM 脱离文档自由回答。原则是：

```text
无引用，不回答。
```

## 六、接口与集成边界

### 13. 模块是否有状态

第一版模块设计成纯函数为主。

```text
parse_document(path) -> result
chunk_document(result) -> chunks
generate_document_tags(result, chunks) -> tags
answer_from_chunks(question, chunks) -> answer
```

缓存由 A/D 负责，不要求 B 模块内部维护数据库状态。

B 可以提供可选 cache 参数，但不作为强要求。

### 14. 异步与并发

第一版不用 `async def`。

要求：

```text
函数本身无全局可变状态
尽量线程安全
不要在模块里维护共享队列
```

并发、任务队列、进度条由 A/D 后端处理。

### 15. 错误处理约定

不要直接让异常一路炸到主系统。

建议统一返回：

```json
{
  "ok": false,
  "error": "password_protected_pdf",
  "message": "PDF 文件受密码保护，无法解析"
}
```

内部严重错误可以抛自定义异常，由 A/D 捕获。

推荐定义：

```python
class DocumentParseError(Exception):
    ...

class DocumentUnsupportedError(Exception):
    ...
```

`ParseResult` 中建议包含：

```python
status: "ok" | "empty_text" | "partial" | "failed"
warnings: list[str]
error: str | None
```

### 16. 与 A/D 的向量生成对接

B 需要额外输出 `embedding_text`。

不要只让 A/D 拿原始 `text` 做 embedding。

建议：

```text
embedding_text = section_title + "\n" + summary + "\n" + text
```

因为标题和摘要能增强语义。

chunk 示例：

```json
{
  "text": "正文原文...",
  "summary": "本段介绍 OCR 接口参数。",
  "embedding_text": "接口规范 > OCR 接口说明\n本段介绍 OCR 接口参数。\n正文原文..."
}
```

A/D 使用 `embedding_text` 生成向量。

## 七、交付与验证

### 17. 测试素材

B 可以自己准备 5-10 个测试文档。

建议包括：

```text
一篇描写春天的文章
一篇计算机教育文章
一份接口说明文档
一份项目报告
一份 PPTX
一份极短 TXT
一份空页/图片型 PDF
```

需要提供一个简单 demo 脚本：

```text
python demo_document_module.py
```

演示：

```text
解析
切片
标签生成
语义搜索样例
文档问答样例
```

### 18. README 粒度

需要详细版，不要只写函数名。

README 至少包括：

```text
模块目标
支持格式
依赖安装
核心函数
输入输出示例
切片策略
标签生成策略
问答策略
错误处理
已知限制
demo 运行方式
```

## 八、最终结论

第一版不要追求“大模型很强”，而是追求：

```text
模块稳定
接口清晰
离线可跑
LLM 可替换
chunk 结构规范
方便 A/D 入库和向量化
```

协作边界：

```text
B 负责文档解析、切片、摘要、标签、问答逻辑。
A/D 负责入库、向量化、搜索融合、前端展示和评测。
```

