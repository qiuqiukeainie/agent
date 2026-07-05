# 数据集选择与检索质量提升方案

## 推荐数据集

第一阶段推荐使用 **MS COCO 2017 validation + captions**。

原因：

- 规模适中，`val2017` 约 5000 张图，适合课程项目和本地机器调试。
- 每张图有英文 caption，天然适合 OpenAI CLIP 的英文文本-图像检索。
- 场景覆盖人、动物、交通、食物、运动、街景、室内等常见素材管理需求。
- 可以直接构建检索评测集：caption 作为 query，对应图片作为 ground truth。

下载入口：

- COCO 官网：https://cocodataset.org/#download
- 常用文件：
  - `val2017.zip`
  - `annotations_trainval2017.zip`

## 为什么不是一开始用很大的数据集

Open Images、YFCC、LAION 这类数据更大，但第一阶段会带来三个问题：

- 下载和清洗成本高。
- 标注格式复杂，容易把时间花在数据工程上。
- A 当前要验证的是 CLIP 召回链路和 Agent 调度，不需要百万级素材。

等 D 的 Faiss + SQLite 管线完成后，再扩到 Open Images 子集更合理。

## 检索效果差的主要原因

当前问题不在语义拆分，而在跨模态召回：

1. OpenAI CLIP 的文本侧主要适配英文，中文自然语言直接输入效果不稳定。
2. 查询里的人名，如“小明”，CLIP 无法知道它对应素材里的哪张脸。
3. 地点、场景如果没有结构化标签，只靠图文相似度会有误差。
4. 素材库太小，候选空间不足，排序结果看起来会更随机。

## A 阶段改进策略

当前已采用：

- Agent 拆出人物、地点、场景、时间等条件。
- 人物、地点、场景进入 `unresolved_conditions`，等待 B/C/D 模块提供结构化索引。
- 对 CLIP 召回使用英文视觉 prompt 增强，而不是只把中文原句丢给 CLIP。
- 每个结果返回 `best_prompt`，方便排查图片为什么被召回。

后续建议：

- 短期：继续用 OpenAI CLIP + 英文 prompt 改写。
- 中期：接入 Chinese-CLIP，直接提升中文图文检索效果。
- 长期：D 接入 Faiss 后，用 COCO caption 做 Recall@K、MRR、NDCG 评测。

## 第二优先级数据集

| 数据集 | 适用场景 | 建议 |
| --- | --- | --- |
| Flickr30k | 图文检索、生活化图片 | 可以作为第二个评测集 |
| COCO-CN / Flickr30K-CN | 中文图文检索 | 接 Chinese-CLIP 后使用 |
| Open Images V7 | 大规模目标/场景素材 | D 的向量库完成后再引入 |
| Places365 | 场景分类 | B 的场景理解模块可参考 |
