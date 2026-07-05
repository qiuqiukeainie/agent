# 项目文档入口

这里放团队协作、接口、算法、数据、前端和演示相关文档。新成员优先从本文件开始看。

## 推荐阅读顺序

1. `phase1-architecture.md`：第一阶段总体架构。
2. `team-interface-spec.md`：早期团队接口约定。
3. `api-contracts.md`：当前 HTTP API 和模块协作契约。
4. `frontend-guide.md`：前端页面结构、命名规则和新增功能流程。
5. `c-multimodal-fusion-api.md`：C 方向 OCR/ASR/标签融合接口。
6. `face-module-api.md`：B 方向人脸模块接口。
7. `document-search-guide.md`：文档素材接入和文档搜索说明。
8. `evaluation-guide.md`：检索评测方法。
9. `demo-checklist.md`：演示前检查和推荐流程。

## 按成员分工查看

### A：算法负责人/组长

重点文档：

- `phase1-architecture.md`
- `api-contracts.md`
- `retrieval-tuning.md`
- `document-search-guide.md`
- `evaluation-guide.md`

关注内容：CLIP 检索、Agent 调度、接口规范、评测解释、团队统筹。

### B：视觉算法工程师

重点文档：

- `face-module-api.md`
- `2026-06-29-face-module-design.md`
- `2026-06-29-face-module-implementation.md`
- `api-contracts.md`

关注内容：人脸检测、向量、聚类、人物 ID、模块输出格式。

### C：多模态融合工程师

重点文档：

- `c-multimodal-fusion-api.md`
- `api-contracts.md`
- `document-search-guide.md`

关注内容：OCR、ASR、字幕、文档文本、统一标签。

### D：数据与系统工程

重点文档：

- `d-system-foundation.md`
- `dataset-plan.md`
- `api-contracts.md`
- `evaluation-guide.md`

关注内容：素材入库、向量索引、SQLite 元数据、评测集、性能对比。

### E：全栈与产品负责人

重点文档：

- `frontend-guide.md`
- `api-contracts.md`
- `demo-checklist.md`

关注内容：页面结构、接口调用、交互体验、演示稳定性、文档和答辩材料。

## 文档维护规则

- 接口变化必须同步更新 `api-contracts.md`。
- 前端结构变化必须同步更新 `frontend-guide.md`。
- 新增评测指标或查询集格式变化必须同步更新 `evaluation-guide.md`。
- 新增演示流程或已知问题必须同步更新 `demo-checklist.md`。
- 历史阶段记录不要随意删除，可以新增日期文档保留演进过程。

