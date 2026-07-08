# E 前端工作台对比与接入建议

## 结论

`agent-materials-workstation(1).zip` 不是简单静态页面，而是一套 React + Ant Design + React Query 的工作台前端，源码位于压缩包 `frontend/src/`，打包产物位于 `static/`。

建议采用“逐步迁移”：

1. 先保留当前后端与当前页面，避免演示环境突然失效。
2. 后端补齐 E 前端所需的兼容 API。
3. 单独试跑 E 前端，确认搜索、素材库、详情、任务中心能通。
4. 再决定是否替换当前 `static/`。

## E 前端主要优点

- 页面分层更清楚：搜索、素材库、标签、人物、任务、系统状态分页面管理。
- 使用 Ant Design，表格、抽屉、分页、批量操作的观感更统一。
- 素材库支持分页，理论上比一次性渲染全部素材更适合大库。
- 详情页用 Drawer，不会阻断当前页面上下文。
- 已经预留批量打标、批量归档、任务中心等产品化能力。

## 与当前后端的主要差异

| E 前端接口 | 当前后端原接口 | 处理方式 |
|---|---|---|
| `GET /api/assets?kind=&library=&limit=&offset=` | `GET /api/assets` | 已补分页/筛选兼容 |
| `GET /api/asset/{id}` | `GET /api/asset?id={id}` | 已补 REST 兼容 |
| `DELETE /api/asset/{id}` | `POST /api/delete-asset` | 已补 REST 删除 |
| `POST /api/asset/{id}/tags` | `POST /api/asset-tags` | 已补手动标签兼容 |
| `GET /api/tag-assets?tag=` | `GET /api/tags?tag=` | 已补标签素材兼容 |
| `POST /api/assets/batch` | 无统一批量接口 | 已补批量 tag/archive/delete/favorite |
| `POST /api/tasks` | `POST /api/process-assets` | 已补任务入口兼容 |
| `POST /api/persons/merge` | `POST /api/merge-persons` | 已补基础兼容 |

## 不建议直接覆盖的原因

- E 的 `static/index.html` 引用 `/assets/index-*.js/css`，当前服务器默认主页面是 `/static/index.html`，直接替换会影响已有页面。
- E 的源码依赖 React 19、Ant Design 6、Vite 8，若要重新开发需要 Node 依赖环境。
- E 当前没有文档检索页面，也没有展示新版 `structured_slots` 的长句拆解结果。
- E 的批量归档等按钮已有前端入口，但后端语义还比较基础，需要继续补业务逻辑。

## 推荐下一步

1. 先保留当前页面，把 E 的分页素材库和 Drawer 详情交互迁移过来。
2. 给 E 增加“文档检索”页面，调用 `GET /api/document-search` 与 `POST /api/document-qa`。
3. 在搜索页展示 `plan.structured_slots`，作为 A 负责的“调度搜索”亮点。
4. 等 E 前端所有核心 API 通了，再切换默认首页。
