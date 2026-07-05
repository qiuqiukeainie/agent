# 前端协作说明

本文档给负责 Web 界面、联调和演示的成员使用。当前前端是轻量原生页面，没有 React/Vue 构建链，主要文件都在 `static/` 目录下。

## 文件分工

- `static/index.html`：页面结构、视图容器、按钮、输入框和弹窗骨架。
- `static/app.js`：页面状态、接口请求、素材渲染、搜索结果渲染、人物分组交互、任务面板和详情弹窗逻辑。
- `static/styles.css`：所有页面布局和组件样式。
- `app.py`：后端 HTTP 服务和 API 路由。前端新增功能时，优先复用已有 API；确实需要新能力再补路由。

## 页面视图

当前 Web 按功能分层，而不是把所有模块常驻在同一页：

- `搜索`：自然语言检索、Agent 执行计划、检索结果、严格筛选。
- `素材库`：公共库/个人库素材浏览，支持图片、视频、文档统一展示。
- `标签`：标签浏览、按标签查看素材、观察 OCR/ASR/人工标签等来源。
- `人物`：个人库人脸聚类、人物命名、合并、删除、按人物筛选。
- `系统`：状态、任务队列、评测、Agent 建议、统计信息。

新功能应优先放入最贴近的视图，避免继续堆到首页。比如 OCR 批处理属于 `系统`，人物别名属于 `人物`，文档搜索结果展示属于 `搜索/素材库`。

## 前端状态约定

`static/app.js` 中的核心状态变量：

- `currentAssets`：素材库当前缓存。
- `currentResults`：最近一次搜索结果。
- `currentDetail`：详情弹窗中打开的素材。
- `currentPersons`：人物聚类列表。
- `assetsLoaded/personsLoaded/statusLoaded/tagsLoaded`：控制懒加载，避免页面初次进入时一次性加载所有模块导致卡顿。

新增视图时建议也采用懒加载：第一次打开视图时请求数据，后续通过刷新按钮或操作结果局部更新。

## API 调用规则

前端请求统一使用相对路径，例如：

```js
fetch("/api/assets")
fetch(`/api/search?q=${encodeURIComponent(query)}&limit=36`)
fetch("/api/asset-tags", { method: "POST", body: JSON.stringify(payload) })
```

约定：

- GET 用查询参数传简单筛选条件。
- POST 请求体统一 JSON，上传文件除外。
- 所有用户输入必须使用 `encodeURIComponent` 或 JSON body，不手拼未转义字符串。
- 接口失败时不要静默失败，至少写入当前面板的状态区域。
- 渲染用户文件名、标签、OCR 文本时必须走 `escapeHtml()`。

## 新增前端功能流程

1. 在 `static/index.html` 中添加按钮、面板或列表容器。
2. 在 `static/app.js` 顶部用 `document.querySelector()` 绑定 DOM。
3. 新增 `loadXxx()` 或 `renderXxx()` 函数，保持请求逻辑和渲染逻辑分开。
4. 在 `showView()` 或对应按钮事件里触发懒加载。
5. 在 `static/styles.css` 中新增样式，类名建议以模块名开头，例如 `doc-`、`eval-`、`person-`。
6. 手动测试：页面刷新、空数据、接口失败、长文件名、视频/文档/图片混排。

## 卡片与媒体展示规范

素材卡片必须保持固定比例，避免视频封面、文档封面、图片封面把列表撑乱：

- 列表卡片封面统一使用固定比例容器。
- 图片使用 `object-fit: cover`。
- 视频缩略图使用后端生成的 poster 或前端固定容器承接。
- 文档使用文档缩略图或统一文档卡片，不直接把长文本塞进封面区域。

详情页可以展示完整视频播放器或文档入口，但列表页不应自动播放视频，也不应加载完整视频。

## UI 命名规则

- 按钮 id：动词 + 对象，例如 `runEvaluation`、`rebuildPersons`。
- 列表容器 id：对象 + `List/Rows/Grid`，例如 `personList`、`tagRows`、`resultGrid`。
- 状态区域 id：对象 + `Status/Panel`，例如 `detailStatus`、`healthPanel`。
- CSS 类名：模块前缀 + 语义名，例如 `person-card`、`eval-profile`、`asset-thumb`。
- data 属性：用于事件委托，例如 `data-asset-id`、`data-person-id`、`data-tag-name`。

## 不建议在前端做的事

- 不在前端做人脸聚类、向量检索、OCR/ASR、标签融合等模型逻辑。
- 不把大段业务规则硬编码在页面中，规则应尽量沉到后端接口。
- 不在列表页一次性请求所有详情信息。
- 不让视频在列表页自动加载完整文件。

