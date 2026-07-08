installDocumentView();

const fileInput = document.querySelector("#fileInput");
const uploadLibrary = document.querySelector("#uploadLibrary");
const queryInput = document.querySelector("#queryInput");
const searchButton = document.querySelector("#searchButton");
const resultLimit = document.querySelector("#resultLimit");
const searchLibrary = document.querySelector("#searchLibrary");
const strictFilter = document.querySelector("#strictFilter");
const assetKindFilter = document.querySelector("#assetKindFilter");
const assetLibraryFilter = document.querySelector("#assetLibraryFilter");
const statusBox = document.querySelector("#status");
const modelStatus = document.querySelector("#modelStatus");
const libraryStats = document.querySelector("#libraryStats");
const agentRecommendations = document.querySelector("#agentRecommendations");
const assetSummary = document.querySelector("#assetSummary");
const personStats = document.querySelector("#personStats");
const rebuildPersons = document.querySelector("#rebuildPersons");
const mergePersons = document.querySelector("#mergePersons");
const personList = document.querySelector("#personList");
const personDetail = document.querySelector("#personDetail");
const agentPlan = document.querySelector("#agentPlan");
const assetList = document.querySelector("#assetList");
const resultGrid = document.querySelector("#resultGrid");
const assetDialog = document.querySelector("#assetDialog");
const detailTitle = document.querySelector("#detailTitle");
const detailKind = document.querySelector("#detailKind");
const detailMedia = document.querySelector("#detailMedia");
const detailMeta = document.querySelector("#detailMeta");
const manualTagsInput = document.querySelector("#manualTagsInput");
const saveManualTags = document.querySelector("#saveManualTags");
const deleteAsset = document.querySelector("#deleteAsset");
const favoriteAsset = document.querySelector("#favoriteAsset");
const detailStatus = document.querySelector("#detailStatus");
const tagRows = document.querySelector("#tagRows");
const faceRows = document.querySelector("#faceRows");
const archiveRows = document.querySelector("#archiveRows");
const ocrText = document.querySelector("#ocrText");
const asrText = document.querySelector("#asrText");
const textSummaryRows = document.querySelector("#textSummaryRows");
const textSignalRows = document.querySelector("#textSignalRows");
const similarRows = document.querySelector("#similarRows");
const healthPanel = document.querySelector("#healthPanel");
const startVideoOcrTask = document.querySelector("#startVideoOcrTask");
const refreshTasks = document.querySelector("#refreshTasks");
const taskList = document.querySelector("#taskList");
const runEvaluation = document.querySelector("#runEvaluation");
const evaluationPanel = document.querySelector("#evaluationPanel");
const tagLibraryFilter = document.querySelector("#tagLibraryFilter");
const tagFacetList = document.querySelector("#tagFacetList");
const tagAssetList = document.querySelector("#tagAssetList");
const tagSummary = document.querySelector("#tagSummary");
const documentQueryInput = document.querySelector("#documentQueryInput");
const documentLibrary = document.querySelector("#documentLibrary");
const documentSearchButton = document.querySelector("#documentSearchButton");
const documentQaButton = document.querySelector("#documentQaButton");
const rebuildDocumentChunks = document.querySelector("#rebuildDocumentChunks");
const documentAnswer = document.querySelector("#documentAnswer");
const documentResultList = document.querySelector("#documentResultList");
const chatPanelEl = document.querySelector("#chatPanel");
const chatMessagesEl = document.querySelector("#chatMessages");
const clarificationHintEl = document.querySelector("#clarificationHint");
const clearChatButton = document.querySelector("#clearChatButton");
const closeChatButton = document.querySelector("#closeChatButton");
const openChatButton = document.querySelector("#openChatButton");
const chatInput = document.querySelector("#chatInput");
const chatSendButton = document.querySelector("#chatSendButton");

const ASSET_PREVIEW_LIMIT = 96;
let currentResults = [];
let currentAssets = [];
let currentDetail = null;
let currentPersons = [];
let assetsLoaded = false;
let personsLoaded = false;
let statusLoaded = false;
let tagsLoaded = false;
let selectedTag = "";
let currentSessionId = null;
let chatMessages = [];

function installDocumentView() {
  const tabs = document.querySelector(".view-tabs");
  const searchView = document.querySelector("#searchView");
  if (!tabs || !searchView || document.querySelector("#documentView")) return;

  const tab = document.createElement("button");
  tab.className = "tab-button";
  tab.type = "button";
  tab.dataset.view = "documentView";
  tab.textContent = "文档检索";
  const libraryTab = tabs.querySelector('[data-view="libraryView"]');
  tabs.insertBefore(tab, libraryTab || null);

  const view = document.createElement("section");
  view.id = "documentView";
  view.className = "view";
  view.innerHTML = `
    <section class="work-panel document-work-panel">
      <div class="panel-head">
        <div>
          <div class="panel-title">文档切片检索与问答</div>
          <p>按段落/章节命中文档片段，适合搜索长文大意、接口说明、项目文档和汇报材料。</p>
        </div>
        <button id="rebuildDocumentChunks" type="button">重建文档切片</button>
      </div>
      <div class="document-searchbar">
        <input id="documentQueryInput" type="search" placeholder="试试：描写春天的文章、计算机教育文章、OCR 接口怎么调用" />
        <select id="documentLibrary">
          <option value="all" selected>全部库</option>
          <option value="personal">个人库</option>
          <option value="public">公共库</option>
        </select>
        <button id="documentSearchButton" type="button">搜文档</button>
        <button id="documentQaButton" type="button">问答</button>
      </div>
      <div id="documentAnswer" class="document-answer"></div>
      <div id="documentResultList" class="document-result-list"></div>
    </section>
  `;
  searchView.insertAdjacentElement("afterend", view);
}

document.querySelectorAll(".tab-button").forEach((button) => {
  button.addEventListener("click", () => showView(button.dataset.view));
});

fileInput.addEventListener("change", async () => {
  if (!fileInput.files.length) return;
  const form = new FormData();
  form.append("library", uploadLibrary?.value || "personal");
  [...fileInput.files].forEach((file) => form.append("files", file));
  statusBox.textContent = `正在上传到 ${libraryLabel(uploadLibrary?.value)} 并建立语义索引...`;

  try {
    const response = await fetch("/api/upload", { method: "POST", body: form });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "上传失败");
    statusBox.textContent = `已上传并索引 ${payload.uploaded.length} 个素材。`;
    fileInput.value = "";
    assetsLoaded = false;
    personsLoaded = false;
    statusLoaded = false;
    if (isViewActive("libraryView")) await loadAssets();
    if (isViewActive("peopleView")) await loadPersons();
    if (isViewActive("systemView")) await loadSystemInfo();
  } catch (error) {
    statusBox.textContent = `上传失败：${error.message}`;
  }
});

searchButton.addEventListener("click", search);
queryInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") search();
});
clearChatButton?.addEventListener("click", clearChat);
closeChatButton?.addEventListener("click", () => {
  if (chatPanelEl) chatPanelEl.hidden = true;
  if (openChatButton) openChatButton.hidden = false;
});
openChatButton?.addEventListener("click", () => {
  showChatPanel();
  chatInput?.focus();
});
chatSendButton?.addEventListener("click", () => chatSearch());
chatInput?.addEventListener("keydown", (event) => {
  if (event.key === "Enter") chatSearch();
});
strictFilter.addEventListener("change", renderSearchResults);
resultLimit.addEventListener("change", () => {
  if (queryInput.value.trim()) search();
});
assetKindFilter.addEventListener("change", renderAssetList);
assetLibraryFilter.addEventListener("change", renderAssetList);
assetList.addEventListener("click", (event) => {
  const item = event.target.closest("[data-asset-id]");
  if (item) openAssetDetail(item.dataset.assetId);
});
resultGrid.addEventListener("click", (event) => {
  const item = event.target.closest("[data-asset-id]");
  if (item) openAssetDetail(item.dataset.assetId);
});
saveManualTags.addEventListener("click", saveManualTagEdits);
deleteAsset.addEventListener("click", deleteCurrentAsset);
favoriteAsset?.addEventListener("click", toggleCurrentFavorite);
rebuildPersons.addEventListener("click", rebuildPersonClusters);
mergePersons.addEventListener("click", mergeSelectedPersons);
startVideoOcrTask?.addEventListener("click", startVideoOcrProcessing);
refreshTasks?.addEventListener("click", loadTasks);
runEvaluation?.addEventListener("click", loadEvaluation);
tagLibraryFilter?.addEventListener("change", () => loadTags(""));
tagFacetList?.addEventListener("click", (event) => {
  const item = event.target.closest("[data-tag-name]");
  if (item) loadTags(item.dataset.tagName);
});
tagAssetList?.addEventListener("click", (event) => {
  const item = event.target.closest("[data-asset-id]");
  if (item) openAssetDetail(item.dataset.assetId);
});
documentSearchButton?.addEventListener("click", searchDocuments);
documentQaButton?.addEventListener("click", askDocumentQuestion);
documentQueryInput?.addEventListener("keydown", (event) => {
  if (event.key === "Enter") searchDocuments();
});
rebuildDocumentChunks?.addEventListener("click", rebuildDocumentChunkIndex);
documentResultList?.addEventListener("click", (event) => {
  const item = event.target.closest("[data-asset-id]");
  if (item) openAssetDetail(item.dataset.assetId);
});
agentPlan?.addEventListener("click", (event) => {
  const item = event.target.closest("[data-agent-query]");
  if (!item) return;
  queryInput.value = item.dataset.agentQuery || "";
  search();
});
personList.addEventListener("click", (event) => {
  if (event.target.matches(".person-select, .person-name-input")) return;
  if (event.target.matches(".person-rename")) {
    renamePersonGroup(event.target.dataset.personId);
    return;
  }
  if (event.target.matches(".person-delete")) {
    deletePersonGroup(event.target.dataset.personId);
    return;
  }
  const item = event.target.closest("[data-person-id]");
  if (item) renderPersonDetail(item.dataset.personId);
});

async function showView(viewId) {
  document.querySelectorAll(".tab-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === viewId);
  });
  document.querySelectorAll(".view").forEach((view) => {
    view.classList.toggle("active", view.id === viewId);
  });

  if (viewId === "libraryView" && !assetsLoaded) await loadAssets();
  if (viewId === "tagView" && !tagsLoaded) await loadTags(selectedTag);
  if (viewId === "peopleView" && !personsLoaded) await loadPersons();
  if (viewId === "systemView" && !statusLoaded) await loadSystemInfo();
}

function isViewActive(viewId) {
  return document.querySelector(`#${viewId}`)?.classList.contains("active");
}

async function ensureAssetsLoaded() {
  if (!assetsLoaded) await loadAssets({silent: true});
}

async function loadAssets(options = {}) {
  try {
    if (!options.silent) {
      assetList.innerHTML = `<div class="empty">正在读取素材库...</div>`;
      assetSummary.textContent = "正在读取素材库...";
    }
    const response = await fetch("/api/assets");
    const payload = await response.json();
    currentAssets = payload.assets || [];
    assetsLoaded = true;
    renderAssetList();
  } catch (error) {
    assetList.innerHTML = `<div class="empty">素材库加载失败：${escapeHtml(error.message)}</div>`;
  }
}

function renderAssetList() {
  const kind = assetKindFilter?.value || "all";
  const library = assetLibraryFilter?.value || "all";
  const assets = currentAssets.filter((item) => {
    const kindMatched = kind === "all" || item.kind === kind;
    const libraryMatched = library === "all" || item.library === library;
    return kindMatched && libraryMatched;
  });
  const visibleAssets = assets.slice(0, ASSET_PREVIEW_LIMIT);
  const videoCount = currentAssets.filter((item) => item.kind === "video").length;
  const imageCount = currentAssets.filter((item) => item.kind === "image").length;
  const publicCount = currentAssets.filter((item) => item.library === "public").length;
  const personalCount = currentAssets.filter((item) => item.library === "personal").length;

  assetSummary.textContent = `共 ${currentAssets.length} 个素材：公共库 ${publicCount} 个，个人库 ${personalCount} 个；图片 ${imageCount} 个，视频 ${videoCount} 个。当前展示 ${visibleAssets.length}/${assets.length} 个。`;
  assetList.innerHTML = assets.length
    ? visibleAssets.map(renderAsset).join("")
    : `<div class="empty">没有符合条件的素材。</div>`;
}

async function loadSystemInfo() {
  await Promise.all([loadStatus(), loadStats(), loadAgentRecommendations(), loadTasks()]);
  statusLoaded = true;
}

async function loadTags(tagName = selectedTag) {
  if (!tagFacetList || !tagAssetList) return;
  selectedTag = tagName || "";
  const library = tagLibraryFilter?.value || "all";
  tagFacetList.innerHTML = `<div class="empty compact">正在读取标签...</div>`;
  tagAssetList.innerHTML = "";
  try {
    const url = `/api/tags?library=${encodeURIComponent(library)}&tag=${encodeURIComponent(selectedTag)}`;
    const response = await fetch(url);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "标签读取失败");
    tagsLoaded = true;
    renderTagBrowser(payload);
  } catch (error) {
    tagFacetList.innerHTML = `<div class="empty compact">标签读取失败：${escapeHtml(error.message)}</div>`;
  }
}

function renderTagBrowser(payload) {
  const tags = payload.tags || [];
  const assets = payload.assets || [];
  tagFacetList.innerHTML = tags.length
    ? tags.map(renderTagFacet).join("")
    : `<div class="empty compact">暂无标签。</div>`;
  tagAssetList.innerHTML = selectedTag
    ? (assets.length ? assets.map(renderAsset).join("") : `<div class="empty">该标签下暂无素材。</div>`)
    : `<div class="empty">选择左侧标签查看关联素材。</div>`;
  if (tagSummary) {
    tagSummary.textContent = selectedTag
      ? `标签 ${selectedTag} 关联 ${assets.length} 个素材。`
      : `共 ${tags.length} 个标签，选择一个标签查看素材。`;
  }
}

function renderTagFacet(row) {
  const active = row.name === selectedTag ? "active" : "";
  const sources = row.sources || "";
  return `
    <button class="tag-facet ${active}" type="button" data-tag-name="${escapeHtml(row.name)}" title="${escapeHtml(sources)}">
      <span>${escapeHtml(row.name)}</span>
      <strong>${Number(row.asset_count || 0)}</strong>
    </button>
  `;
}

async function searchDocuments() {
  const query = (documentQueryInput?.value || "").trim();
  if (!query) {
    documentResultList.innerHTML = `<div class="empty">请输入文档检索问题。</div>`;
    return;
  }
  documentAnswer.innerHTML = "";
  documentResultList.innerHTML = `<div class="empty">正在检索文档片段...</div>`;
  try {
    const library = documentLibrary?.value || "all";
    const url = `/api/document-search?q=${encodeURIComponent(query)}&limit=12&library=${encodeURIComponent(library)}`;
    const response = await fetch(url);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "文档检索失败");
    renderDocumentResults(payload.results || [], payload);
  } catch (error) {
    documentResultList.innerHTML = `<div class="empty">文档检索失败：${escapeHtml(error.message)}</div>`;
  }
}

async function askDocumentQuestion() {
  const question = (documentQueryInput?.value || "").trim();
  if (!question) {
    documentAnswer.innerHTML = `<div class="empty compact">请输入文档问题。</div>`;
    return;
  }
  documentAnswer.innerHTML = `<div class="empty compact">正在基于文档片段生成回答...</div>`;
  try {
    const response = await fetch("/api/document-qa", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        question,
        top_k: 6,
        library: documentLibrary?.value || "all",
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "文档问答失败");
    const result = payload.result || {};
    documentAnswer.innerHTML = `
      <div class="document-answer-card">
        <strong>回答</strong>
        <p>${escapeHtml(result.answer || "未找到相关答案。").replace(/\n/g, "<br />")}</p>
      </div>
    `;
    renderDocumentResults(result.sources || [], {mode: result.mode});
  } catch (error) {
    documentAnswer.innerHTML = `<div class="empty compact">文档问答失败：${escapeHtml(error.message)}</div>`;
  }
}

async function rebuildDocumentChunkIndex() {
  rebuildDocumentChunks.disabled = true;
  documentResultList.innerHTML = `<div class="empty">正在重建文档切片...</div>`;
  try {
    const response = await fetch("/api/rebuild-document-chunks", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({library: documentLibrary?.value || "all"}),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "重建失败");
    const result = payload.result || {};
    documentResultList.innerHTML = `<div class="empty">已处理 ${Number(result.processed || 0)} 个文档，当前 ${Number(result.document_chunk_count || 0)} 个切片。</div>`;
    statusLoaded = false;
  } catch (error) {
    documentResultList.innerHTML = `<div class="empty">重建文档切片失败：${escapeHtml(error.message)}</div>`;
  } finally {
    rebuildDocumentChunks.disabled = false;
  }
}

function renderDocumentResults(results, payload = {}) {
  if (!results.length) {
    documentResultList.innerHTML = `<div class="empty">没有命中文档片段。可以先上传 PDF、DOCX、PPTX、MD 或 TXT。</div>`;
    return;
  }
  documentResultList.innerHTML = results.map(renderDocumentResult).join("");
}

function renderDocumentResult(item) {
  const reason = item.match_reason || "chunk 向量与查询语义接近";
  const section = item.section_title ? `<span>${escapeHtml(item.section_title)}</span>` : "";
  const keywords = (item.keywords || []).slice(0, 8).map((tag) => `<em>${escapeHtml(tag)}</em>`).join("");
  return `
    <article class="document-result-card" data-asset-id="${escapeHtml(item.asset_id || "")}" role="button" tabindex="0">
      <div class="document-result-head">
        <strong>${escapeHtml(item.filename || "document")}</strong>
        <small>${Number(item.score || 0).toFixed(4)}</small>
      </div>
      <div class="document-result-meta">
        ${section}
        <span>片段 ${Number(item.chunk_index || 0) + 1}</span>
        <span>${escapeHtml(reason)}</span>
      </div>
      <p>${escapeHtml(item.snippet || item.summary || "")}</p>
      <div class="document-keywords">${keywords}</div>
    </article>
  `;
}

async function startVideoOcrProcessing() {
  if (!startVideoOcrTask) return;
  startVideoOcrTask.disabled = true;
  statusBox.textContent = "正在提交视频字幕 OCR 后台任务...";
  try {
    const response = await fetch("/api/process-assets", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        actions: ["text_signals"],
        library: "public",
        kind: "video",
        offset: 8,
        limit: 6,
        ocr_engine: "easyocr",
        asr_engine: "none",
        video_frames: 2,
        min_confidence: 0.45,
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "任务提交失败");
    statusBox.textContent = `已提交任务 ${payload.task.task_id}。`;
    await loadTasks();
  } catch (error) {
    statusBox.textContent = `任务提交失败：${error.message}`;
  } finally {
    startVideoOcrTask.disabled = false;
  }
}

async function loadAgentRecommendations() {
  if (!agentRecommendations) return;
  agentRecommendations.innerHTML = `<div class="empty compact">正在分析素材库...</div>`;
  try {
    const response = await fetch("/api/agent-recommendations");
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "智能建议读取失败");
    renderAgentRecommendations(payload);
  } catch (error) {
    agentRecommendations.innerHTML = `<div class="empty compact">智能建议读取失败：${escapeHtml(error.message)}</div>`;
  }
}

function renderAgentRecommendations(payload) {
  const summary = payload.summary || {};
  const rows = payload.recommendations || [];
  const summaryHtml = `
    <div class="recommendation-summary">
      <strong>平均健康度 ${Number(summary.average_health || 0)}%</strong>
      <span>${Number(summary.asset_count || 0)} 个素材 · ${Number(summary.weak_count || 0)} 个待整理 · ${Number(summary.video_without_text_count || 0)} 个视频缺文本信号</span>
    </div>
  `;
  agentRecommendations.innerHTML = summaryHtml + (rows.length
    ? rows.map(renderAgentRecommendation).join("")
    : `<div class="empty compact">当前没有明显的智能整理建议。</div>`);
}

function renderAgentRecommendation(item) {
  const button = item.payload
    ? `<button type="button" data-agent-task="${escapeHtml(JSON.stringify(item.payload))}">执行建议任务</button>`
    : "";
  return `
    <div class="recommendation-row">
      <div>
        <strong>${escapeHtml(item.title)}</strong>
        <span>${Number(item.count || 0)} 个素材 · 优先级 ${Number(item.priority || 0)}</span>
      </div>
      <p>${escapeHtml(item.reason || "")}</p>
      ${button}
    </div>
  `;
}

agentRecommendations?.addEventListener("click", async (event) => {
  const button = event.target.closest("[data-agent-task]");
  if (!button) return;
  try {
    const payload = JSON.parse(button.dataset.agentTask || "{}");
    button.disabled = true;
    statusBox.textContent = "正在提交 Agent 建议任务...";
    const response = await fetch("/api/process-assets", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "任务提交失败");
    statusBox.textContent = `已提交任务 ${result.task.task_id}。`;
    await loadTasks();
  } catch (error) {
    statusBox.textContent = `建议任务提交失败：${error.message}`;
  } finally {
    button.disabled = false;
  }
});

async function loadEvaluation() {
  if (!evaluationPanel) return;
  runEvaluation.disabled = true;
  evaluationPanel.innerHTML = `<div class="empty compact">正在运行检索对比评测...</div>`;
  try {
    const response = await fetch("/api/evaluate-retrieval?limit=20");
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "评测失败");
    renderEvaluation(payload);
  } catch (error) {
    evaluationPanel.innerHTML = `<div class="empty compact">评测失败：${escapeHtml(error.message)}</div>`;
  } finally {
    runEvaluation.disabled = false;
  }
}

function renderEvaluation(payload) {
  const warning = payload.warning ? `<div class="eval-warning">${escapeHtml(payload.warning)}</div>` : "";
  const header = `
    <div class="eval-summary">
      <strong>${Number(payload.query_count || 0)} 条查询 · ${Number(payload.labeled_query_count || 0)} 条已标注</strong>
      <span>${escapeHtml(payload.source || "")}</span>
    </div>
  `;
  const profiles = payload.profiles || [];
  evaluationPanel.innerHTML = header + warning + (profiles.length
    ? profiles.map(renderEvaluationProfile).join("")
    : `<div class="empty compact">暂无评测结果。</div>`);
}

function renderEvaluationProfile(profile) {
  const summary = profile.summary || {};
  const recall = summary.recall || {};
  const rows = (profile.evaluations || []).slice(0, 4).map(renderEvaluationRow).join("");
  return `
    <div class="eval-profile">
      <div class="eval-profile-head">
        <strong>${profileLabel(profile.profile)}</strong>
        <span>MRR ${Number(summary.mrr || 0).toFixed(3)} · R@1 ${Number(recall["R@1"] || 0).toFixed(3)} · R@5 ${Number(recall["R@5"] || 0).toFixed(3)} · ${Math.round(Number(summary.avg_latency_ms || 0))} ms</span>
      </div>
      <div class="eval-rows">${rows}</div>
    </div>
  `;
}

function renderEvaluationRow(row) {
  const rank = row.first_relevant_rank ? `命中第 ${row.first_relevant_rank}` : "未命中/未标注";
  return `
    <div class="eval-row">
      <span>${escapeHtml(row.query)}</span>
      <small>${escapeHtml(rank)} · Top1 ${escapeHtml(row.top_result || "-")}</small>
    </div>
  `;
}

function profileLabel(profile) {
  return {
    clip_only: "纯 CLIP",
    clip_tags: "CLIP + 标签",
    agent_v3: "Agent v3",
  }[profile] || profile || "-";
}

async function loadTasks() {
  if (!taskList) return;
  try {
    const response = await fetch("/api/tasks");
    const payload = await response.json();
    const tasks = payload.tasks || [];
    taskList.innerHTML = tasks.length
      ? tasks.map(renderTask).join("")
      : `<div class="empty compact">暂无后台任务。</div>`;
  } catch (error) {
    taskList.innerHTML = `<div class="empty compact">任务读取失败：${escapeHtml(error.message)}</div>`;
  }
}

function renderTask(task) {
  const total = Number(task.total || 0);
  const done = Number(task.done || 0);
  const failed = Number(task.failed || 0);
  const progress = total ? `${done}/${total}` : "-";
  const actions = (task.actions || []).join(", ");
  return `
    <div class="task-row">
      <strong>${escapeHtml(task.task_id)}</strong>
      <span>${escapeHtml(task.status)} · ${escapeHtml(actions)} · ${progress} · 失败 ${failed}</span>
      <small>${escapeHtml(task.updated_at || task.created_at || "")}</small>
    </div>
  `;
}

async function loadStatus() {
  try {
    const response = await fetch("/api/status");
    const payload = await response.json();
    modelStatus.textContent = `检索模型：${payload.vector_model}；索引：${payload.vector_index || "numpy"}；Agent：${payload.agent}；已索引 ${payload.indexed_count}/${payload.asset_count} 个素材。`;
  } catch (error) {
    modelStatus.textContent = `状态读取失败：${error.message}`;
  }
}

async function loadStats() {
  try {
    const response = await fetch("/api/stats");
    const payload = await response.json();
    const publicCount = payload.by_library?.public || 0;
    const personalCount = payload.by_library?.personal || 0;
    libraryStats.textContent = `SQLite 元数据：${payload.total_assets} 个素材；公共库 ${publicCount} 个，个人库 ${personalCount} 个；标签关联 ${payload.asset_tag_count || 0} 条；文本信号 ${payload.ocr_text_count || 0} 条；人物 ${payload.person_count || 0} 组。`;
  } catch (error) {
    libraryStats.textContent = `素材库统计读取失败：${error.message}`;
  }
}

async function loadPersons() {
  try {
    personList.innerHTML = `<div class="empty compact">正在读取人物相册...</div>`;
    const response = await fetch("/api/persons?library=personal");
    const payload = await response.json();
    currentPersons = payload.persons || [];
    personsLoaded = true;
    const faceCount = currentPersons.reduce((sum, item) => sum + Number(item.face_count || 0), 0);
    personStats.textContent = `个人库 ${currentPersons.length} 个人物分组，${faceCount} 张人脸。`;
    renderPersonList();
  } catch (error) {
    personStats.textContent = `人物读取失败：${error.message}`;
    personList.innerHTML = "";
  }
}

function renderPersonList() {
  personList.innerHTML = currentPersons.length
    ? currentPersons.map(renderPersonCard).join("")
    : `<div class="empty compact">个人库还没有人物分组。</div>`;
}

function renderPersonCard(person) {
  const name = person.display_name || person.alias || person.person_id;
  const assets = person.assets || [];
  const assetLine = assets.slice(0, 2).map((item) => item.filename).join("、");
  return `
    <div class="person-card" data-person-id="${escapeHtml(person.person_id)}" role="button" tabindex="0">
      <input class="person-select" type="checkbox" value="${escapeHtml(person.person_id)}" title="选择用于合并" />
      <button class="person-delete" type="button" data-person-id="${escapeHtml(person.person_id)}" title="删除这个人物组">x</button>
      <span class="person-thumbs">${renderPersonThumbs(assets)}</span>
      <strong>${escapeHtml(name)}</strong>
      <span class="person-name-editor">
        <input class="person-name-input" value="${escapeHtml(person.display_name || "")}" placeholder="给这个人物命名" />
        <button class="person-rename" type="button" data-person-id="${escapeHtml(person.person_id)}">保存</button>
      </span>
      <span>${escapeHtml(person.person_id)} · ${Number(person.face_count || 0)} 张人脸 · ${Number(person.asset_count || 0)} 个素材</span>
      <small title="${escapeHtml(assetLine)}">${escapeHtml(assetLine || "点击查看相关素材")}</small>
    </div>
  `;
}

async function renamePersonGroup(personId) {
  const card = personList.querySelector(`[data-person-id="${cssEscape(personId)}"]`);
  const input = card?.querySelector(".person-name-input");
  const displayName = input?.value.trim() || "";
  personStats.textContent = `正在保存 ${personId} 的名称...`;
  try {
    const response = await fetch("/api/rename-person", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({person_id: personId, display_name: displayName, library: "personal"}),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "保存人物名称失败");
    personsLoaded = false;
    await loadPersons();
  } catch (error) {
    personStats.textContent = `保存人物名称失败：${error.message}`;
  }
}

async function deletePersonGroup(personId) {
  const confirmed = window.confirm(`确定删除人物组 ${personId} 吗？这只会移除人物分组，不会删除素材文件。`);
  if (!confirmed) return;
  personStats.textContent = `正在删除 ${personId}...`;
  try {
    const response = await fetch("/api/delete-person", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({person_id: personId, library: "personal"}),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "删除人物组失败");
    assetsLoaded = false;
    personsLoaded = false;
    await loadPersons();
  } catch (error) {
    personStats.textContent = `删除人物组失败：${error.message}`;
  }
}

async function mergeSelectedPersons() {
  const selected = [...personList.querySelectorAll(".person-select:checked")].map((item) => item.value);
  if (selected.length < 2) {
    personStats.textContent = "请至少选择 2 个人物分组再合并。";
    return;
  }
  const target = selected[0];
  mergePersons.disabled = true;
  personStats.textContent = `正在合并到 ${target}...`;
  try {
    const response = await fetch("/api/merge-persons", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        target_person_id: target,
        source_person_ids: selected.slice(1),
        library: "personal",
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "合并失败");
    assetsLoaded = false;
    personsLoaded = false;
    await loadPersons();
    await applyPersonFilter(target);
  } catch (error) {
    personStats.textContent = `合并失败：${error.message}`;
  } finally {
    mergePersons.disabled = false;
  }
}

function renderPersonThumbs(assets) {
  return assets.slice(0, 4).map((item) => {
    return `<img src="${thumbnailUrl(item)}" alt="${escapeHtml(item.filename)}" loading="lazy" decoding="async" />`;
  }).join("") || `<span class="person-empty-thumb"></span>`;
}

function renderPersonDetail(personId) {
  if (!personDetail) return;
  const person = currentPersons.find((item) => item.person_id === personId);
  if (!person) {
    personDetail.innerHTML = `<div class="empty compact">未找到人物分组。</div>`;
    return;
  }
  document.querySelectorAll(".person-card").forEach((card) => {
    card.classList.toggle("selected", card.dataset.personId === personId);
  });
  const name = person.display_name || person.alias || person.person_id;
  const assets = person.assets || [];
  personDetail.innerHTML = `
    <div class="person-home-head">
      <div class="person-home-cover">${renderPersonThumbs(assets)}</div>
      <div>
        <strong>${escapeHtml(name)}</strong>
        <span>${escapeHtml(person.person_id)}</span>
        <small>${Number(person.face_count || 0)} 张人脸 · ${Number(person.asset_count || 0)} 个素材</small>
      </div>
    </div>
    <div class="person-home-actions">
      <button type="button" onclick="applyPersonFilter('${escapeHtml(person.person_id)}')">查看相关素材</button>
      <button type="button" onclick="focusPersonName('${escapeHtml(person.person_id)}')">命名</button>
    </div>
    <div class="person-home-assets">
      ${assets.length ? assets.map(renderPersonHomeAsset).join("") : `<span class="muted">暂无关联素材</span>`}
    </div>
  `;
}

function renderPersonHomeAsset(asset) {
  return `
    <div class="mini-asset" data-asset-id="${escapeHtml(asset.id)}" onclick="openAssetDetail('${escapeHtml(asset.id)}')">
      <div class="media-frame square">${renderMedia(asset, "asset")}${renderKindBadge(asset)}${renderFavoriteBadge(asset)}</div>
      <span title="${escapeHtml(asset.filename)}">${escapeHtml(asset.filename)}</span>
    </div>
  `;
}

function focusPersonName(personId) {
  const input = personList.querySelector(`[data-person-id="${cssEscape(personId)}"] .person-name-input`);
  input?.focus();
}

async function applyPersonFilter(personId) {
  await ensureAssetsLoaded();
  const person = currentPersons.find((item) => item.person_id === personId);
  searchLibrary.value = "personal";
  queryInput.value = person?.display_name || personId;
  const assetIds = new Set((person?.assets || []).map((item) => item.id));
  currentResults = currentAssets
    .filter((item) => assetIds.has(item.id))
    .map((item) => ({
      ...item,
      display_score: 1,
      score: 1,
      raw_clip_score: 0,
      tag_score: 1,
      ocr_score: 0,
      best_prompt: personId,
      tag_hits: [personId],
      ocr_hits: [],
      metadata_hits: [],
    }));
  renderSearchResults();
  await showView("searchView");
  statusBox.textContent = `${personId} 关联 ${currentResults.length} 个个人库素材。`;
}

async function rebuildPersonClusters() {
  rebuildPersons.disabled = true;
  personStats.textContent = "正在重建个人库人物聚类...";
  try {
    const response = await fetch("/api/rebuild-persons", { method: "POST" });
    const payload = await response.json();
    if (!payload.ok) throw new Error(payload.error || "重建失败");
    assetsLoaded = false;
    personsLoaded = false;
    statusLoaded = false;
    await loadPersons();
  } catch (error) {
    personStats.textContent = `人物聚类失败：${error.message}`;
  } finally {
    rebuildPersons.disabled = false;
  }
}

async function search() {
  const query = queryInput.value.trim();
  if (!query) {
    statusBox.textContent = "请输入检索描述。";
    return;
  }
  statusBox.textContent = "Agent 正在解析语义并调度检索...";
  try {
    const limit = resultLimit?.value || "36";
    const library = searchLibrary?.value || "all";
    const response = await fetch(`/api/search?q=${encodeURIComponent(query)}&limit=${encodeURIComponent(limit)}&library=${encodeURIComponent(library)}`);
    const payload = await response.json();
    renderPlan(payload.plan);
    currentResults = payload.results || [];
    renderSearchResults();
    currentSessionId = null;
    chatMessages = [];
    hideClarification();
    if (chatMessagesEl) chatMessagesEl.innerHTML = "";
    if (chatPanelEl) chatPanelEl.hidden = true;
  } catch (error) {
    statusBox.textContent = `检索失败：${error.message}`;
  }
}

function renderSearchResults() {
  const filtered = strictFilter.checked ? currentResults.filter(isStrongResult) : currentResults;
  resultGrid.innerHTML = filtered.length
    ? filtered.map(renderResult).join("")
    : `<div class="empty">没有结果。输入描述后开始检索。</div>`;
  const suffix = strictFilter.checked ? `，严格筛选后展示 ${filtered.length} 个` : "";
  statusBox.textContent = currentResults.length ? `找到 ${currentResults.length} 个结果${suffix}。` : "等待检索。";
}

function isStrongResult(item) {
  const displayScore = Number(item.display_score || item.score || 0);
  const hasStructuredHit = Boolean(
    item.tag_hits?.length ||
    item.ocr_hits?.length ||
    item.metadata_hits?.length ||
    item.filename_score >= 0.5
  );
  return displayScore >= 0.72 || hasStructuredHit;
}

async function chatSearch(queryOverride) {
  const query = (queryOverride || chatInput?.value || "").trim();
  if (!query) {
    statusBox.textContent = "请输入对话内容。";
    return;
  }
  statusBox.textContent = "Agent 正在结合对话上下文检索...";
  try {
    const library = searchLibrary?.value || "all";
    const limit = resultLimit?.value || "36";
    const body = { query, library, limit };
    if (currentSessionId) body.session_id = currentSessionId;
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "对话检索失败");
    currentSessionId = payload.session_id;
    renderPlan(payload.plan);

    // 对话检索结果仅在对话框内展示，不更新主界面
    const allResults = payload.results || [];
    const imageResults = allResults.filter((item) => item.kind === "image");
    const strictImageResults = imageResults.filter(isStrongResult);
    // 严格筛选图片结果，限制最多展示 9 个
    const dialogResults = strictImageResults.slice(0, 9);

    addChatMessage("user", query);
    addChatMessage("agent", payload.answer || (allResults.length ? `找到 ${allResults.length} 个结果。` : "没有找到匹配结果。"), payload.plan, dialogResults, allResults.length);

    if (chatInput) chatInput.value = "";
    if (payload.should_clarify && payload.clarification_text) {
      showClarification(payload.clarification_text);
    } else {
      hideClarification();
    }
    showChatPanel();
    statusBox.textContent = allResults.length
      ? `对话检索找到 ${allResults.length} 个结果（对话框内展示 ${dialogResults.length} 个精选图片）。`
      : "没有找到匹配结果，可以换个描述方式。";
  } catch (error) {
    statusBox.textContent = `对话检索失败：${error.message}`;
  }
}

function addChatMessage(role, content, plan, dialogResults, totalCount) {
  chatMessages.push({ role, content, plan, dialogResults: dialogResults || [], totalCount: totalCount || 0 });
  renderChatMessages();
}

function renderChatMessages() {
  if (!chatMessagesEl) return;
  chatMessagesEl.innerHTML = chatMessages.map((message) => {
    const label = message.role === "user" ? "你" : "Agent";
    const meta = message.role === "agent" && message.plan?.agent_summary
      ? `<div class="bubble-meta">${escapeHtml(message.plan.agent_summary)}</div>`
      : "";

    // 渲染对话框内的图片结果
    let resultsHtml = "";
    if (message.role === "agent" && message.dialogResults && message.dialogResults.length > 0) {
      const cards = message.dialogResults.map((item) => {
        return `
          <div class="bubble-result-card" data-asset-id="${escapeHtml(item.id)}" role="button" tabindex="0" title="${escapeHtml(item.filename)}">
            <img class="bubble-thumb" src="${thumbnailUrl(item)}" alt="${escapeHtml(item.filename)}" loading="lazy" decoding="async" onerror="this.classList.add('thumb-missing'); this.removeAttribute('src');" />
            <div class="bubble-result-info">${escapeHtml(truncateFilename(item.filename, 18))}</div>
          </div>`;
      }).join("");

      let moreHtml = "";
      if (message.totalCount > 9) {
        moreHtml = `<div class="bubble-result-more">还有 ${message.totalCount - 9} 个结果未展示，可继续精炼条件</div>`;
      } else if (message.totalCount > message.dialogResults.length) {
        moreHtml = `<div class="bubble-result-more">严格筛选展示 ${message.dialogResults.length} 个精选结果（共 ${message.totalCount} 个）</div>`;
      }

      resultsHtml = `
        <div class="bubble-results">${cards}</div>
        ${moreHtml}`;
    }

    return `
      <div class="chat-bubble ${message.role}">
        <div class="bubble-label">${label}</div>
        <div>${escapeHtml(message.content)}</div>
        ${resultsHtml}
        ${meta}
      </div>`;
  }).join("");
  chatMessagesEl.scrollTop = chatMessagesEl.scrollHeight;

  // 为对话框内的图片结果绑定点击事件
  chatMessagesEl.querySelectorAll(".bubble-result-card[data-asset-id]").forEach((card) => {
    card.addEventListener("click", (event) => {
      const assetId = event.currentTarget.dataset.assetId;
      if (assetId) openAssetDetail(assetId);
    });
  });
}

function truncateFilename(filename, maxLen) {
  if (!filename) return "";
  if (filename.length <= maxLen) return filename;
  const ext = filename.lastIndexOf(".");
  if (ext > 0) {
    const name = filename.substring(0, ext);
    const suffix = filename.substring(ext);
    const available = maxLen - suffix.length - 1;
    if (available > 3) return name.substring(0, available) + "…" + suffix;
  }
  return filename.substring(0, maxLen - 1) + "…";
}

function showClarification(text) {
  if (!clarificationHintEl) return;
  clarificationHintEl.textContent = text;
  clarificationHintEl.hidden = false;
  clarificationHintEl.onclick = () => {
    if (chatInput) chatInput.value = text;
    chatInput?.focus();
  };
}

function hideClarification() {
  if (clarificationHintEl) clarificationHintEl.hidden = true;
}

function showChatPanel() {
  if (chatPanelEl) chatPanelEl.hidden = false;
  if (openChatButton) openChatButton.hidden = true;
}

async function clearChat() {
  if (currentSessionId) {
    try {
      await fetch("/api/chat/clear", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: currentSessionId }),
      });
    } catch {
      // Local UI can still be cleared even if the server session already expired.
    }
  }
  currentSessionId = null;
  chatMessages = [];
  hideClarification();
  if (chatMessagesEl) chatMessagesEl.innerHTML = "";
  if (chatInput) chatInput.value = "";
  statusBox.textContent = "对话已清空。";
}

  function renderPlan(plan) {
    if (!plan) {
      agentPlan.innerHTML = "";
      return;
    }
    const steps = (plan.execution_steps || []).length
      ? `<div class="plan-block"><div class="plan-label">执行链路</div><ol>${plan.execution_steps.map((step) => `<li>${escapeHtml(step)}</li>`).join("")}</ol></div>`
      : "";
    const rewrites = (plan.query_rewrites || plan.semantic_queries || []).length
      ? `<div class="plan-block"><div class="plan-label">语义改写</div><div class="plan-chips">${(plan.query_rewrites || plan.semantic_queries || []).map((item) => `<button type="button" data-agent-query="${escapeHtml(item)}">${escapeHtml(item)}</button>`).join("")}</div></div>`
      : "";
    const hints = (plan.clarification_hints || []).length
      ? `<div class="plan-block warning"><div class="plan-label">Agent 提醒</div><ul>${plan.clarification_hints.map((hint) => `<li>${escapeHtml(hint)}</li>`).join("")}</ul></div>`
      : "";
    const quality = plan.query_quality
      ? `<span>清晰度：${Number(plan.query_quality.score || 0)}% · ${escapeHtml(plan.query_quality.message || "")}</span>`
      : "";
    const summary = plan.agent_summary
      ? `<div class="plan-summary">${escapeHtml(plan.agent_summary)}</div>`
      : "";
    const trace = (plan.trace || []).length
      ? `<div class="plan-block"><div class="plan-label">Agent Trace</div><div class="trace-list">${plan.trace.map(renderTraceStep).join("")}</div></div>`
      : "";
    agentPlan.innerHTML = `
      <div class="plan-row">
        <strong>Agent 检索计划</strong>
        <span>意图：${escapeHtml(plan.intent)}</span>
        <span>召回：${(plan.recall_routes || []).map(escapeHtml).join(" / ")}</span>
        ${quality}
      </div>
        ${summary}
        ${steps}
        ${rewrites}
        ${trace}
        <div class="plan-row">
        <span>待下游解析：${renderDeferred(plan.unresolved_conditions)}</span>
      </div>
      <div class="plan-row">
        <span>融合重排：${escapeHtml(plan.rerank_policy)}</span>
      </div>
      ${hints}
      `;
    }

  function renderTraceStep(step) {
    return `
      <div class="trace-step">
        <strong>${escapeHtml(step.title || step.stage || "-")}</strong>
        <code>${escapeHtml(formatTraceDetail(step.detail))}</code>
      </div>
    `;
  }

  function formatTraceDetail(value) {
    if (value === null || value === undefined) return "-";
    if (typeof value === "string") return value;
    try {
      return JSON.stringify(value);
    } catch {
      return String(value);
    }
  }

function renderAsset(item) {
  return `
    <div class="asset-item" data-asset-id="${escapeHtml(item.id)}" role="button" tabindex="0">
      <div class="media-frame square">${renderMedia(item, "asset")}${renderKindBadge(item)}${renderFavoriteBadge(item)}</div>
      <div>
        <div class="asset-name" title="${escapeHtml(item.filename)}">${escapeHtml(item.filename)}</div>
        <div class="asset-meta">${assetMetaText(item)}</div>
      </div>
    </div>
  `;
}

function renderResult(item) {
  const explanation = item.explanation?.length
    ? `<div class="explain-list">${item.explanation.map((reason) => `<span>${escapeHtml(reason)}</span>`).join("")}</div>`
    : "";
  const tagLine = item.tag_hits?.length
    ? `<div class="matched">标签命中：${item.tag_hits.map(escapeHtml).join("、")}</div>`
    : "";
  const ocrLine = item.ocr_hits?.length
    ? `<div class="matched">文本命中：${item.ocr_hits.map(escapeHtml).join("、")}</div>`
    : "";
  const matched = item.metadata_hits?.length
    ? `<div class="matched">元数据命中：${item.metadata_hits.map(escapeHtml).join("、")}</div>`
    : `<div class="matched">主要依据：CLIP 语义相似</div>`;
  return `
    <article class="card" data-asset-id="${escapeHtml(item.id)}" role="button" tabindex="0">
      <div class="thumb media-frame">${renderMedia(item, "result")}${renderKindBadge(item)}${renderFavoriteBadge(item)}</div>
      <div class="card-body">
        <div class="asset-name" title="${escapeHtml(item.filename)}">${escapeHtml(item.filename)}</div>
        <div class="asset-meta">${assetMetaText(item)}</div>
        <div class="score">排序置信 ${Math.round((item.display_score || item.score) * 100)}%</div>
          <div class="score-detail">原始 CLIP ${Math.round(item.raw_clip_score * 1000) / 1000} · 标签 ${Math.round(item.tag_score * 100)}% · 文本 ${Math.round(item.ocr_score * 100)}% · 关系 ${Math.round((item.relation_score || 0) * 100)}% · margin ${Math.round((item.relation_margin || 0) * 1000) / 1000}</div>
        <div class="score-detail">最佳提示：${escapeHtml(item.best_prompt || "-")}</div>
        ${tagLine}
        ${ocrLine}
        ${matched}
        ${explanation}
      </div>
    </article>
  `;
}

function renderMedia(item, mode) {
  if (mode !== "detail") {
    return `<img src="${thumbnailUrl(item)}" alt="" loading="lazy" decoding="async" onerror="this.classList.add('thumb-missing'); this.removeAttribute('src');" />`;
  }
  if (item.kind === "video") {
    const controls = mode === "detail" ? "controls" : "";
    const preload = mode === "detail" ? "metadata" : "none";
    return `<video src="${item.url}" muted ${controls} preload="${preload}" title="${escapeHtml(item.filename)}" onclick="event.stopPropagation()"></video>`;
  }
  if (item.kind === "document") {
    return `
      <div class="document-preview">
        <img src="${thumbnailUrl(item)}" alt="" />
        <a href="${item.url}" target="_blank" rel="noopener">打开文档</a>
      </div>
    `;
  }
  return `<img src="${item.url}" alt="${escapeHtml(item.filename)}" loading="lazy" />`;
}

function renderKindBadge(item) {
  if (item.kind === "video") return `<span class="kind-badge">VIDEO</span>`;
  if (item.kind === "document") return `<span class="kind-badge">DOC</span>`;
  return "";
}

function assetMetaText(item) {
  const kind = item.kind || "image";
  const size = kind === "document" ? "文档" : `${item.width} x ${item.height}`;
  return `${libraryLabel(item.library)} · ${kind} · ${size}`;
}

function renderFavoriteBadge(item) {
  return item.favorite ? `<span class="favorite-badge">精选</span>` : "";
}

function thumbnailUrl(item) {
  return `/thumbs/${encodeURIComponent(item.id)}.jpg`;
}

async function openAssetDetail(assetId) {
  detailStatus.textContent = "";
  try {
    const response = await fetch(`/api/asset?id=${encodeURIComponent(assetId)}`);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "详情读取失败");
    currentDetail = payload.asset;
    renderAssetDetail(currentDetail);
    if (!assetDialog.open) assetDialog.showModal();
  } catch (error) {
    statusBox.textContent = `详情读取失败：${error.message}`;
  }
}

function renderAssetDetail(asset) {
  detailTitle.textContent = asset.filename;
  detailKind.textContent = asset.kind === "document" ? "document · 文档" : `${asset.kind} · ${asset.width} x ${asset.height}`;
  detailMedia.innerHTML = renderMedia(asset, "detail");
  detailMeta.innerHTML = `
    ${renderMetaItem("ID", asset.id)}
    ${renderMetaItem("素材库", libraryLabel(asset.library))}
    ${renderMetaItem("类型", asset.kind)}
      ${renderMetaItem("尺寸", asset.kind === "document" ? "文档" : `${asset.width} x ${asset.height}`)}
    ${renderMetaItem("模型", asset.vector_model)}
    ${renderMetaItem("入库时间", asset.created_at)}
  `;
  manualTagsInput.value = (asset.manual_tags || []).join(", ");
    if (favoriteAsset) {
      favoriteAsset.textContent = asset.favorite ? "取消精选" : "设为精选";
      favoriteAsset.classList.toggle("secondary-button", Boolean(asset.favorite));
    }
    renderHealthPanel(asset.health);
    tagRows.innerHTML = asset.tag_rows?.length
      ? asset.tag_rows.map(renderTagPill).join("")
      : `<span class="muted">暂无标签</span>`;
    faceRows.innerHTML = asset.faces?.length
      ? asset.faces.map(renderFaceRow).join("")
      : `<span class="muted">暂无人物</span>`;
    renderArchiveSuggestions(asset.archive_suggestions || []);
    renderTextSignals(asset);
    renderSimilarAssets(asset.similar_assets || []);
  }

  function renderHealthPanel(health) {
    if (!healthPanel) return;
    if (!health) {
      healthPanel.innerHTML = "";
      return;
    }
    const issues = (health.issues || []).map((item) => `<li>${escapeHtml(item)}</li>`).join("");
    const actions = (health.actions || []).map((item) => `<span>${escapeHtml(item)}</span>`).join("");
    healthPanel.innerHTML = `
      <div class="health-head">
        <strong>素材健康度 ${Number(health.score || 0)}%</strong>
        <span class="health-badge ${escapeHtml(health.level || "")}">${escapeHtml(health.label || "")}</span>
      </div>
      <ul>${issues}</ul>
      ${actions ? `<div class="health-actions">${actions}</div>` : ""}
    `;
  }

  function renderArchiveSuggestions(items) {
    if (!archiveRows) return;
    archiveRows.innerHTML = items.length
      ? items.map(renderArchiveSuggestion).join("")
      : `<span class="muted">暂无归档建议</span>`;
  }

  function renderArchiveSuggestion(item) {
    const confidence = Math.round(Number(item.confidence || 0) * 100);
    const reasons = (item.reasons || []).map((reason) => `<li>${escapeHtml(reason)}</li>`).join("");
    const tags = (item.action_tags || []).map((tag) => `<span>${escapeHtml(tag)}</span>`).join("");
    return `
      <div class="archive-suggestion">
        <div>
          <strong>${escapeHtml(item.title || "归档建议")}</strong>
          <span>${escapeHtml(item.folder || "-")} · ${confidence}%</span>
        </div>
        <ul>${reasons}</ul>
        <div class="archive-tags">${tags}</div>
        <button type="button" onclick="applyArchiveSuggestion('${escapeHtml(item.key)}')">采纳建议</button>
      </div>
    `;
  }

function renderSimilarAssets(items) {
  if (!similarRows) return;
  similarRows.innerHTML = items.length
    ? items.map(renderSimilarAsset).join("")
    : `<span class="muted">暂无相似素材</span>`;
}

function renderSimilarAsset(item) {
  return `
    <div class="similar-card" data-asset-id="${escapeHtml(item.id)}" onclick="openAssetDetail('${escapeHtml(item.id)}')">
      <div class="media-frame square">${renderMedia(item, "asset")}${renderKindBadge(item)}${renderFavoriteBadge(item)}</div>
      <strong title="${escapeHtml(item.filename)}">${escapeHtml(item.filename)}</strong>
      <span>相似度 ${Math.round(Number(item.similarity || 0) * 100)}%</span>
    </div>
  `;
}

function renderTextSignals(asset) {
  const summary = asset.text_summary || {};
  const signalRows = asset.text_signals || [];
  const visualText = asset.visual_text || asset.ocr_text || "";
  const audioText = asset.asr_text || "";

  if (asrText) asrText.textContent = audioText || "暂无音频转写";
  if (ocrText) ocrText.textContent = visualText || "暂无画面文字";

  if (textSummaryRows) {
    const summaryItems = [
      renderTextSummaryCard("音频转写", summary.asr),
      renderTextSummaryCard("画面/字幕 OCR", summary.visual),
      renderTextSummaryCard("人工文本", summary.manual),
    ].join("");
    textSummaryRows.innerHTML = summaryItems;
  }

  if (textSignalRows) {
    textSignalRows.innerHTML = signalRows.length
      ? signalRows.map(renderTextSignalRow).join("")
      : `<span class="muted">暂无文本信号</span>`;
  }
}

function renderTextSummaryCard(label, item = {}) {
  const count = Number(item.count || 0);
  const engines = (item.engines || []).join(", ") || "-";
  const summary = item.summary || "暂无摘要";
  return `
    <div class="text-summary-card ${count ? "" : "empty-summary"}">
      <strong>${escapeHtml(label)}</strong>
      <span>${count} 条 · ${escapeHtml(engines)}</span>
      <p>${escapeHtml(summary)}</p>
    </div>
  `;
}

function renderTextSignalRow(signal) {
  const typeLabel = textTypeLabel(signal.text_type);
  const confidence = signal.confidence === null || signal.confidence === undefined
    ? "-"
    : `${Math.round(Number(signal.confidence || 0) * 100)}%`;
  return `
    <div class="text-signal-row">
      <div>
        <strong>${escapeHtml(typeLabel)}</strong>
        <span>${escapeHtml(signal.engine || "-")} · ${confidence} · ${escapeHtml(signal.created_at || "")}</span>
      </div>
      <p>${escapeHtml(signal.summary || signal.text || "")}</p>
    </div>
  `;
}

function textTypeLabel(value) {
  return {
    asr: "音频转写",
    ocr: "画面 OCR",
      subtitle_ocr: "字幕 OCR",
      manual_text: "人工文本",
      document_text: "文档正文",
    }[value] || value || "文本";
  }

function renderMetaItem(label, value) {
  return `
    <div>
      <span>${escapeHtml(label)}</span>
      <strong title="${escapeHtml(value)}">${escapeHtml(value)}</strong>
    </div>
  `;
}

function renderTagPill(row) {
  const confidence = Math.round(Number(row.confidence || 0) * 100);
  return `<span class="tag-pill" title="${escapeHtml(row.source)}">${escapeHtml(row.name)}<small>${escapeHtml(row.source)} · ${confidence}%</small></span>`;
}

function renderFaceRow(face) {
  return `
    <div class="face-row">
      <strong>${escapeHtml(face.person_id)}</strong>
      <span>bbox ${escapeHtml(face.bbox.join(", "))}</span>
      <span>quality ${Math.round(Number(face.quality || 0) * 100)}%</span>
      <button type="button" onclick="removeAssetPerson('${escapeHtml(face.person_id)}')">移除此人物</button>
    </div>
  `;
}

async function removeAssetPerson(personId) {
  if (!currentDetail) return;
  detailStatus.textContent = `正在从当前素材移除 ${personId}...`;
  try {
    const response = await fetch("/api/remove-asset-person", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({asset_id: currentDetail.id, person_id: personId, library: "personal"}),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "移除失败");
    assetsLoaded = false;
    personsLoaded = false;
    await openAssetDetail(currentDetail.id);
    if (isViewActive("peopleView")) await loadPersons();
  } catch (error) {
    detailStatus.textContent = `移除失败：${error.message}`;
  }
}

async function applyArchiveSuggestion(suggestionKey) {
  if (!currentDetail) return;
  detailStatus.textContent = "正在采纳 Agent 归档建议...";
  try {
    const response = await fetch("/api/apply-archive-suggestion", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({asset_id: currentDetail.id, suggestion_key: suggestionKey}),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "采纳失败");
    detailStatus.textContent = "已写入自动归档标签";
    assetsLoaded = false;
    tagsLoaded = false;
    statusLoaded = false;
    await openAssetDetail(currentDetail.id);
    if (isViewActive("tagView")) await loadTags(selectedTag);
  } catch (error) {
    detailStatus.textContent = `采纳失败：${error.message}`;
  }
}

window.removeAssetPerson = removeAssetPerson;
window.applyArchiveSuggestion = applyArchiveSuggestion;
window.applyPersonFilter = applyPersonFilter;
window.focusPersonName = focusPersonName;
window.openAssetDetail = openAssetDetail;

async function saveManualTagEdits() {
  if (!currentDetail) return;
  const tags = manualTagsInput.value
    .split(/[\s,，、]+/)
    .map((item) => item.trim())
    .filter(Boolean);
  detailStatus.textContent = "正在保存...";
  try {
    const response = await fetch("/api/asset-tags", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        asset_id: currentDetail.id,
        tags,
        source: "manual",
        confidence: 1.0,
        replace_source: true,
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "保存失败");
    detailStatus.textContent = "已保存";
    await openAssetDetail(currentDetail.id);
    statusLoaded = false;
  } catch (error) {
    detailStatus.textContent = `保存失败：${error.message}`;
  }
}

async function toggleCurrentFavorite() {
  if (!currentDetail) return;
  const nextFavorite = !currentDetail.favorite;
  detailStatus.textContent = nextFavorite ? "正在设为精选..." : "正在取消精选...";
  try {
    const response = await fetch("/api/favorite", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({asset_id: currentDetail.id, favorite: nextFavorite}),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "精选状态保存失败");
    currentDetail.favorite = nextFavorite;
    detailStatus.textContent = nextFavorite ? "已设为精选" : "已取消精选";
    assetsLoaded = false;
    tagsLoaded = false;
    currentResults = currentResults.map((item) =>
      item.id === currentDetail.id ? {...item, favorite: nextFavorite} : item
    );
    renderSearchResults();
    await openAssetDetail(currentDetail.id);
    if (isViewActive("libraryView")) await loadAssets();
    if (isViewActive("tagView")) await loadTags(selectedTag);
  } catch (error) {
    detailStatus.textContent = `精选状态保存失败：${error.message}`;
  }
}

async function deleteCurrentAsset() {
  if (!currentDetail) return;
  const confirmed = window.confirm(`确定删除素材 ${currentDetail.filename} 吗？这会从素材库和本地上传目录移除它。`);
  if (!confirmed) return;
  detailStatus.textContent = "正在删除...";
  try {
    const response = await fetch("/api/delete-asset", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({asset_id: currentDetail.id, delete_file: true}),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "删除失败");
    assetDialog.close();
    currentDetail = null;
    assetsLoaded = false;
    personsLoaded = false;
    statusLoaded = false;
    currentResults = currentResults.filter((item) => item.id !== payload.result.asset_id);
    renderSearchResults();
    if (isViewActive("libraryView")) await loadAssets();
    if (isViewActive("peopleView")) await loadPersons();
    statusBox.textContent = "素材已删除。";
  } catch (error) {
    detailStatus.textContent = `删除失败：${error.message}`;
  }
}

function renderDeferred(conditions) {
  const entries = Object.entries(conditions || {});
  if (!entries.length) return "无";
  return entries
    .map(([key, values]) => `${labelOf(key)}：${values.map(escapeHtml).join("、")}`)
    .join("；");
}

function labelOf(key) {
  return {
    people: "人物",
    locations: "地点",
    scenes: "场景",
    time_words: "时间词",
    time: "时间",
    location: "地点",
    scene: "场景",
    media: "媒介",
  }[key] || key;
}

function libraryLabel(value) {
  return value === "personal" ? "个人库" : "公共库";
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  })[char]);
}

function cssEscape(value) {
  if (window.CSS?.escape) return CSS.escape(value);
  return String(value).replace(/["\\]/g, "\\$&");
}

renderSearchResults();
