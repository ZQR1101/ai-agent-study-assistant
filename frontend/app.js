const API_BASE_URL = "http://127.0.0.1:8000"
const HISTORY_LIMIT = 6
const CONVERSATIONS_KEY = "aiStudyAssistant.conversations.v1"
const CARD_LIBRARY_KEY = "aiStudyAssistant.cardLibrary.v1"

let currentSessionId = createSessionId()
let chatHistory = []
let conversations = loadConversations()
let cardLibrary = loadCardLibrary()
let knowledgeFiles = []
let knowledgeLoaded = false
let activeKnowledgeObjectUrl = ""

const MODE_LABELS = {
    auto: "自动规划",
    chat: "普通问答",
    rag: "知识库问答",
    explain: "概念解释",
    summarize: "内容总结",
    quiz: "自动出题",
    learn: "学习模式",
    langgraph: "LangGraph",
}


function createSessionId() {
    if (window.crypto && window.crypto.randomUUID) {
        return window.crypto.randomUUID()
    }

    return `session-${Date.now()}-${Math.random().toString(16).slice(2)}`
}


function getElement(id) {
    return document.getElementById(id)
}


function escapeHtml(value) {
    return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;")
}


function clampNumber(value, fallback, min, max) {
    const number = Number(value)

    if (Number.isNaN(number)) {
        return fallback
    }

    return Math.min(Math.max(number, min), max)
}


function loadConversations() {
    try {
        const value = JSON.parse(localStorage.getItem(CONVERSATIONS_KEY) || "[]")
        return Array.isArray(value) ? value : []
    } catch (error) {
        return []
    }
}


function saveConversations() {
    localStorage.setItem(CONVERSATIONS_KEY, JSON.stringify(conversations.slice(0, 20)))
}


function loadCardLibrary() {
    try {
        const value = JSON.parse(localStorage.getItem(CARD_LIBRARY_KEY) || "[]")
        return Array.isArray(value) ? value : []
    } catch (error) {
        return []
    }
}


function saveCardLibrary() {
    localStorage.setItem(CARD_LIBRARY_KEY, JSON.stringify(cardLibrary.slice(0, 200)))
}


function getConversationTitle(messages) {
    const firstUserMessage = messages.find(message => message.role === "user")
    const title = String(firstUserMessage?.content || "新学习对话").trim()
    return title.length > 22 ? `${title.slice(0, 22)}...` : title
}


function persistCurrentConversation() {
    if (chatHistory.length === 0) {
        return
    }

    const now = new Date().toISOString()
    const nextConversation = {
        id: currentSessionId,
        title: getConversationTitle(chatHistory),
        updatedAt: now,
        messages: chatHistory.map(message => ({ ...message })),
    }
    const existingIndex = conversations.findIndex(item => item.id === currentSessionId)

    if (existingIndex >= 0) {
        conversations.splice(existingIndex, 1)
    }

    conversations.unshift(nextConversation)
    saveConversations()
    renderConversationHistory()
}


function renderConversationHistory() {
    const panel = getElement("conversationHistoryPanel")

    if (!panel) {
        return
    }

    if (conversations.length === 0) {
        panel.innerHTML = emptyState("暂无历史对话", "发送消息后，当前会话会自动保存在这里。")
        return
    }

    panel.innerHTML = conversations.slice(0, 8).map(conversation => {
        const updatedAt = new Date(conversation.updatedAt).toLocaleString()
        const activeClass = conversation.id === currentSessionId ? " active" : ""

        return `
            <button class="conversation-item${activeClass}" type="button" data-conversation-id="${escapeHtml(conversation.id)}">
                <strong>${escapeHtml(conversation.title)}</strong>
                <span>${escapeHtml(updatedAt)}</span>
            </button>
        `
    }).join("")
}


function renderWelcomeMessage(message = "把一个主题、问题或材料片段发给我。我会先组织解释，再把可核对的来源、学习计划和复习卡片整理在右侧。") {
    getElement("chatBox").innerHTML = `
        <article class="ai-message welcome-message">
            <div class="message-meta">
                <span>助手</span>
                <span>学习画布已就绪</span>
            </div>
            <div class="answer">${escapeHtml(message)}</div>
        </article>
    `
}


function startNewConversation() {
    currentSessionId = createSessionId()
    chatHistory = []
    getElement("userInput").value = ""
    getElement("loadingText").style.display = "none"
    renderWelcomeMessage("这是一个新对话。输入新的学习主题，我会重新检索资料、组织解释并生成可复习的学习成果。")
    resetInsights()
    renderConversationHistory()
}


function restoreConversation(conversationId) {
    const conversation = conversations.find(item => item.id === conversationId)

    if (!conversation) {
        return
    }

    currentSessionId = conversation.id
    chatHistory = conversation.messages.map(message => ({ ...message }))
    renderChatMessages()
    restoreInsightsFromHistory()
    renderConversationHistory()
}


function renderChatMessages() {
    if (chatHistory.length === 0) {
        renderWelcomeMessage()
        return
    }

    getElement("chatBox").innerHTML = chatHistory.map(message => {
        const isAssistant = message.role === "assistant"
        const messageClass = isAssistant ? "ai-message" : "user-message"
        const roleLabel = isAssistant ? "助手" : "你"
        const response = isAssistant ? message.response : null
        const contentHtml = response
            ? renderAnswer(response)
            : `<div>${escapeHtml(message.content)}</div>`

        return `
            <article class="${messageClass}">
                <div class="message-meta">
                    <span>${roleLabel}</span>
                </div>
                ${contentHtml}
            </article>
        `
    }).join("")
}


function getRecentHistory() {
    return chatHistory
        .slice(-HISTORY_LIMIT)
        .map(message => ({
            role: message.role,
            content: message.content,
        }))
}


function addHistoryMessage(role, content, extra = {}) {
    const cleanContent = String(content || "").trim()

    if (!cleanContent) {
        return
    }

    chatHistory.push({ role, content: cleanContent, ...extra })

    if (chatHistory.length > HISTORY_LIMIT * 2) {
        chatHistory = chatHistory.slice(-HISTORY_LIMIT * 2)
    }

    persistCurrentConversation()
}


function getResponseSnapshot(data) {
    return {
        answer: data.answer || "",
        mode: data.mode || "",
        model: data.model || "",
        sources: Array.isArray(data.sources) ? data.sources : [],
        plan: Array.isArray(data.plan) ? data.plan : [],
        trace: Array.isArray(data.trace) ? data.trace : [],
        flashcards: Array.isArray(data.flashcards) ? data.flashcards : [],
        runtime_info: hasRuntimeInfo(data.runtime_info) ? data.runtime_info : {},
    }
}


function addAssistantHistoryMessage(data, requestMessage = "") {
    addHistoryMessage("assistant", data.answer || "", {
        response: getResponseSnapshot(data),
        requestMessage,
    })
}


function updateActiveModeUI(mode) {
    getElement("activeModeLabel").textContent = MODE_LABELS[mode] || mode

    document.querySelectorAll(".mode-card").forEach(button => {
        button.classList.toggle("active", button.dataset.modeChoice === mode)
    })
}


function updateUseRagState() {
    const mode = getElement("modeSelect").value
    const useRagInput = getElement("useRagInput")

    if (mode === "rag") {
        useRagInput.checked = true
        useRagInput.disabled = true
    } else {
        useRagInput.disabled = false
    }

    updateActiveModeUI(mode)
}


function updateRuntimeModeState(changedMode = "") {
    const useAgentInput = getElement("useAgentInput")
    const useLangGraphInput = getElement("useLangGraphInput")

    if (!useAgentInput || !useLangGraphInput) {
        return
    }

    if (changedMode === "langgraph" && useLangGraphInput.checked) {
        useAgentInput.checked = false
    } else if (changedMode === "agent" && useAgentInput.checked) {
        useLangGraphInput.checked = false
    }

    if (!useAgentInput.checked && !useLangGraphInput.checked) {
        useAgentInput.checked = true
    }
}


function setMode(mode) {
    getElement("modeSelect").value = mode
    updateUseRagState()
}


function buildChatRequest(modeOverride) {
    const message = getElement("userInput").value.trim()
    const mode = "auto"

    return {
        message,
        mode,
        model: getElement("modelSelect").value,
        temperature: clampNumber(getElement("temperatureInput").value, 0.7, 0, 2),
        use_agent: getElement("useAgentInput")?.checked ?? true,
        use_rag: getElement("useRagInput").checked,
        use_langgraph: getElement("useLangGraphInput")?.checked || false,
        top_k: Math.round(clampNumber(getElement("topKInput").value, 3, 1, 10)),
        session_id: currentSessionId,
        history: getRecentHistory(),
    }
}


function clearConversation() {
    startNewConversation()
}


function appendUserMessage(message) {
    const displayMessage = compactDisplayText(message)

    getElement("chatBox").insertAdjacentHTML("beforeend", `
        <article class="user-message">
            <div class="message-meta">
                <span>你</span>
            </div>
            <div>${escapeHtml(displayMessage)}</div>
        </article>
    `)
}


function appendErrorMessage(message) {
    getElement("chatBox").insertAdjacentHTML("beforeend", `
        <article class="ai-message error-message">
            <div class="message-meta">
                <span>请求失败</span>
            </div>
            <div>${escapeHtml(message)}</div>
        </article>
    `)
}


function flattenTraceItems(trace) {
    if (!Array.isArray(trace)) {
        return []
    }

    return trace.flatMap(block => {
        if (typeof block === "string") {
            return [block]
        }

        if (block && Array.isArray(block.items)) {
            return block.items
        }

        return []
    })
}


function traceIncludes(trace, text) {
    return flattenTraceItems(trace).some(item => String(item).includes(text))
}


function compactDisplayText(value) {
    return String(value || "")
        .replace(/[ \t]+\n/g, "\n")
        .replace(/\n[ \t]+/g, "\n")
        .replace(/\n{3,}/g, "\n\n")
        .trim()
}


function renderAnswer(data) {
    const rawAnswer = compactDisplayText(data.answer)

    if (data.mode !== "learn") {
        return `<div class="answer">${escapeHtml(rawAnswer)}</div>`
    }

    const trace = data.trace || []
    const ragPassed = traceIncludes(trace, "RAG 是否通过阈值：是")
    const ragFailed = traceIncludes(trace, "RAG 是否通过阈值：否")
    const fallbackUsed = traceIncludes(trace, "是否启用 fallback：是")
    const ragDisabled = traceIncludes(trace, "use_rag：False")
        || traceIncludes(trace, "use_rag：false")

    let title = "学习内容"
    let noteHtml = ""

    if (ragPassed) {
        title = "知识库学习内容"
    } else if (ragFailed && fallbackUsed) {
        title = "普通模型学习内容"
        noteHtml = `
            <div class="fallback-note">
                知识库未找到可靠相关内容，本部分由普通模型生成。
            </div>
        `
    } else if (ragDisabled) {
        title = "学习内容"
    }

    const answerBody = rawAnswer.replace(/^知识内容：\s*/, "")

    return `
        <div class="answer">
            <strong>${title}</strong>
            ${noteHtml}
            <div>${escapeHtml(answerBody)}</div>
        </div>
    `
}


function getExecutionModeLabel(mode) {
    if (mode === "langgraph") {
        return "\u6267\u884c\u65b9\u5f0f\uff1aLangGraph \u5de5\u4f5c\u6d41"
    }

    if (mode === "agent") {
        return "\u6267\u884c\u65b9\u5f0f\uff1aAgent Planner + Executor"
    }

    return mode ? `\u6267\u884c\u65b9\u5f0f\uff1a${MODE_LABELS[mode] || mode}` : ""
}


function appendChatResponse(data, requestMessage = "") {
    const answerHtml = renderAnswer(data)
    const modeLabel = MODE_LABELS[data.mode] || data.mode || ""
    const executionModeLabel = getExecutionModeLabel(data.mode)

    getElement("chatBox").insertAdjacentHTML("beforeend", `
        <article class="ai-message">
            <div class="response-meta">
                ${executionModeLabel ? `<span>${escapeHtml(executionModeLabel)}</span>` : ""}
                <span>模式：${escapeHtml(modeLabel)}</span>
                <span>模型：${escapeHtml(data.model || "")}</span>
            </div>
            ${answerHtml}
        </article>
    `)
}


function emptyState(title, text) {
    return `
        <div class="empty-state">
            <strong>${escapeHtml(title)}</strong>
            <span>${escapeHtml(text)}</span>
        </div>
    `
}


function resetInsights() {
    getElement("sourcesPanel").innerHTML = emptyState("等待一次知识检索", "启用 RAG 后，这里会显示命中文档和相关片段。")
    getElement("planPanel").innerHTML = emptyState("等待 Agent 计划", "启用 Agent 后，复杂学习任务会拆成工具步骤。")
    getElement("flashcardsPanel").innerHTML = emptyState("还没有记忆卡片", "请求生成 flashcard 后，可以在这里翻看和下载 PNG。")
    getElement("tracePanel").innerHTML = emptyState("暂无执行路径", "这里会记录检索、规划、fallback 等运行细节。")
}


function getLatestAssistantResponse() {
    for (let index = chatHistory.length - 1; index >= 0; index -= 1) {
        const message = chatHistory[index]
        if (message.role === "assistant" && message.response) {
            return message.response
        }
    }

    return null
}


function restoreInsightsFromHistory() {
    if (getConversationTurns().length > 0) {
        renderConversationInsights()
    } else {
        resetInsights()
    }
}


function getConversationTurns() {
    const turns = []
    let pendingQuestion = ""

    chatHistory.forEach(message => {
        if (message.role === "user") {
            pendingQuestion = message.content || ""
            return
        }

        if (message.role === "assistant" && message.response) {
            turns.push({
                question: message.requestMessage || pendingQuestion || "本轮问题",
                response: message.response,
            })
            pendingQuestion = ""
        }
    })

    return turns
}


function getTurnTitle(turn, index) {
    return `${index + 1}. ${turn.question || "本轮问题"}`
}


function renderConversationSourcesPanel(turns) {
    if (turns.length === 0) {
        return emptyState("本次没有来源", "如果需要引用知识库，请选择知识库问答或打开 RAG 检索。")
    }

    return turns.map((turn, turnIndex) => {
        const sources = Array.isArray(turn.response.sources) ? turn.response.sources : []
        const body = sources.length > 0
            ? sources.map((source, sourceIndex) => {
                const sourceName = typeof source === "string" ? source : source.source || "未知来源"
                const score = typeof source === "string" || source.score === null || source.score === undefined
                    ? ""
                    : `相似度 ${Number(source.score).toFixed(4)}`
                const snippet = typeof source === "string" ? "" : source.text || source.snippet || ""

                return `
                    <li>
                        <strong>${sourceIndex + 1}. ${escapeHtml(sourceName)}</strong>
                        ${score ? `<span>${escapeHtml(score)}</span>` : ""}
                        ${snippet ? `<span>${escapeHtml(snippet)}</span>` : ""}
                    </li>
                `
            }).join("")
            : `<li><span>本轮没有来源。若启用 RAG 且命中知识库，来源会显示在这里。</span></li>`

        return `
            <article class="turn-panel-card">
                <h3>${escapeHtml(getTurnTitle(turn, turnIndex))}</h3>
                <ul class="turn-detail-list">${body}</ul>
            </article>
        `
    }).join("")
}


function renderConversationPlanPanel(turns) {
    if (turns.length === 0) {
        return emptyState("本次没有学习计划", "发送问题后，每轮的 Agent 或 LangGraph 计划会显示在这里。")
    }

    return turns.map((turn, turnIndex) => {
        const plan = Array.isArray(turn.response.plan) ? turn.response.plan : []
        const body = plan.length > 0
            ? plan.map((step, stepIndex) => `
                <li>
                    <strong>${stepIndex + 1}. ${escapeHtml(step.tool || "unknown")}</strong>
                    ${step.input ? `<span>输入：${escapeHtml(step.input)}</span>` : ""}
                    ${step.reason ? `<span>原因：${escapeHtml(step.reason)}</span>` : ""}
                </li>
            `).join("")
            : `<li><span>本轮没有返回学习计划。</span></li>`

        return `
            <article class="turn-panel-card">
                <h3>${escapeHtml(getTurnTitle(turn, turnIndex))}</h3>
                <ul class="turn-detail-list">${body}</ul>
            </article>
        `
    }).join("")
}


function renderConversationFlashcardsPanel(turns) {
    if (turns.length === 0) {
        return emptyState("本次没有卡片", "让 Agent 或 LangGraph 生成卡片后，会按每轮问题保存在这里。")
    }

    let nextIndex = 1
    return turns.map((turn, turnIndex) => {
        const cards = Array.isArray(turn.response.flashcards) ? turn.response.flashcards : []
        const body = cards.length > 0
            ? renderFlashcardsPanel(cards, {
                startIndex: nextIndex,
                showTopic: true,
            })
            : emptyState("本轮没有卡片", "如果请求生成记忆卡片，结果会显示在这里。")
        nextIndex += cards.length

        return `
            <article class="turn-panel-card">
                <h3>${escapeHtml(getTurnTitle(turn, turnIndex))}</h3>
                ${body}
            </article>
        `
    }).join("")
}


function renderConversationTracePanel(turns) {
    if (turns.length === 0) {
        return emptyState("本次没有执行路径", "每轮对话的执行路径会保存在这里。")
    }

    return turns.map((turn, turnIndex) => {
        const runtimeInfo = turn.response.runtime_info || {}
        const path = getExecutionPathForResponse(turn.response)
        const pathHtml = path.length > 0
            ? `
                <div class="execution-path-line">
                    ${path.map(node => `<span>${escapeHtml(node)}</span>`).join("<b>→</b>")}
                </div>
            `
            : `<div class="execution-path-line muted">暂无执行路径摘要</div>`
        const runtimeSummaryHtml = hasRuntimeInfo(runtimeInfo)
            ? `
                <dl class="execution-summary">
                    <dt>执行引擎</dt>
                    <dd>${escapeHtml(runtimeInfo.runtime || "langgraph")}</dd>
                    <dt>节点数量</dt>
                    <dd>${escapeHtml(runtimeInfo.node_count ?? path.length)}</dd>
                    <dt>Finalizer</dt>
                    <dd>${runtimeInfo.finalizer_used ? "已启用" : "未启用"}</dd>
                    ${runtimeInfo.error ? `
                        <dt>Error</dt>
                        <dd class="runtime-error-text">${escapeHtml(runtimeInfo.error)}</dd>
                    ` : ""}
                </dl>
            `
            : ""
        const detailHtml = renderTracePanel(turn.response.trace)

        return `
            <article class="turn-panel-card">
                <h3>${escapeHtml(getTurnTitle(turn, turnIndex))}</h3>
                ${pathHtml}
                ${runtimeSummaryHtml}
                <details class="trace-debug-details">
                    <summary>调试详情</summary>
                    ${detailHtml}
                </details>
            </article>
        `
    }).join("")
}


function getExecutionPathForResponse(response) {
    const runtimeInfo = response.runtime_info || {}

    if (Array.isArray(runtimeInfo.graph_path) && runtimeInfo.graph_path.length > 0) {
        return runtimeInfo.graph_path
    }

    const plan = Array.isArray(response.plan) ? response.plan : []
    if (plan.length > 0) {
        return ["agent", ...plan.map(step => step.tool || "tool")]
    }

    const traceItems = flattenTraceItems(response.trace)
    const finalMode = traceItems.find(item => String(item).includes("最终执行的模式"))
    if (finalMode) {
        const mode = String(finalMode).split(/[：:]/).pop().trim()
        return mode ? [mode] : []
    }

    return []
}


function renderConversationInsights() {
    const turns = getConversationTurns()
    getElement("sourcesPanel").innerHTML = renderConversationSourcesPanel(turns)
    getElement("planPanel").innerHTML = renderConversationPlanPanel(turns)
    getElement("flashcardsPanel").innerHTML = renderConversationFlashcardsPanel(turns)
    getElement("tracePanel").innerHTML = renderConversationTracePanel(turns)
}


function formatFileSize(bytes) {
    if (bytes < 1024) {
        return `${bytes} B`
    }

    if (bytes < 1024 * 1024) {
        return `${(bytes / 1024).toFixed(1)} KB`
    }

    return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}


function buildKnowledgeFileUrl(file) {
    if (file && file.url) {
        return `${API_BASE_URL}${file.url}`
    }

    return `${API_BASE_URL}/knowledge-files/${encodeURIComponent(file.name || "")}`
}


function buildKnowledgeContentUrl(file) {
    return `${API_BASE_URL}/knowledge-files/${encodeURIComponent(file.name || "")}/content`
}


function clearKnowledgeObjectUrl() {
    if (activeKnowledgeObjectUrl) {
        URL.revokeObjectURL(activeKnowledgeObjectUrl)
        activeKnowledgeObjectUrl = ""
    }
}


async function openKnowledgeFile(index) {
    const file = knowledgeFiles[index]
    const viewer = getElement("knowledgeViewerPanel")

    if (!file || !viewer) {
        return
    }

    clearKnowledgeObjectUrl()
    viewer.innerHTML = emptyState("正在打开文件", `${file.name} 加载中...`)

    try {
        const fileUrl = buildKnowledgeFileUrl(file)

        if (file.type === "pdf") {
            viewer.innerHTML = `
                <article class="knowledge-viewer-card">
                    <div class="knowledge-source-header">
                        <strong>${escapeHtml(file.name)}</strong>
                        <span>PDF · ${formatFileSize(Number(file.size || 0))}</span>
                    </div>
                    <div class="knowledge-actions">
                        <a class="ghost-link" href="${escapeHtml(fileUrl)}" target="_blank" rel="noopener noreferrer">新窗口打开</a>
                    </div>
                    <iframe class="knowledge-pdf-frame" src="${escapeHtml(fileUrl)}" title="${escapeHtml(file.name)}"></iframe>
                </article>
            `
            return
        }

        const response = await fetch(buildKnowledgeContentUrl(file))

        if (!response.ok) {
            throw new Error(`文本预览接口返回 ${response.status}`)
        }

        const payload = await response.json()
        const text = payload.content || ""
        viewer.innerHTML = `
            <article class="knowledge-viewer-card">
                <div class="knowledge-source-header">
                    <strong>${escapeHtml(file.name)}</strong>
                    <span>${escapeHtml(String(file.type || "text").toUpperCase())} · ${formatFileSize(Number(file.size || 0))}</span>
                </div>
                <pre class="knowledge-text-preview">${escapeHtml(text)}</pre>
            </article>
        `
    } catch (error) {
        viewer.innerHTML = emptyState("文件打开失败", error.message)
    }
}


async function loadKnowledgeLibrary(force = false) {
    const panel = getElement("knowledgeLibraryPanel")

    if (!panel || (knowledgeLoaded && !force)) {
        return
    }

    panel.innerHTML = emptyState("正在加载知识库", "正在读取 docs 目录文件...")

    try {
        const response = await fetch(`${API_BASE_URL}/knowledge-files`)

        if (!response.ok) {
            throw new Error(`知识库接口返回 ${response.status}`)
        }

        const data = await response.json()
        const files = Array.isArray(data.files) ? data.files : []
        knowledgeFiles = files
        knowledgeLoaded = true

        const status = document.querySelector(".knowledge-status")
        if (status) {
            status.innerHTML = `<span class="status-dot"></span>本地知识库 ${files.length} 个文件`
        }

        if (files.length === 0) {
            panel.innerHTML = emptyState("知识库为空", "请先上传 PDF，或把 md / txt / pdf 文件放入 docs 后刷新。")
            return
        }

        panel.innerHTML = files.map((file, index) => {
            return `
                <article class="knowledge-source">
                    <div class="knowledge-source-header">
                        <strong>${index + 1}. ${escapeHtml(file.name)}</strong>
                        <span>${escapeHtml(file.type.toUpperCase())} · ${formatFileSize(Number(file.size || 0))}</span>
                    </div>
                    <div class="knowledge-actions">
                        <button class="ghost-link open-knowledge-button" type="button" data-file-index="${index}">打开</button>
                    </div>
                </article>
            `
        }).join("")
    } catch (error) {
        panel.innerHTML = emptyState("知识库加载失败", error.message)
    }
}


function renderSourcesPanel(sources) {
    if (!Array.isArray(sources) || sources.length === 0) {
        return emptyState("本次没有来源", "如果需要引用知识库，请选择知识库问答或打开 RAG 检索。")
    }

    return sources.map((source, index) => {
        if (typeof source === "string") {
            return `
                <article class="source-card">
                    <strong>${index + 1}. ${escapeHtml(source)}</strong>
                </article>
            `
        }

        const sourceName = escapeHtml(source.source || "未知来源")
        const score = source.score === null || source.score === undefined
            ? "暂无分数"
            : Number(source.score).toFixed(4)
        const snippet = escapeHtml(source.text || source.snippet || "")

        return `
            <article class="source-card">
                <div class="knowledge-source-header">
                    <strong>${index + 1}. ${sourceName}</strong>
                    <span class="source-score">相似度 ${score}</span>
                </div>
                <div class="source-snippet">${snippet}</div>
            </article>
        `
    }).join("")
}


function renderPlanPanel(plan) {
    if (!Array.isArray(plan) || plan.length === 0) {
        return emptyState("本次没有 Agent 计划", "打开 Agent 规划后，复杂任务会显示工具执行步骤。")
    }

    const items = plan.map((step, index) => {
        const tool = escapeHtml(step.tool || "unknown")
        const input = escapeHtml(step.input || "")
        const reason = escapeHtml(step.reason || "")

        return `
            <li class="plan-item">
                <span class="plan-step">${index + 1}. ${tool}</span>
                <div>输入：${input}</div>
                ${reason ? `<div>原因：${reason}</div>` : ""}
            </li>
        `
    }).join("")

    return `<ol class="plan-list">${items}</ol>`
}


function renderTracePanel(trace) {
    if (!Array.isArray(trace) || trace.length === 0) {
        return emptyState("本次没有执行路径", "后端返回 trace 后，这里会显示系统处理过程。")
    }

    if (typeof trace[0] === "object" && !Array.isArray(trace[0])) {
        return trace.map(block => {
            const blockTitle = escapeHtml(block.title || "执行信息")
            const blockItems = Array.isArray(block.items) ? block.items : []
            const items = blockItems.map(item => `<li>${escapeHtml(item)}</li>`).join("")

            return `
                <article class="trace-block">
                    <strong>${blockTitle}</strong>
                    <ul>${items}</ul>
                </article>
            `
        }).join("")
    }

    const items = trace.map(item => `<li>${escapeHtml(item)}</li>`).join("")
    return `
        <article class="trace-block">
            <strong>执行信息</strong>
            <ul>${items}</ul>
        </article>
    `
}


function hasRuntimeInfo(runtimeInfo) {
    return runtimeInfo
        && typeof runtimeInfo === "object"
        && !Array.isArray(runtimeInfo)
        && Object.keys(runtimeInfo).length > 0
}


function renderRuntimeInfoPanel(runtimeInfo) {
    if (!hasRuntimeInfo(runtimeInfo)) {
        return ""
    }

    const graphPath = Array.isArray(runtimeInfo.graph_path) ? runtimeInfo.graph_path : []
    const toolCalls = Array.isArray(runtimeInfo.tool_calls) ? runtimeInfo.tool_calls : []
    const graphPathText = graphPath.length > 0 ? graphPath.join(" → ") : "无"
    const finalizerText = runtimeInfo.finalizer_used ? "已启用" : "未启用"
    const error = runtimeInfo.error ? String(runtimeInfo.error) : ""
    const toolCallsHtml = toolCalls.length > 0
        ? `
            <div class="runtime-tool-calls">
                <strong>工具调用</strong>
                ${toolCalls.map((call, index) => {
                    const success = Boolean(call.success)
                    const contextSources = Array.isArray(call.context_sources) && call.context_sources.length > 0
                        ? call.context_sources.join(" / ")
                        : "无"

                    return `
                        <article class="runtime-tool-call">
                            <div class="runtime-tool-header">
                                <strong>${index + 1}. ${escapeHtml(call.tool || "unknown")}</strong>
                                <span class="runtime-status ${success ? "success" : "failed"}">
                                    ${success ? "成功" : "失败"}
                                </span>
                            </div>
                            <dl class="runtime-detail-grid">
                                <dt>节点</dt>
                                <dd>${escapeHtml(call.node || "")}</dd>
                                <dt>工具说明</dt>
                                <dd>${escapeHtml(call.description || "")}</dd>
                                <dt>使用上下文</dt>
                                <dd>${call.used_context ? "是" : "否"}</dd>
                                <dt>上下文来源</dt>
                                <dd>${escapeHtml(contextSources)}</dd>
                                <dt>输出长度</dt>
                                <dd>${escapeHtml(call.output_length ?? 0)}</dd>
                                ${call.error ? `
                                    <dt>错误</dt>
                                    <dd class="runtime-error-text">${escapeHtml(call.error)}</dd>
                                ` : ""}
                            </dl>
                        </article>
                    `
                }).join("")}
            </div>
        `
        : ""

    return `
        <article class="runtime-info-panel">
            <div class="runtime-info-header">
                <strong>Runtime 信息</strong>
                <span class="runtime-engine">${escapeHtml(runtimeInfo.runtime || "")}</span>
            </div>
            <dl class="runtime-summary">
                <dt>执行引擎</dt>
                <dd>${escapeHtml(runtimeInfo.runtime || "")}</dd>
                <dt>图路径</dt>
                <dd class="runtime-graph-path">${escapeHtml(graphPathText)}</dd>
                <dt>节点数量</dt>
                <dd>${escapeHtml(runtimeInfo.node_count ?? graphPath.length)}</dd>
                <dt>Finalizer</dt>
                <dd>${finalizerText}</dd>
                ${error ? `
                    <dt>Error</dt>
                    <dd class="runtime-error-text">${escapeHtml(error)}</dd>
                ` : ""}
            </dl>
            ${toolCallsHtml}
        </article>
    `
}


function renderResponseArtifacts(data) {
    const sections = []
    const sources = Array.isArray(data.sources) ? data.sources : []
    const plan = Array.isArray(data.plan) ? data.plan : []
    const flashcards = Array.isArray(data.flashcards) ? data.flashcards : []
    const traceItems = flattenTraceItems(data.trace)
    const runtimePath = hasRuntimeInfo(data.runtime_info) && Array.isArray(data.runtime_info.graph_path)
        ? data.runtime_info.graph_path
        : []

    if (sources.length > 0) {
        const items = sources.slice(0, 5).map((source, index) => {
            const sourceName = typeof source === "string" ? source : source.source || "未知来源"
            const snippet = typeof source === "string" ? "" : source.snippet || source.text || ""

            return `
                <li>
                    <strong>${index + 1}. ${escapeHtml(sourceName)}</strong>
                    ${snippet ? `<span>${escapeHtml(snippet)}</span>` : ""}
                </li>
            `
        }).join("")

        sections.push(`
            <details class="message-artifact">
                <summary>本轮来源追踪（${sources.length}）</summary>
                <ul>${items}</ul>
            </details>
        `)
    }

    if (plan.length > 0) {
        const items = plan.map((step, index) => `
            <li>
                <strong>${index + 1}. ${escapeHtml(step.tool || "unknown")}</strong>
                <span>${escapeHtml(step.reason || step.input || "")}</span>
            </li>
        `).join("")

        sections.push(`
            <details class="message-artifact">
                <summary>本轮学习计划（${plan.length}）</summary>
                <ul>${items}</ul>
            </details>
        `)
    }

    if (flashcards.length > 0) {
        const items = flashcards.map((card, index) => `
            <li>
                <strong>${index + 1}. ${escapeHtml(card.front || "卡片")}</strong>
                <span>${escapeHtml(card.back || "")}</span>
            </li>
        `).join("")

        sections.push(`
            <details class="message-artifact">
                <summary>本轮卡片（${flashcards.length}）</summary>
                <ul>${items}</ul>
            </details>
        `)
    }

    if (traceItems.length > 0 || runtimePath.length > 0) {
        const pathLine = runtimePath.length > 0
            ? `<li><strong>Graph Path</strong><span>${escapeHtml(runtimePath.join(" → "))}</span></li>`
            : ""
        const items = traceItems.slice(0, 8).map(item => `<li><span>${escapeHtml(item)}</span></li>`).join("")

        sections.push(`
            <details class="message-artifact">
                <summary>本轮执行路径</summary>
                <ul>${pathLine}${items}</ul>
            </details>
        `)
    }

    return sections.length > 0
        ? `<div class="message-artifacts">${sections.join("")}</div>`
        : ""
}


function renderFlashcardsPanel(flashcards, options = {}) {
    if (!Array.isArray(flashcards) || flashcards.length === 0) {
        return emptyState("本次没有记忆卡片", "让 Agent 生成记忆卡片后，可以在这里翻面和下载 PNG。")
    }

    return flashcards.map((card, index) => {
        const displayIndex = (options.startIndex || 1) + index
        const front = escapeHtml(card.front || "")
        const back = escapeHtml(card.back || "")
        const rawDifficulty = String(card.difficulty || "medium").toLowerCase()
        const difficulty = ["easy", "medium", "hard"].includes(rawDifficulty)
            ? rawDifficulty
            : "medium"
        const tags = Array.isArray(card.tags) ? card.tags : []
        const tagsJson = escapeHtml(JSON.stringify(tags))
        const tagsHtml = tags
            .map(tag => `<span class="flashcard-tag">${escapeHtml(tag)}</span>`)
            .join("")

        return `
            <article
                class="flashcard"
                data-card-index="${displayIndex}"
                data-front="${front}"
                data-back="${back}"
                data-tags="${tagsJson}"
                data-difficulty="${escapeHtml(difficulty)}"
            >
                <div class="flashcard-card-toolbar">
                    <span class="flashcard-card-number">卡片 ${displayIndex}</span>
                    <div class="flashcard-card-actions">
                        <button class="download-single-flashcard-button" type="button">下载</button>
                    </div>
                </div>
                ${options.showTopic ? `
                    <div class="flashcard-library-meta">
                        <span>${escapeHtml(card.topic || "本轮学习")}</span>
                        <span>${escapeHtml(card.createdAt || "")}</span>
                    </div>
                ` : ""}
                <div class="flashcard-inner">
                    <div class="flashcard-face flashcard-front">
                        <div class="flashcard-label">正面</div>
                        <div class="flashcard-content">${front}</div>
                        <div class="flashcard-meta">
                            <span class="flashcard-difficulty flashcard-difficulty-${difficulty}">${difficulty}</span>
                            ${tagsHtml}
                        </div>
                    </div>
                    <div class="flashcard-face flashcard-back">
                        <div class="flashcard-label">背面</div>
                        <div class="flashcard-content">${back}</div>
                        <div class="flashcard-meta">
                            <span class="flashcard-difficulty flashcard-difficulty-${difficulty}">${difficulty}</span>
                            ${tagsHtml}
                        </div>
                    </div>
                </div>
            </article>
        `
    }).join("")
}


function renderInsights(data) {
    renderConversationInsights()
}


function addCardsToLibrary(cards, topic) {
    if (!Array.isArray(cards) || cards.length === 0) {
        return
    }

    const createdAt = new Date().toLocaleString()
    const cleanTopic = String(topic || "本轮学习").trim()

    cards.forEach(card => {
        cardLibrary.push({
            ...card,
            topic: cleanTopic,
            createdAt,
        })
    })

    saveCardLibrary()
    renderCardLibrary()
}


function clearCardLibrary() {
    cardLibrary = []
    saveCardLibrary()
    renderCardLibrary()
}


function renderCardLibrary() {
    const panel = getElement("cardLibraryPanel")

    if (!panel) {
        return
    }

    if (cardLibrary.length === 0) {
        panel.innerHTML = emptyState("还没有卡片", "让 Agent 生成 flashcard 后，卡片会出现在这里。")
        return
    }

    panel.innerHTML = renderFlashcardsPanel(cardLibrary, {
        showTopic: true,
        startIndex: 1,
    })
}


function getFlashcardFromElement(cardElement) {
    let tags = []

    try {
        tags = JSON.parse(cardElement.dataset.tags || "[]")
    } catch (error) {
        tags = []
    }

    return {
        index: Number(cardElement.dataset.cardIndex || 1),
        front: cardElement.dataset.front || "",
        back: cardElement.dataset.back || "",
        tags,
        difficulty: cardElement.dataset.difficulty || "medium",
    }
}


function wrapCanvasText(context, text, maxWidth) {
    const normalizedText = String(text || "").replace(/\s+/g, " ").trim()
    const words = normalizedText.includes(" ")
        ? normalizedText.split(" ")
        : Array.from(normalizedText)
    const lines = []
    let line = ""

    words.forEach(word => {
        const separator = normalizedText.includes(" ") ? " " : ""
        const testLine = line ? `${line}${separator}${word}` : word

        if (context.measureText(testLine).width > maxWidth && line) {
            lines.push(line)
            line = word
        } else {
            line = testLine
        }
    })

    if (line) {
        lines.push(line)
    }

    return lines
}


function drawRoundedRect(context, x, y, width, height, radius) {
    context.beginPath()
    context.moveTo(x + radius, y)
    context.lineTo(x + width - radius, y)
    context.quadraticCurveTo(x + width, y, x + width, y + radius)
    context.lineTo(x + width, y + height - radius)
    context.quadraticCurveTo(x + width, y + height, x + width - radius, y + height)
    context.lineTo(x + radius, y + height)
    context.quadraticCurveTo(x, y + height, x, y + height - radius)
    context.lineTo(x, y + radius)
    context.quadraticCurveTo(x, y, x + radius, y)
    context.closePath()
}


function drawCardFace(context, x, y, width, height, title, content, meta, fillColor, borderColor) {
    context.fillStyle = fillColor
    context.strokeStyle = borderColor
    context.lineWidth = 2
    drawRoundedRect(context, x, y, width, height, 16)
    context.fill()
    context.stroke()

    context.fillStyle = "#65727b"
    context.font = "700 22px Arial"
    context.fillText(title, x + 24, y + 36)

    context.fillStyle = "#172026"
    context.font = "24px Arial"
    let currentY = y + 78
    wrapCanvasText(context, content, width - 48).slice(0, 6).forEach(line => {
        context.fillText(line, x + 24, currentY)
        currentY += 34
    })

    context.fillStyle = "#34454f"
    context.font = "18px Arial"
    wrapCanvasText(context, meta, width - 48).slice(0, 2).forEach((line, index) => {
        context.fillText(line, x + 24, y + height - 52 + index * 22)
    })
}


function downloadCanvas(canvas, filename) {
    const link = document.createElement("a")

    link.href = canvas.toDataURL("image/png")
    link.download = filename
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
}


function createFlashcardFaceCanvas(card, side) {
    const scale = 2
    const padding = 28
    const faceWidth = 560
    const faceHeight = 300
    const canvas = document.createElement("canvas")
    const context = canvas.getContext("2d")
    const width = faceWidth + padding * 2
    const height = faceHeight + padding * 2
    const tags = Array.isArray(card.tags) && card.tags.length > 0
        ? card.tags.join(" / ")
        : "无标签"
    const meta = `标签：${tags}    难度：${card.difficulty || "medium"}`
    const isFront = side === "front"

    canvas.width = width * scale
    canvas.height = height * scale
    context.scale(scale, scale)
    context.fillStyle = "#fafaf7"
    context.fillRect(0, 0, width, height)

    drawCardFace(
        context,
        padding,
        padding,
        faceWidth,
        faceHeight,
        isFront ? "正面" : "背面",
        isFront ? card.front : card.back,
        meta,
        isFront ? "#ffffff" : "#fff6e6",
        isFront ? "#e1e6e0" : "#f1dfb7",
    )

    return canvas
}


function downloadFlashcardFiles(cards, side = "front") {
    const sides = side === "both" ? ["front", "back"] : [side]

    cards.forEach((card, cardIndex) => {
        sides.forEach((currentSide, sideIndex) => {
            const canvas = createFlashcardFaceCanvas(card, currentSide)
            const label = currentSide === "front" ? "front" : "back"
            const sequence = String(card.index || cardIndex + 1).padStart(2, "0")
            const delay = (cardIndex * sides.length + sideIndex) * 180

            setTimeout(() => {
                downloadCanvas(canvas, `flashcard-${sequence}-${label}.png`)
            }, delay)
        })
    })
}


function downloadSingleFlashcard(cardElement) {
    downloadFlashcardFiles([getFlashcardFromElement(cardElement)], "both")
}


function handleFlashcardClick(event) {
    const singleDownloadButton = event.target.closest(".download-single-flashcard-button")
    if (singleDownloadButton) {
        const card = singleDownloadButton.closest(".flashcard")
        if (card) {
            downloadSingleFlashcard(card)
        }
        return
    }

    const card = event.target.closest(".flashcard")
    const selectedText = window.getSelection ? window.getSelection().toString() : ""

    if (card && !selectedText) {
        card.classList.toggle("flipped")
    }
}


function handleKnowledgeLibraryClick(event) {
    const openButton = event.target.closest(".open-knowledge-button")

    if (!openButton) {
        return
    }

    openKnowledgeFile(Number(openButton.dataset.fileIndex))
}


async function uploadPDF() {
    const fileInput = getElement("pdfFile")
    const file = fileInput.files[0]

    if (!file) {
        alert("请先选择一个 PDF 文件")
        return
    }

    const formData = new FormData()
    formData.append("file", file)

    try {
        const response = await fetch(`${API_BASE_URL}/upload`, {
            method: "POST",
            body: formData,
        })

        if (!response.ok) {
            throw new Error(`上传失败：${response.status}`)
        }

        const data = await response.json()
        alert(data.message || "上传完成")
        knowledgeLoaded = false
        loadKnowledgeLibrary(true)
    } catch (error) {
        alert(`上传失败：${error.message}`)
    }
}


async function sendMessage(modeOverride) {
    updateUseRagState()
    updateRuntimeModeState()
    const requestBody = buildChatRequest(modeOverride)

    if (!requestBody.message) {
        alert("请输入内容")
        return
    }

    const loadingText = getElement("loadingText")
    const chatBox = getElement("chatBox")

    appendUserMessage(requestBody.message)
    addHistoryMessage("user", requestBody.message)
    getElement("userInput").value = ""
    loadingText.textContent = requestBody.use_rag
        ? "正在检索知识库、规划回答并整理学习成果..."
        : "正在组织解释、计划和复习卡片..."
    loadingText.style.display = "block"

    try {
        const response = await fetch(`${API_BASE_URL}/chat`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(requestBody),
        })

        if (!response.ok) {
            throw new Error(`后端请求失败：${response.status}`)
        }

        const data = await response.json()
        const cards = Array.isArray(data.flashcards) ? data.flashcards : []

        appendChatResponse(data, requestBody.message)
        addAssistantHistoryMessage(data, requestBody.message)
        addCardsToLibrary(cards, requestBody.message)
        renderConversationInsights()
        chatBox.scrollTop = chatBox.scrollHeight
    } catch (error) {
        appendErrorMessage(error.message)
    } finally {
        loadingText.style.display = "none"
    }
}


function activateInsightTab(tabName) {
    document.querySelectorAll(".insight-tab").forEach(tab => {
        tab.classList.toggle("active", tab.dataset.tab === tabName)
    })

    document.querySelectorAll(".insight-section").forEach(panel => {
        const isActive = panel.id === `${tabName}Panel`
        panel.classList.toggle("active", isActive)
    })
}


function activateMainView(viewName) {
    document.querySelectorAll(".nav-item").forEach(button => {
        button.classList.toggle("active", button.dataset.view === viewName)
    })

    const resultsPanel = document.querySelector(".results-panel")
    const views = {
        study: getElement("studyView"),
        knowledge: getElement("knowledgeView"),
        cards: getElement("cardsView"),
    }

    Object.entries(views).forEach(([name, element]) => {
        if (element) {
            element.classList.toggle("active", name === viewName)
        }
    })

    if (resultsPanel) {
        resultsPanel.style.display = viewName === "study" ? "flex" : "none"
    }

    if (viewName === "knowledge") {
        loadKnowledgeLibrary()
    } else if (viewName === "cards") {
        renderCardLibrary()
    }
}


function setupModeCards() {
    document.querySelectorAll(".mode-card").forEach(button => {
        button.addEventListener("click", () => setMode(button.dataset.modeChoice))
    })
}


document.addEventListener("DOMContentLoaded", () => {
    getElement("modeSelect").addEventListener("change", updateUseRagState)
    getElement("useAgentInput").addEventListener("change", () => updateRuntimeModeState("agent"))
    getElement("useLangGraphInput").addEventListener("change", () => updateRuntimeModeState("langgraph"))
    getElement("sendButton").addEventListener("click", () => sendMessage())
    getElement("uploadButton").addEventListener("click", uploadPDF)
    getElement("clearChatButton").addEventListener("click", clearConversation)
    getElement("newConversationButton").addEventListener("click", startNewConversation)
    getElement("conversationHistoryPanel").addEventListener("click", event => {
        const button = event.target.closest(".conversation-item")
        if (button) {
            restoreConversation(button.dataset.conversationId)
        }
    })
    getElement("flashcardsPanel").addEventListener("click", handleFlashcardClick)
    getElement("cardLibraryPanel").addEventListener("click", handleFlashcardClick)
    getElement("knowledgeLibraryPanel").addEventListener("click", handleKnowledgeLibraryClick)
    getElement("refreshKnowledgeButton").addEventListener("click", () => loadKnowledgeLibrary(true))
    getElement("clearCardsButton").addEventListener("click", clearCardLibrary)
    setupModeCards()

    document.querySelectorAll(".nav-item").forEach(button => {
        button.addEventListener("click", () => activateMainView(button.dataset.view))
    })

    document.querySelectorAll(".insight-tab").forEach(tab => {
        tab.addEventListener("click", () => activateInsightTab(tab.dataset.tab))
    })

    getElement("userInput").addEventListener("keydown", event => {
        if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
            sendMessage()
        }
    })

    renderWelcomeMessage()
    resetInsights()
    updateUseRagState()
    updateRuntimeModeState()
    renderConversationHistory()
    renderCardLibrary()
})
