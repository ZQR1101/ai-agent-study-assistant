import { useEffect, useMemo, useRef, useState } from "react"

const API_BASE_URL = "http://127.0.0.1:8000"
const HISTORY_LIMIT = 6
const CONVERSATIONS_KEY = "aiStudyAssistant.conversations.v1"
const CARD_LIBRARY_KEY = "aiStudyAssistant.cardLibrary.v1"

const MODE_LABELS = {
  auto: "自动规划",
  chat: "普通问答",
  rag: "知识库问答",
  explain: "概念解释",
  summarize: "内容总结",
  quiz: "自动出题",
  learn: "学习模式",
  langgraph: "LangGraph",
  agent: "Agent",
  image: "Image",
}

const TOOL_LABELS = {
  planner: "Planner",
  rag: "RAG",
  flashcard: "Flashcard",
  explain: "Explain",
  summarize: "Summarize",
  quiz: "Quiz",
  chat: "Chat",
}

const EMPTY_INSIGHTS = {
  sources: [],
  plan: [],
  flashcards: [],
  trace: [],
  runtime_info: {},
}

function createSessionId() {
  if (window.crypto?.randomUUID) {
    return window.crypto.randomUUID()
  }

  return `session-${Date.now()}-${Math.random().toString(16).slice(2)}`
}

function compactText(value) {
  return String(value || "")
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n[ \t]+/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim()
}

function renderAnswerContent(text) {
  const content = String(text || "")
  const imagePattern = /!\[([^\]]*)\]\((https?:\/\/[^)\s]+)\)/g
  const nodes = []
  let lastIndex = 0
  let match

  while ((match = imagePattern.exec(content)) !== null) {
    if (match.index > lastIndex) {
      nodes.push(<span key={`text-${lastIndex}`}>{content.slice(lastIndex, match.index)}</span>)
    }
    nodes.push(
      <a className="answer-image-link" href={match[2]} target="_blank" rel="noreferrer" key={`image-${match.index}`}>
        <ImagePreview className="answer-image" url={match[2]} alt={match[1] || "generated image"} />
      </a>,
    )
    lastIndex = imagePattern.lastIndex
  }

  if (lastIndex < content.length) {
    nodes.push(<span key={`text-${lastIndex}`}>{content.slice(lastIndex)}</span>)
  }

  return nodes.length > 0 ? nodes : content
}

function getImagePreviewUrl(url) {
  return url ? `${API_BASE_URL}/image-proxy?url=${encodeURIComponent(url)}` : ""
}

function ImagePreview({ url, alt, className }) {
  const [failed, setFailed] = useState(false)
  const previewUrl = getImagePreviewUrl(url)

  if (!url || failed) {
    return (
      <span className={`${className || ""} image-preview-fallback`}>
        <strong>图片预览加载失败</strong>
        <em>点击打开原图</em>
      </span>
    )
  }

  return (
    <img
      className={className}
      src={previewUrl}
      alt={alt || "generated image"}
      onError={() => setFailed(true)}
    />
  )
}

function extractImageCardsFromAnswer(answer, prompt = "") {
  const content = String(answer || "")
  const imagePattern = /!\[([^\]]*)\]\((https?:\/\/[^)\s]+)\)/g
  const cards = []
  let match

  while ((match = imagePattern.exec(content)) !== null) {
    cards.push({
      front: "图卡",
      back: prompt || "生成图片",
      tags: ["image"],
      difficulty: "easy",
      card_type: "image",
      image_url: match[2],
      image_alt: match[1] || prompt || "图卡",
    })
  }

  return cards
}

function getResponseCards(response, prompt = "") {
  const cards = Array.isArray(response?.flashcards) ? response.flashcards : []
  if (cards.length > 0) {
    return cards
  }

  return extractImageCardsFromAnswer(response?.answer, prompt)
}

function clampNumber(value, fallback, min, max) {
  const number = Number(value)
  if (Number.isNaN(number)) {
    return fallback
  }

  return Math.min(Math.max(number, min), max)
}

function readStorage(key, fallback) {
  try {
    const value = JSON.parse(localStorage.getItem(key) || "null")
    return Array.isArray(value) ? value : fallback
  } catch {
    return fallback
  }
}

function writeStorage(key, value) {
  localStorage.setItem(key, JSON.stringify(value))
}

function getConversationTitle(messages) {
  const firstUserMessage = messages.find((message) => message.role === "user")
  const title = String(firstUserMessage?.content || "新学习对话").trim()
  return title.length > 22 ? `${title.slice(0, 22)}...` : title
}

function getResponseSnapshot(data) {
  const flashcards = Array.isArray(data.flashcards) && data.flashcards.length > 0
    ? data.flashcards
    : extractImageCardsFromAnswer(data.answer)

  return {
    answer: data.answer || "",
    mode: data.mode || "",
    model: data.model || "",
    sources: Array.isArray(data.sources) ? data.sources : [],
    plan: Array.isArray(data.plan) ? data.plan : [],
    trace: Array.isArray(data.trace) ? data.trace : [],
    flashcards,
    runtime_info: hasRuntimeInfo(data.runtime_info) ? data.runtime_info : {},
  }
}

function hasRuntimeInfo(runtimeInfo) {
  return Boolean(
    runtimeInfo
      && typeof runtimeInfo === "object"
      && !Array.isArray(runtimeInfo)
      && Object.keys(runtimeInfo).length > 0,
  )
}

function flattenTraceItems(trace) {
  if (!Array.isArray(trace)) {
    return []
  }

  return trace.flatMap((block) => {
    if (typeof block === "string") {
      return [block]
    }

    if (block && Array.isArray(block.items)) {
      return block.items
    }

    return []
  })
}

function getExecutionModeLabel(mode) {
  if (mode === "langgraph") {
    return "执行方式：LangGraph 工作流"
  }

  if (mode === "agent") {
    return "执行方式：Agent Planner + Executor"
  }

  return mode ? `执行方式：${MODE_LABELS[mode] || mode}` : ""
}

function formatFileSize(bytes) {
  const size = Number(bytes || 0)
  if (size < 1024) {
    return `${size} B`
  }

  if (size < 1024 * 1024) {
    return `${(size / 1024).toFixed(1)} KB`
  }

  return `${(size / 1024 / 1024).toFixed(1)} MB`
}

function buildKnowledgeFileUrl(file) {
  if (file?.url) {
    return `${API_BASE_URL}${file.url}`
  }

  return `${API_BASE_URL}/knowledge-files/${encodeURIComponent(file?.name || "")}`
}

function buildKnowledgeContentUrl(file) {
  return `${API_BASE_URL}/knowledge-files/${encodeURIComponent(file?.name || "")}/content`
}

function getExecutionPath(response) {
  const runtimeInfo = response?.runtime_info || {}

  if (Array.isArray(runtimeInfo.graph_path) && runtimeInfo.graph_path.length > 0) {
    return runtimeInfo.graph_path
  }

  const plan = Array.isArray(response?.plan) ? response.plan : []
  if (plan.length > 0) {
    return ["agent", ...plan.map((step) => step.tool || "tool")]
  }

  const finalMode = flattenTraceItems(response?.trace).find((item) => String(item).includes("最终执行的模式"))
  if (finalMode) {
    const mode = String(finalMode).split(/[：:]/).pop().trim()
    return mode ? [mode] : []
  }

  return []
}

function formatRuntimeValue(value, fallback = "无") {
  if (value === null || value === undefined || value === "") {
    return fallback
  }

  if (typeof value === "boolean") {
    return value ? "是" : "否"
  }

  return String(value)
}

function formatToolLabel(name) {
  const normalized = String(name || "tool").toLowerCase()
  return TOOL_LABELS[normalized] || normalized.replace(/(^|[-_\s])([a-z])/g, (_, prefix, letter) => `${prefix}${letter.toUpperCase()}`)
}

function getToolLatency(call) {
  const value = call?.latency_ms ?? call?.latencyMs ?? call?.duration_ms
  const number = Number(value)
  return Number.isFinite(number) ? Math.max(0, Math.round(number)) : null
}

function formatLatency(value, fallback = "N/A") {
  const latency = Number(value)
  if (!Number.isFinite(latency)) {
    return fallback
  }

  return `${Math.max(0, Math.round(latency))}ms`
}

function getConversationTurns(messages) {
  const turns = []
  let pendingQuestion = ""

  messages.forEach((message) => {
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

function wrapCanvasText(context, text, maxWidth) {
  const normalizedText = String(text || "").replace(/\s+/g, " ").trim()
  const words = normalizedText.includes(" ") ? normalizedText.split(" ") : Array.from(normalizedText)
  const lines = []
  let line = ""

  words.forEach((word) => {
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

function drawCardFace(context, x, y, width, height, title, content, meta, fillColor, borderColor) {
  context.fillStyle = fillColor
  context.strokeStyle = borderColor
  context.lineWidth = 2
  drawRoundedRect(context, x, y, width, height, 24)
  context.fill()
  context.stroke()

  context.fillStyle = "#006b56"
  context.font = "700 18px Microsoft YaHei, sans-serif"
  context.fillText(title, x + 28, y + 42)

  context.fillStyle = "#1b1c1a"
  context.font = "500 22px Microsoft YaHei, sans-serif"
  wrapCanvasText(context, content, width - 56).slice(0, 6).forEach((line, index) => {
    context.fillText(line, x + 28, y + 92 + index * 32)
  })

  context.fillStyle = "#6e7a75"
  context.font = "600 14px Microsoft YaHei, sans-serif"
  context.fillText(meta, x + 28, y + height - 28)
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
  const tags = Array.isArray(card.tags) && card.tags.length > 0 ? card.tags.join(" / ") : "无标签"
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

function downloadCanvas(canvas, filename) {
  const link = document.createElement("a")
  link.download = filename
  link.href = canvas.toDataURL("image/png")
  link.click()
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

function EmptyState({ title, text }) {
  return (
    <div className="empty-state">
      <strong>{title}</strong>
      <span>{text}</span>
    </div>
  )
}

function RailIcon({ name }) {
  if (name === "workspace") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M4.5 10.5 12 4l7.5 6.5" />
        <path d="M6.5 9.5V20h15" />
        <path d="M9.5 20v-7h5v7" />
      </svg>
    )
  }

  if (name === "knowledge") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M6 5.5h9.5A2.5 2.5 0 0 1 18 8v13H7.5A2.5 2.5 0 0 1 5 18.5v-14A1.5 1.5 0 0 1 6.5 3H18" />
        <path d="M8 7h7" />
        <path d="M8 10h6" />
        <path d="M7.5 18H18" />
      </svg>
    )
  }

  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M6.5 4.5h11A1.5 1.5 0 0 1 19 6v16l-7-3-7 3V6a1.5 1.5 0 0 1 1.5-1.5Z" />
      <path d="M8.5 8h7" />
      <path d="M8.5 11h5" />
    </svg>
  )
}

function AppLogo() {
  return (
    <svg viewBox="0 0 40 40" aria-hidden="true">
      <rect x="6" y="6" width="28" height="28" rx="8" />
      <circle cx="15" cy="15" r="3.2" />
      <circle cx="25" cy="15" r="3.2" />
      <circle cx="20" cy="25" r="3.2" />
      <path d="M17.8 16.8 20 22" />
      <path d="M22.2 16.8 20 22" />
    </svg>
  )
}

function IconRail({ activeView, onViewChange }) {
  const items = [
    ["study", "workspace", "学习工作台"],
    ["knowledge", "knowledge", "知识库"],
    ["cards", "cards", "复习卡片"],
  ]

  return (
    <aside className="icon-rail" aria-label="全局导航">
      <div className="rail-logo" aria-label="AI 学习助手">
        <AppLogo />
      </div>
      <nav className="rail-nav">
        {items.map(([view, icon, label]) => (
          <button
            className={`rail-button${activeView === view ? " active" : ""}`}
            key={view}
            type="button"
            title={label}
            aria-label={label}
            onClick={() => onViewChange(view)}
          >
            <RailIcon name={icon} />
          </button>
        ))}
      </nav>
      <div className="rail-bottom">
        <button className="rail-button" type="button" title="设置" aria-label="设置">⚙</button>
        <button className="rail-button user-dot" type="button" title="账户" aria-label="账户">A</button>
      </div>
    </aside>
  )
}

function Sidebar({
  activeView,
  conversations,
  currentSessionId,
  settings,
  selectedFileName,
  onNewConversation,
  onRestoreConversation,
  onViewChange,
  onFileChange,
  onUpload,
  onSettingsChange,
}) {
  return (
    <aside className="app-sidebar project-sidebar" aria-label="学习项目导航">
      <header className="project-sidebar-header">
        <h1>学习项目</h1>
        <span>AI 学习构建器</span>
      </header>

      <button type="button" className="secondary-button new-thread-button" onClick={onNewConversation}>
        <span>+</span>
        新建学习
      </button>

      <div className="sidebar-search">⌕ 搜索学习项目</div>

      <section className="sidebar-section conversation-card">
        <div className="section-header">
          <div>
            <p className="eyebrow">项目</p>
            <h2>最近学习</h2>
          </div>
        </div>
        <div className="conversation-history">
          {conversations.length === 0 ? (
            <EmptyState title="暂无学习项目" text="发送消息后，当前学习项目会自动保存在这里。" />
          ) : (
            conversations.slice(0, 8).map((conversation) => (
              <button
                className={`conversation-item${conversation.id === currentSessionId ? " active" : ""}`}
                key={conversation.id}
                type="button"
                onClick={() => onRestoreConversation(conversation.id)}
              >
                <strong>{conversation.title}</strong>
                <span>{new Date(conversation.updatedAt).toLocaleString()}</span>
              </button>
            ))
          )}
        </div>
      </section>

      <section className="sidebar-section archive-card">
        <button type="button" className="archive-row">▾ 第 1 页</button>
        <button type="button" className="archive-row">▤ 归档</button>
      </section>
    </aside>
  )
}

function ChatMessage({ message }) {
  const isAssistant = message.role === "assistant"

  return (
    <article className={isAssistant ? "ai-message" : "user-message"}>
      <div className={isAssistant && message.response ? "response-meta" : "message-meta"}>
        {isAssistant && message.response ? (
          <>
            {getExecutionModeLabel(message.response.mode) ? <span>{getExecutionModeLabel(message.response.mode)}</span> : null}
            <span>模式：{MODE_LABELS[message.response.mode] || message.response.mode || "auto"}</span>
            <span>模型：{message.response.model || "默认模型"}</span>
          </>
        ) : (
          <span>{isAssistant ? "助手" : "你"}</span>
        )}
      </div>
      {isAssistant && message.response ? (
        <AnswerBlock response={message.response} />
      ) : (
        <div>{compactText(message.content)}</div>
      )}
    </article>
  )
}

function AnswerBlock({ response }) {
  const rawAnswer = compactText(response.answer)
  const traceItems = flattenTraceItems(response.trace)
  const ragPassed = traceItems.some((item) => String(item).includes("RAG 是否通过阈值：是"))
  const ragFailed = traceItems.some((item) => String(item).includes("RAG 是否通过阈值：否"))
  const fallbackUsed = traceItems.some((item) => String(item).includes("是否启用 fallback：是"))
  let title = response.mode === "learn" ? "学习内容" : ""
  let note = ""

  if (response.mode === "learn" && ragPassed) {
    title = "知识库学习内容"
  } else if (response.mode === "learn" && ragFailed && fallbackUsed) {
    title = "普通模型学习内容"
    note = "知识库未找到可靠相关内容，本部分由普通模型生成。"
  }

  return (
    <div className="answer">
      {title ? <strong>{title}</strong> : null}
      {note ? <div className="fallback-note">{note}</div> : null}
      <div>{renderAnswerContent(rawAnswer.replace(/^知识内容：\s*/, ""))}</div>
    </div>
  )
}

function StudyView({ messages, loading, input, setInput, onSend, onClear, settings }) {
  const chatEndRef = useRef(null)

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" })
  }, [messages, loading])

  return (
    <div className="workspace-page workspace-view active builder-workspace">
      <header className="builder-topbar">
        <div className="builder-breadcrumb">
          <span>AI 学习助手</span>
          <b>/</b>
          <strong>学习构建器</strong>
        </div>
        <nav className="builder-tabs" aria-label="Builder views">
          <button type="button" className="active">工作台</button>
          <button type="button">知识库</button>
          <button type="button">复习</button>
          <button type="button">路径</button>
        </nav>
        <div className="builder-actions">
          <span className="status-pill">● 草稿</span>
          <button type="button" className="ghost-button" onClick={onClear}>重置</button>
          <button type="button" className="primary-button" onClick={onSend}>运行学习</button>
        </div>
      </header>

      <div className="builder-canvas workflow-canvas" aria-label="学习 Builder 画布">
        <section className="workflow-board">
          <article className="canvas-label label-user">用户问题</article>
          <article className="canvas-label label-files">上传资料</article>

          <article className="workflow-node node-input">
            <header>
              <span className="node-icon">✎</span>
              <strong>文本输入</strong>
            </header>
            <p>输入要学习的主题、问题或材料片段。</p>
            <section className="composer builder-composer" aria-label="发送消息">
              <textarea
                rows="3"
                placeholder="例如：帮我理解 Agentic RAG，并生成复习卡片"
                value={input}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={(event) => {
                  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
                    onSend()
                  }
                }}
              />
              <button type="button" className="send-button" aria-label="发送" onClick={onSend}>↑</button>
            </section>
            <footer>Ctrl / ⌘ + Enter 运行</footer>
          </article>

          <article className="workflow-node node-files">
            <header>
              <span className="node-icon">▤</span>
              <strong>知识库文件</strong>
            </header>
            <p>连接本地 PDF、Markdown 和 TXT 资料。</p>
            <div className="node-action-row">
              <span>{settings.useRag ? "RAG 检索已启用" : "可在右侧启用 RAG"}</span>
            </div>
            <footer>docs/ 本地知识库</footer>
          </article>

          <article className="workflow-node node-agent">
            <header>
              <span className="node-icon logo-node"><AppLogo /></span>
              <strong>AI 学习助手</strong>
            </header>
            <p>组织解释、来源、学习计划和复习卡片。</p>
            <div className="node-action-row">
              <span>{settings.useLangGraph ? "LangGraph 工作流" : "Agent Planner"}</span>
            </div>
            <footer>{loading ? "正在运行..." : "等待运行"}</footer>
          </article>

          <article className="workflow-node node-output">
            <header>
              <span className="node-icon">↳</span>
              <strong>学习产出</strong>
            </header>
            <div className="chat-panel" aria-label="学习对话">
              <div className="chat-feed" aria-live="polite">
                {messages.length === 0 ? (
                  <article className="ai-message welcome-message">
                    <div className="answer">运行后，这里会显示解释、来源摘要和下一步复习建议。</div>
                  </article>
                ) : (
                  messages.map((message) => <ChatMessage key={message.id} message={message} />)
                )}
                {loading ? <div className="loading-text visible">正在整理学习成果...</div> : null}
                <div ref={chatEndRef} />
              </div>
            </div>
            <footer>用户可见结果</footer>
          </article>

          <svg className="workflow-lines" viewBox="0 0 980 560" preserveAspectRatio="none" aria-hidden="true">
            <path d="M202 168 C270 168, 302 282, 335 282" />
            <path d="M202 420 C270 420, 302 312, 335 312" />
            <path d="M595 292 C604 292, 612 292, 620 292" />
          </svg>
        </section>
      </div>
    </div>
  )
}

function KnowledgeView({ files, loading, viewer, onRefresh, onOpenFile }) {
  return (
    <section className="workspace-page content-page active" aria-label="知识库内容">
      <header className="workspace-header">
        <div>
          <p className="eyebrow">知识库</p>
          <h2>本地知识库</h2>
        </div>
        <button type="button" className="secondary-button" onClick={onRefresh}>刷新知识库</button>
      </header>
      <div className="workspace-toolbar" aria-label="知识库工具栏">
        <div className="search-surface">搜索 docs/ 中的 PDF、Markdown、TXT</div>
        <span className="metadata-chip">本地文件</span>
        <span className="metadata-chip">内联预览</span>
        <span className="metadata-chip warm">RAG 可引用</span>
      </div>
      <p className="workspace-copy">
        这里直接读取后端 docs/ 目录中的文件。点击“打开”会在当前页面预览 PDF、Markdown 或 TXT。
      </p>
      <div className="library-list">
        {loading ? (
          <EmptyState title="正在加载知识库" text="正在读取 docs 目录文件..." />
        ) : files.length === 0 ? (
          <EmptyState title="知识库为空" text="请先上传 PDF，或把 md / txt / pdf 文件放入 docs 后刷新。" />
        ) : (
          files.map((file, index) => (
            <article className="knowledge-source" key={file.name}>
              <div className="knowledge-source-header">
                <strong>{index + 1}. {file.name}</strong>
                <span>{String(file.type || "file").toUpperCase()} · {formatFileSize(file.size)}</span>
              </div>
              <div className="knowledge-actions">
                <button className="ghost-link" type="button" onClick={() => onOpenFile(file)}>打开</button>
              </div>
            </article>
          ))
        )}
      </div>
      <KnowledgeViewer viewer={viewer} />
    </section>
  )
}

function KnowledgeViewer({ viewer }) {
  if (!viewer) {
    return (
      <div className="knowledge-viewer">
        <EmptyState title="尚未打开文件" text="从上方知识库列表选择一个文件后，会在这里显示内容。" />
      </div>
    )
  }

  if (viewer.loading) {
    return (
      <div className="knowledge-viewer">
        <EmptyState title="正在打开文件" text={`${viewer.name} 加载中...`} />
      </div>
    )
  }

  if (viewer.error) {
    return (
      <div className="knowledge-viewer">
        <EmptyState title="文件打开失败" text={viewer.error} />
      </div>
    )
  }

  if (viewer.type === "pdf") {
    return (
      <div className="knowledge-viewer">
        <article className="knowledge-viewer-card">
          <div className="knowledge-source-header">
            <strong>{viewer.name}</strong>
            <span>PDF · {formatFileSize(viewer.size)}</span>
          </div>
          <div className="knowledge-actions">
            <a className="ghost-link" href={viewer.url} target="_blank" rel="noreferrer">新窗口打开</a>
          </div>
          <iframe className="knowledge-pdf-frame" src={viewer.url} title={viewer.name} />
        </article>
      </div>
    )
  }

  return (
    <div className="knowledge-viewer">
      <article className="knowledge-viewer-card">
        <div className="knowledge-source-header">
          <strong>{viewer.name}</strong>
          <span>{String(viewer.type || "text").toUpperCase()} · {formatFileSize(viewer.size)}</span>
        </div>
        <pre className="knowledge-text-preview">{viewer.content}</pre>
      </article>
    </div>
  )
}

function Flashcard({ card, index, showTopic = false }) {
  const [flipped, setFlipped] = useState(false)
  const tags = Array.isArray(card.tags) ? card.tags : []
  const isImageCard = card.card_type === "image" || Boolean(card.image_url)
  const difficulty = ["easy", "medium", "hard"].includes(String(card.difficulty || "").toLowerCase())
    ? String(card.difficulty).toLowerCase()
    : "medium"

  return (
    <article
      className={`flashcard${flipped ? " flipped" : ""}`}
      role="button"
      tabIndex={0}
      onClick={() => setFlipped(!flipped)}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault()
          setFlipped(!flipped)
        }
      }}
    >
      <div className="flashcard-card-toolbar">
        <span className="flashcard-card-number">{isImageCard ? "图卡" : `#${index}`}</span>
        <span className="flashcard-card-actions">
          {isImageCard ? (
            <a href={card.image_url} target="_blank" rel="noreferrer" onClick={(event) => event.stopPropagation()}>
              打开
            </a>
          ) : (
            <button
              type="button"
              onClick={(event) => {
                event.stopPropagation()
                downloadFlashcardFiles([{ ...card, index }], "both")
              }}
            >
              下载
            </button>
          )}
        </span>
      </div>
      {showTopic && card.topic ? <div className="flashcard-library-meta">{card.topic}</div> : null}
      <div className="flashcard-inner">
        <div className="flashcard-face flashcard-front">
          <span className="flashcard-label">{isImageCard ? "图卡" : "正面"}</span>
          {isImageCard ? (
            <a className="flashcard-image-link" href={card.image_url} target="_blank" rel="noreferrer" onClick={(event) => event.stopPropagation()}>
              <ImagePreview className="flashcard-image" url={card.image_url} alt={card.image_alt || card.front || "图卡"} />
            </a>
          ) : (
            <div className="flashcard-content">{card.front || "空白卡片"}</div>
          )}
          <FlashcardMeta difficulty={difficulty} tags={tags} />
        </div>
        <div className="flashcard-face flashcard-back">
          <span className="flashcard-label">{isImageCard ? "提示词" : "背面"}</span>
          <div className="flashcard-content">{card.back || (isImageCard ? "点击正面图片查看原图" : "暂无答案")}</div>
          <FlashcardMeta difficulty={difficulty} tags={tags} />
        </div>
      </div>
    </article>
  )
}

function FlashcardMeta({ difficulty, tags }) {
  return (
    <div className="flashcard-meta">
      <span className={`flashcard-difficulty flashcard-difficulty-${difficulty}`}>{difficulty}</span>
      {tags.slice(0, 3).map((tag) => <span className="flashcard-tag" key={tag}>{tag}</span>)}
    </div>
  )
}

function CardsView({ cards, onClear }) {
  return (
    <section className="workspace-page content-page active" aria-label="卡片库">
      <header className="workspace-header">
        <div>
          <p className="eyebrow">Flashcard Library</p>
          <h2>卡片库</h2>
        </div>
        <button type="button" className="secondary-button" onClick={onClear}>清空卡片库</button>
      </header>
      <div className="workspace-toolbar" aria-label="卡片库工具栏">
        <div className="search-surface">按主题、标签或难度筛选卡片</div>
        <span className="metadata-chip">点击翻面</span>
        <span className="metadata-chip">PNG 导出</span>
        <span className="metadata-chip warm">温习模式</span>
      </div>
      <p className="workspace-copy">
        每轮对话生成的卡片会自动保存在这里，方便后续复习、翻面和下载。
      </p>
      <div className="card-library-grid">
        {cards.length === 0 ? (
          <EmptyState title="还没有卡片" text="让 Agent 生成 flashcard 后，卡片会出现在这里。" />
        ) : (
          cards.map((card, index) => <Flashcard card={card} index={index + 1} key={card.id || `${card.front}-${index}`} showTopic />)
        )}
      </div>
    </section>
  )
}

function ResultsPanel({ turns, activeTab, onTabChange, settings, onSettingsChange }) {
  return (
    <aside className="results-panel inspector-panel" aria-label="学习配置与成果">
      <header className="results-header inspector-header">
        <div>
          <p className="eyebrow">检查器</p>
          <h2>学习配置</h2>
        </div>
        <span className="saved-pill">● 已保存</span>
      </header>

      <section className="inspector-section">
        <div className="inspector-row">
          <span>学习模式</span>
          <strong>{settings.useLangGraph ? "LangGraph" : "Agent 导师"}</strong>
        </div>
        <div className="inspector-row">
          <span>默认产出</span>
          <strong>计划 + 卡片</strong>
        </div>
      </section>

      <details className="inspector-form inspector-settings" aria-label="运行设置">
        <summary>运行设置</summary>
        <label>
          Planner 模式
          <select value={settings.plannerMode} onChange={(event) => onSettingsChange({ plannerMode: event.target.value })}>
            <option value="rule">rule（规则）</option>
            <option value="llm">llm（模型规划）</option>
          </select>
        </label>
        <label>
          模型
          <select value={settings.model} onChange={(event) => onSettingsChange({ model: event.target.value })}>
            <option value="mimo-v2.5">mimo-v2.5</option>
            <option value="deepseek-v4-pro">deepseek-v4-pro</option>
            <option value="deepseek-v4-flash">deepseek-v4-flash</option>
            <option value="qwen3.7-max">qwen3.7-max</option>
            <option value="wanx2.1-t2i-plus">wanx2.1-t2i-plus</option>
          </select>
        </label>
        <div className="inspector-grid">
          <label>
            温度
            <input
              type="number"
              min="0"
              max="2"
              step="0.1"
              value={settings.temperature}
              onChange={(event) => onSettingsChange({ temperature: event.target.value })}
            />
          </label>
          <label>
            Top K
            <input
              type="number"
              min="1"
              max="10"
              step="1"
              value={settings.topK}
              onChange={(event) => onSettingsChange({ topK: event.target.value })}
            />
          </label>
        </div>
        <label className="inspector-toggle">
          <span>
            <strong>Agent 模式</strong>
            <em>启用规划与工具执行</em>
          </span>
          <input
            type="checkbox"
            checked={settings.useAgent}
            onChange={(event) => onSettingsChange({
              useAgent: event.target.checked,
              useLangGraph: event.target.checked ? false : settings.useLangGraph,
            })}
          />
        </label>
        <label className="inspector-toggle">
          <span>
            <strong>RAG 检索</strong>
            <em>从本地知识库查找来源</em>
          </span>
          <input
            type="checkbox"
            checked={settings.useRag}
            onChange={(event) => onSettingsChange({ useRag: event.target.checked })}
          />
        </label>
        <label className="inspector-toggle">
          <span>
            <strong>LangGraph Workflow</strong>
            <em>使用图工作流运行</em>
          </span>
          <input
            type="checkbox"
            checked={settings.useLangGraph}
            onChange={(event) => onSettingsChange({
              useLangGraph: event.target.checked,
              useAgent: event.target.checked ? false : settings.useAgent,
            })}
          />
        </label>
      </details>

      <nav className="insight-tabs inspector-tabs" aria-label="成果类型">
        {[
          ["sources", "来源"],
          ["plan", "计划"],
          ["flashcards", "卡片"],
          ["trace", "路径"],
        ].map(([tab, label]) => (
          <button
            type="button"
            className={`insight-tab${activeTab === tab ? " active" : ""}`}
            key={tab}
            onClick={() => onTabChange(tab)}
          >
            {label}
          </button>
        ))}
      </nav>

      <section className="insight-section active">
        <InsightContent tab={activeTab} turns={turns} />
      </section>

      <div className="study-progress">
        <div>
          <span>今日专注时间</span>
          <strong>45 分钟</strong>
        </div>
        <div className="progress-track"><span /></div>
      </div>
    </aside>
  )
}

function InsightContent({ tab, turns }) {
  if (turns.length === 0) {
    const copy = {
      sources: ["等待一次知识检索", "启用 RAG 后，这里会显示命中文档和相关片段。"],
      plan: ["等待 Agent 计划", "启用 Agent 后，复杂学习任务会拆成工具步骤。"],
      flashcards: ["还没有记忆卡片", "请求生成 flashcard 后，可以在这里翻看和下载 PNG。"],
      trace: ["暂无执行路径", "这里会记录检索、规划、fallback 等运行细节。"],
    }[tab]

    return <EmptyState title={copy[0]} text={copy[1]} />
  }

  if (tab === "sources") {
    return turns.map((turn, turnIndex) => {
      const sources = Array.isArray(turn.response.sources) ? turn.response.sources : []
      return (
        <article className="turn-panel-card" key={`${turn.question}-${turnIndex}`}>
          <h3>{turnIndex + 1}. {turn.question}</h3>
          <ul className="turn-detail-list">
            {sources.length === 0 ? (
              <li><span>本轮没有来源。若启用 RAG 且命中知识库，来源会显示在这里。</span></li>
            ) : (
              sources.map((source, sourceIndex) => {
                const sourceName = typeof source === "string" ? source : source.source || "未知来源"
                const score = typeof source === "string" || source.score == null ? "" : `相似度 ${Number(source.score).toFixed(4)}`
                const snippet = typeof source === "string" ? "" : source.text || source.snippet || ""
                return (
                  <li key={`${sourceName}-${sourceIndex}`}>
                    <strong>{sourceIndex + 1}. {sourceName}</strong>
                    {score ? <span>{score}</span> : null}
                    {snippet ? <span>{snippet}</span> : null}
                  </li>
                )
              })
            )}
          </ul>
        </article>
      )
    })
  }

  if (tab === "plan") {
    return turns.map((turn, turnIndex) => {
      const plan = Array.isArray(turn.response.plan) ? turn.response.plan : []
      return (
        <article className="turn-panel-card" key={`${turn.question}-${turnIndex}`}>
          <h3>{turnIndex + 1}. {turn.question}</h3>
          <ul className="turn-detail-list">
            {plan.length === 0 ? (
              <li><span>本轮没有返回学习计划。</span></li>
            ) : (
              plan.map((step, stepIndex) => (
                <li key={`${step.tool}-${stepIndex}`}>
                  <strong>{stepIndex + 1}. {step.tool || "unknown"}</strong>
                  {step.input ? <span>输入：{step.input}</span> : null}
                  {step.reason ? <span>原因：{step.reason}</span> : null}
                </li>
              ))
            )}
          </ul>
        </article>
      )
    })
  }

  if (tab === "flashcards") {
    let nextIndex = 1
    return turns.map((turn, turnIndex) => {
      const cards = getResponseCards(turn.response, turn.question)
      const startIndex = nextIndex
      nextIndex += cards.length
      return (
        <article className="turn-panel-card" key={`${turn.question}-${turnIndex}`}>
          <h3>{turnIndex + 1}. {turn.question}</h3>
          {cards.length === 0 ? (
            <EmptyState title="本轮没有卡片" text="如果请求生成记忆卡片，结果会显示在这里。" />
          ) : (
            <div className="card-library-grid">
              {cards.map((card, index) => <Flashcard card={card} index={startIndex + index} key={`${card.front}-${index}`} />)}
            </div>
          )}
        </article>
      )
    })
  }

  return turns.map((turn, turnIndex) => {
    const runtimeInfo = turn.response.runtime_info || {}
    const path = getExecutionPath(turn.response)
    return (
      <article className="turn-panel-card" key={`${turn.question}-${turnIndex}`}>
        <h3>{turnIndex + 1}. {turn.question}</h3>
        {path.length > 0 ? (
          <div className="execution-path-line">
            {path.map((node, index) => (
              <span key={`${node}-${index}`}>{node}</span>
            ))}
          </div>
        ) : (
          <div className="execution-path-line muted">暂无执行路径摘要</div>
        )}
        {hasRuntimeInfo(runtimeInfo) ? (
          <RuntimeInfoPanel runtimeInfo={runtimeInfo} fallbackPath={path} />
        ) : null}
        <TraceBlocks trace={turn.response.trace} />
      </article>
    )
  })
}

function RuntimeInfoPanel({ runtimeInfo, fallbackPath }) {
  const graphPath = Array.isArray(runtimeInfo.graph_path) ? runtimeInfo.graph_path : []
  const path = graphPath.length > 0 ? graphPath : fallbackPath
  const toolCalls = Array.isArray(runtimeInfo.tool_calls) ? runtimeInfo.tool_calls : []
  const timedToolCalls = toolCalls.filter((call) => getToolLatency(call) !== null)

  return (
    <section className="runtime-panel runtime-info-panel compact-runtime">
      <div className="runtime-info-header">
        <strong>Runtime Info</strong>
        <span className="runtime-engine">{formatRuntimeValue(runtimeInfo.runtime, "runtime")}</span>
      </div>
      {timedToolCalls.length > 0 ? (
        <div className="runtime-latency-summary" aria-label="Tool latency summary">
          {timedToolCalls.map((call, index) => {
            const label = formatToolLabel(call.tool || call.name || call.node)
            return (
              <span className="runtime-latency-pill" key={`${label}-${index}`}>
                <strong>{label}:</strong> {formatLatency(getToolLatency(call))}
              </span>
            )
          })}
        </div>
      ) : null}
      <dl className="runtime-grid execution-summary">
        <dt>runtime</dt>
        <dd>{formatRuntimeValue(runtimeInfo.runtime)}</dd>
        <dt>planner_mode</dt>
        <dd>{formatRuntimeValue(runtimeInfo.planner_mode)}</dd>
        <dt>planner_fallback</dt>
        <dd>{formatRuntimeValue(runtimeInfo.planner_fallback)}</dd>
        <dt>planner_error</dt>
        <dd className={runtimeInfo.planner_error ? "runtime-error-text" : ""}>
          {formatRuntimeValue(runtimeInfo.planner_error)}
        </dd>
        <dt>graph_path</dt>
        <dd className="runtime-path">{path.length > 0 ? path.join(" → ") : "无"}</dd>
        <dt>node_count</dt>
        <dd>{formatRuntimeValue(runtimeInfo.node_count ?? path.length)}</dd>
        <dt>finalizer</dt>
        <dd>{runtimeInfo.finalizer_used ? "启用" : "未启用"}</dd>
        {runtimeInfo.error ? (
          <>
            <dt>error</dt>
            <dd className="runtime-error-text">{String(runtimeInfo.error)}</dd>
          </>
        ) : null}
      </dl>
      <div className="runtime-tool-calls">
        <strong>tool_calls</strong>
        {toolCalls.length === 0 ? (
          <span className="runtime-empty">无工具调用</span>
        ) : (
          toolCalls.map((call, index) => (
            <article className="runtime-tool-call" key={`${call.tool || "tool"}-${index}`}>
              <div className="runtime-tool-header">
                <strong>{index + 1}. {call.tool || call.name || "unknown"}</strong>
                <span className={`runtime-badge runtime-status ${call.success === false ? "failed" : "success"}`}>
                  {call.success === false ? "失败" : "成功"}
                </span>
              </div>
              <dl className="runtime-grid runtime-detail-grid">
                <dt>node</dt>
                <dd>{formatRuntimeValue(call.node)}</dd>
                <dt>description</dt>
                <dd>{formatRuntimeValue(call.description)}</dd>
                <dt>success</dt>
                <dd>{formatRuntimeValue(call.success)}</dd>
                <dt>used_context</dt>
                <dd>{formatRuntimeValue(call.used_context)}</dd>
                <dt>context_sources</dt>
                <dd>
                  {Array.isArray(call.context_sources) && call.context_sources.length > 0
                    ? call.context_sources.join("、")
                    : "无"}
                </dd>
                <dt>latency_ms</dt>
                <dd>{formatLatency(getToolLatency(call))}</dd>
                <dt>output_length</dt>
                <dd>{formatRuntimeValue(call.output_length, "0")}</dd>
                {call.error ? (
                  <>
                    <dt>error</dt>
                    <dd className="runtime-error-text">{String(call.error)}</dd>
                  </>
                ) : null}
              </dl>
            </article>
          ))
        )}
      </div>
    </section>
  )
}

function TraceBlocks({ trace }) {
  if (!Array.isArray(trace) || trace.length === 0) {
    return <EmptyState title="本轮没有执行路径" text="后端返回 trace 后，这里会显示系统处理过程。" />
  }

  if (typeof trace[0] === "object" && !Array.isArray(trace[0])) {
    return trace.map((block, index) => (
      <article className="trace-block" key={`${block.title}-${index}`}>
        <strong>{block.title || "执行信息"}</strong>
        <ul>{(block.items || []).map((item) => <li key={item}>{item}</li>)}</ul>
      </article>
    ))
  }

  return (
    <article className="trace-block">
      <strong>执行信息</strong>
      <ul>{trace.map((item) => <li key={item}>{item}</li>)}</ul>
    </article>
  )
}

export default function App() {
  const [activeView, setActiveView] = useState("study")
  const [activeTab, setActiveTab] = useState("sources")
  const [currentSessionId, setCurrentSessionId] = useState(createSessionId)
  const [messages, setMessages] = useState([])
  const [conversations, setConversations] = useState(() => readStorage(CONVERSATIONS_KEY, []))
  const [cardLibrary, setCardLibrary] = useState(() => readStorage(CARD_LIBRARY_KEY, []))
  const [knowledgeFiles, setKnowledgeFiles] = useState([])
  const [knowledgeLoading, setKnowledgeLoading] = useState(false)
  const [knowledgeLoaded, setKnowledgeLoaded] = useState(false)
  const [knowledgeViewer, setKnowledgeViewer] = useState(null)
  const [selectedFile, setSelectedFile] = useState(null)
  const [input, setInput] = useState("")
  const [loading, setLoading] = useState(false)
  const [settings, setSettings] = useState({
    model: "mimo-v2.5",
    plannerMode: "rule",
    temperature: "0.7",
    topK: "3",
    useAgent: true,
    useRag: false,
    useLangGraph: false,
  })

  const turns = useMemo(() => getConversationTurns(messages), [messages])

  useEffect(() => {
    writeStorage(CONVERSATIONS_KEY, conversations.slice(0, 20))
  }, [conversations])

  useEffect(() => {
    writeStorage(CARD_LIBRARY_KEY, cardLibrary.slice(0, 200))
  }, [cardLibrary])

  useEffect(() => {
    if (activeView === "knowledge") {
      loadKnowledgeLibrary()
    }
  }, [activeView])

  function persistConversation(nextMessages, sessionId = currentSessionId) {
    if (nextMessages.length === 0) {
      return
    }

    const nextConversation = {
      id: sessionId,
      title: getConversationTitle(nextMessages),
      updatedAt: new Date().toISOString(),
      messages: nextMessages.map((message) => ({ ...message })),
    }

    setConversations((current) => [
      nextConversation,
      ...current.filter((conversation) => conversation.id !== sessionId),
    ])
  }

  function startNewConversation() {
    const nextSessionId = createSessionId()
    setCurrentSessionId(nextSessionId)
    setMessages([])
    setInput("")
    setActiveTab("sources")
  }

  function restoreConversation(conversationId) {
    const conversation = conversations.find((item) => item.id === conversationId)
    if (!conversation) {
      return
    }

    setCurrentSessionId(conversation.id)
    setMessages(conversation.messages.map((message) => ({ ...message })))
    setActiveView("study")
  }

  function addCardsToLibrary(cards, topic) {
    if (!Array.isArray(cards) || cards.length === 0) {
      return
    }

    const createdAt = new Date().toLocaleString()
    const cleanTopic = String(topic || "本轮学习").trim()
    const nextCards = cards.map((card, index) => ({
      ...card,
      id: `${Date.now()}-${index}-${Math.random().toString(16).slice(2)}`,
      topic: cleanTopic,
      createdAt,
    }))

    setCardLibrary((current) => [...nextCards, ...current].slice(0, 200))
  }

  async function loadKnowledgeLibrary(force = false) {
    if (knowledgeLoaded && !force) {
      return
    }

    setKnowledgeLoading(true)

    try {
      const response = await fetch(`${API_BASE_URL}/knowledge-files`)
      if (!response.ok) {
        throw new Error(`知识库接口返回 ${response.status}`)
      }

      const data = await response.json()
      setKnowledgeFiles(Array.isArray(data.files) ? data.files : [])
      setKnowledgeLoaded(true)
    } catch (error) {
      setKnowledgeFiles([])
      setKnowledgeViewer({ error: error.message })
    } finally {
      setKnowledgeLoading(false)
    }
  }

  async function openKnowledgeFile(file) {
    setKnowledgeViewer({ loading: true, name: file.name })

    try {
      const fileUrl = buildKnowledgeFileUrl(file)
      if (file.type === "pdf") {
        setKnowledgeViewer({ ...file, url: fileUrl })
        return
      }

      const response = await fetch(buildKnowledgeContentUrl(file))
      if (!response.ok) {
        throw new Error(`文本预览接口返回 ${response.status}`)
      }

      const payload = await response.json()
      setKnowledgeViewer({ ...file, content: payload.content || "" })
    } catch (error) {
      setKnowledgeViewer({ error: error.message })
    }
  }

  async function uploadPDF() {
    if (!selectedFile) {
      alert("请先选择一个 PDF 文件")
      return
    }

    const formData = new FormData()
    formData.append("file", selectedFile)

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
      setSelectedFile(null)
      setKnowledgeLoaded(false)
      await loadKnowledgeLibrary(true)
    } catch (error) {
      alert(`上传失败：${error.message}`)
    }
  }

  async function sendMessage() {
    const message = input.trim()
    if (!message) {
      alert("请输入内容")
      return
    }

    const normalizedSettings = {
      ...settings,
      useAgent: settings.useAgent || (!settings.useAgent && !settings.useLangGraph),
    }
    const userMessage = {
      id: createSessionId(),
      role: "user",
      content: message,
    }
    const requestMessages = [...messages, userMessage].slice(-HISTORY_LIMIT * 2)

    setMessages(requestMessages)
    persistConversation(requestMessages)
    setInput("")
    setLoading(true)

    const requestBody = {
      message,
      mode: "auto",
      model: normalizedSettings.model,
      temperature: clampNumber(normalizedSettings.temperature, 0.7, 0, 2),
      planner_mode: normalizedSettings.plannerMode || "rule",
      use_agent: normalizedSettings.useAgent,
      use_rag: normalizedSettings.useRag,
      use_langgraph: normalizedSettings.useLangGraph,
      top_k: Math.round(clampNumber(normalizedSettings.topK, 3, 1, 10)),
      session_id: currentSessionId,
      history: messages.slice(-HISTORY_LIMIT).map((item) => ({
        role: item.role,
        content: item.content,
      })),
    }

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
      const assistantMessage = {
        id: createSessionId(),
        role: "assistant",
        content: data.answer || "",
        response: getResponseSnapshot(data),
        requestMessage: message,
      }
      const nextMessages = [...requestMessages, assistantMessage].slice(-HISTORY_LIMIT * 2)

      setMessages(nextMessages)
      persistConversation(nextMessages)
      const responseCards = getResponseCards(data, message)
      addCardsToLibrary(responseCards, message)
      setActiveTab(data.mode === "image" && responseCards.length > 0 ? "flashcards" : "sources")
    } catch (error) {
      const errorMessage = {
        id: createSessionId(),
        role: "assistant",
        content: error.message,
        response: {
          ...EMPTY_INSIGHTS,
          answer: error.message,
          mode: "error",
          model: "",
        },
        requestMessage: message,
      }
      const nextMessages = [...requestMessages, errorMessage].slice(-HISTORY_LIMIT * 2)
      setMessages(nextMessages)
      persistConversation(nextMessages)
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="app-shell">
      <IconRail activeView={activeView} onViewChange={setActiveView} />
      <Sidebar
        activeView={activeView}
        conversations={conversations}
        currentSessionId={currentSessionId}
        settings={settings}
        selectedFileName={selectedFile?.name}
        onNewConversation={startNewConversation}
        onRestoreConversation={restoreConversation}
        onViewChange={setActiveView}
        onFileChange={(event) => setSelectedFile(event.target.files?.[0] || null)}
        onUpload={uploadPDF}
        onSettingsChange={(patch) => setSettings((current) => {
          const next = { ...current, ...patch }
          if (!next.useAgent && !next.useLangGraph) {
            next.useAgent = true
          }
          return next
        })}
      />

      <section className="workspace-column">
        {activeView === "study" ? (
          <StudyView
            messages={messages}
            loading={loading}
            input={input}
            setInput={setInput}
            onSend={sendMessage}
            onClear={startNewConversation}
            settings={settings}
          />
        ) : null}
        {activeView === "knowledge" ? (
          <KnowledgeView
            files={knowledgeFiles}
            loading={knowledgeLoading}
            viewer={knowledgeViewer}
            onRefresh={() => loadKnowledgeLibrary(true)}
            onOpenFile={openKnowledgeFile}
          />
        ) : null}
        {activeView === "cards" ? (
          <CardsView
            cards={cardLibrary}
            onClear={() => {
              if (confirm("确定要清空卡片库吗？")) {
                setCardLibrary([])
              }
            }}
          />
        ) : null}
      </section>

      {activeView === "study" ? (
        <ResultsPanel
          turns={turns}
          activeTab={activeTab}
          onTabChange={setActiveTab}
          settings={settings}
          onSettingsChange={(patch) => setSettings((current) => {
            const next = { ...current, ...patch }
            if (!next.useAgent && !next.useLangGraph) {
              next.useAgent = true
            }
            return next
          })}
        />
      ) : null}
    </main>
  )
}
