const API_BASE_URL = "http://127.0.0.1:8000"
const HISTORY_LIMIT = 6
const SESSION_ID = (
    window.crypto && window.crypto.randomUUID
        ? window.crypto.randomUUID()
        : `session-${Date.now()}-${Math.random().toString(16).slice(2)}`
)
const chatHistory = []


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


function getRecentHistory() {
    return chatHistory.slice(-HISTORY_LIMIT)
}


function addHistoryMessage(role, content) {
    const cleanContent = String(content || "").trim()

    if (!cleanContent) {
        return
    }

    chatHistory.push({
        role,
        content: cleanContent,
    })

    if (chatHistory.length > HISTORY_LIMIT * 2) {
        chatHistory.splice(0, chatHistory.length - HISTORY_LIMIT * 2)
    }
}


function clearConversation() {
    chatHistory.length = 0
    getElement("chatBox").innerHTML = `
        <article class="ai-message">
            <div class="message-meta">
                <span>助手</span>
                <span>学习工作台已就绪</span>
            </div>
            <div class="answer">
                对话已清空。选择左侧模式，输入新的学习主题即可继续。
            </div>
        </article>
    `
    getElement("userInput").value = ""
    getElement("loadingText").style.display = "none"
    resetInsights()
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
}


function buildChatRequest(modeOverride) {
    const message = getElement("userInput").value.trim()
    const mode = modeOverride || getElement("modeSelect").value

    return {
        message,
        mode,
        model: getElement("modelSelect").value,
        temperature: clampNumber(getElement("temperatureInput").value, 0.7, 0, 2),
        use_agent: getElement("useAgentInput").checked,
        use_rag: mode === "rag" ? true : getElement("useRagInput").checked,
        top_k: Math.round(clampNumber(getElement("topKInput").value, 3, 1, 10)),
        session_id: SESSION_ID,
        history: getRecentHistory(),
    }
}


function appendUserMessage(message) {
    getElement("chatBox").insertAdjacentHTML("beforeend", `
        <article class="user-message">
            <div class="message-meta">
                <span>你</span>
            </div>
            <div>${escapeHtml(message)}</div>
        </article>
    `)
}


function appendErrorMessage(message) {
    getElement("chatBox").insertAdjacentHTML("beforeend", `
        <article class="ai-message error-message">
            <div class="message-meta">
                <span>错误</span>
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


function renderAnswer(data) {
    const rawAnswer = String(data.answer || "")

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


function appendChatResponse(data) {
    const answerHtml = renderAnswer(data)

    getElement("chatBox").insertAdjacentHTML("beforeend", `
        <article class="ai-message">
            <div class="response-meta">
                <span>模式：${escapeHtml(data.mode || "")}</span>
                <span>模型：${escapeHtml(data.model || "")}</span>
            </div>
            ${answerHtml}
        </article>
    `)

    renderInsights(data)
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
    getElement("sourcesPanel").innerHTML = emptyState("暂无来源", "启用 RAG 后，这里会显示命中文档、相似度和片段。")
    getElement("planPanel").innerHTML = emptyState("暂无 Agent 计划", "启用 Agent 后，这里会显示工具调用步骤。")
    getElement("tracePanel").innerHTML = emptyState("暂无执行路径", "请求完成后，这里会记录检索、规划和 fallback 状态。")
    getElement("flashcardsPanel").innerHTML = emptyState("暂无记忆卡片", "让 Agent 生成 flashcard 后，可以在这里翻看和下载。")
}


function renderSourcesPanel(sources) {
    if (!Array.isArray(sources) || sources.length === 0) {
        return emptyState("本次没有来源", "如果需要引用知识库，请选择 RAG 模式或打开 RAG 检索。")
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
                <strong>${index + 1}. ${sourceName}</strong>
                <span class="source-score">相似度 ${score}</span>
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


function renderFlashcardsPanel(flashcards) {
    if (!Array.isArray(flashcards) || flashcards.length === 0) {
        return emptyState("本次没有记忆卡片", "让 Agent 生成记忆卡片后，可以在这里翻面和下载 PNG。")
    }

    return flashcards.map((card, index) => {
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
                data-card-index="${index + 1}"
                data-front="${front}"
                data-back="${back}"
                data-tags="${tagsJson}"
                data-difficulty="${escapeHtml(difficulty)}"
            >
                <div class="flashcard-card-toolbar">
                    <span class="flashcard-card-number">卡片 ${index + 1}</span>
                    <div class="flashcard-card-actions">
                        <button class="download-single-flashcard-button" type="button">下载</button>
                    </div>
                </div>
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
    getElement("sourcesPanel").innerHTML = renderSourcesPanel(data.sources)
    getElement("planPanel").innerHTML = renderPlanPanel(data.plan)
    getElement("tracePanel").innerHTML = renderTracePanel(data.trace)
    getElement("flashcardsPanel").innerHTML = renderFlashcardsPanel(data.flashcards)
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

    context.fillStyle = "#65747c"
    context.font = "700 22px Arial"
    context.fillText(title, x + 24, y + 36)

    context.fillStyle = "#172126"
    context.font = "24px Arial"
    let currentY = y + 78
    wrapCanvasText(context, content, width - 48).slice(0, 6).forEach(line => {
        context.fillText(line, x + 24, currentY)
        currentY += 34
    })

    context.fillStyle = "#334155"
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
    context.fillStyle = "#f7faf9"
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
        isFront ? "#ffffff" : "#ecfdf5",
        isFront ? "#d8e2df" : "#a7f3d0",
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
    } catch (error) {
        alert(`上传失败：${error.message}`)
    }
}


async function sendMessage(modeOverride) {
    updateUseRagState()
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

        appendChatResponse(data)
        addHistoryMessage("assistant", data.answer || "")
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


function setupQuickActions() {
    document.querySelectorAll(".quick-action").forEach(button => {
        button.addEventListener("click", async () => {
            const mode = button.dataset.mode
            getElement("modeSelect").value = mode
            updateUseRagState()

            if (getElement("userInput").value.trim()) {
                await sendMessage(mode)
            } else {
                getElement("userInput").focus()
            }
        })
    })
}


document.addEventListener("DOMContentLoaded", () => {
    getElement("modeSelect").addEventListener("change", updateUseRagState)
    getElement("sendButton").addEventListener("click", () => sendMessage())
    getElement("uploadButton").addEventListener("click", uploadPDF)
    getElement("clearChatButton").addEventListener("click", clearConversation)
    getElement("flashcardsPanel").addEventListener("click", handleFlashcardClick)
    setupQuickActions()

    document.querySelectorAll(".insight-tab").forEach(tab => {
        tab.addEventListener("click", () => activateInsightTab(tab.dataset.tab))
    })

    getElement("userInput").addEventListener("keydown", event => {
        if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
            sendMessage()
        }
    })

    updateUseRagState()
})
