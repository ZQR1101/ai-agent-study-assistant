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
    getElement("chatBox").innerHTML = ""
    getElement("userInput").value = ""
    getElement("loadingText").style.display = "none"
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
    getElement("chatBox").innerHTML += `
        <div class="user-message">
            <b>你：</b>${escapeHtml(message)}
        </div>
    `
}


function appendErrorMessage(message) {
    getElement("chatBox").innerHTML += `
        <div class="ai-message error-message">
            <b>错误：</b>${escapeHtml(message)}
        </div>
    `
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


function renderTrace(title, trace) {
    if (!trace || trace.length === 0) {
        return ""
    }

    if (typeof trace[0] === "object" && !Array.isArray(trace[0])) {
        const blocksHtml = trace.map(block => {
            const blockTitle = escapeHtml(block.title || "执行信息")
            const blockItems = Array.isArray(block.items) ? block.items : []
            const listItems = blockItems
                .map(item => `<li>${escapeHtml(item)}</li>`)
                .join("")

            return `
                <div class="trace-block">
                    <h4>${blockTitle}</h4>
                    <ul>${listItems}</ul>
                </div>
            `
        }).join("")

        return `
            <div class="meta-section">
                <h3>${title}</h3>
                ${blocksHtml}
            </div>
        `
    }

    const listItems = trace
        .map(item => `<li>${escapeHtml(item)}</li>`)
        .join("")

    return `
        <div class="meta-section">
            <h3>${title}</h3>
            <ul>${listItems}</ul>
        </div>
    `
}


function renderSources(sources) {
    if (!sources || sources.length === 0) {
        return ""
    }

    const listItems = sources.map((source, index) => {
        if (typeof source === "string") {
            return `<li>${escapeHtml(source)}</li>`
        }

        const sourceName = escapeHtml(source.source || "未知来源")
        const score = source.score === null || source.score === undefined
            ? "无"
            : Number(source.score).toFixed(4)
        const text = escapeHtml(source.text || source.snippet || "")

        return `
            <li class="source-item">
                <div><b>${index + 1}. ${sourceName}</b></div>
                <div>相似度：${score}</div>
                <div>命中片段：${text}</div>
            </li>
        `
    }).join("")

    return `
        <div class="meta-section">
            <h3>参考来源</h3>
            <ol class="source-list">${listItems}</ol>
        </div>
    `
}


function renderPlan(plan) {
    if (!plan || plan.length === 0) {
        return ""
    }

    const listItems = plan.map((step, index) => {
        const tool = escapeHtml(step.tool || "unknown")
        const input = escapeHtml(step.input || "")
        const reason = escapeHtml(step.reason || "")

        return `
            <li class="plan-item">
                <div><b>${index + 1}. ${tool}</b></div>
                <div>输入：${input}</div>
                ${reason ? `<div>原因：${reason}</div>` : ""}
            </li>
        `
    }).join("")

    return `
        <div class="meta-section">
            <h3>Agent 计划</h3>
            <ol class="plan-list">${listItems}</ol>
        </div>
    `
}


function renderFlashcards(flashcards) {
    if (!flashcards || flashcards.length === 0) {
        return ""
    }

    const cardsHtml = flashcards.map((card, index) => {
        const front = escapeHtml(card.front || "")
        const back = escapeHtml(card.back || "")
        const rawDifficulty = String(card.difficulty || "medium").toLowerCase()
        const difficulty = ["easy", "medium", "hard"].includes(rawDifficulty)
            ? rawDifficulty
            : "medium"
        const difficultyText = escapeHtml(difficulty)
        const difficultyClass = `flashcard-difficulty flashcard-difficulty-${difficulty}`
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
                data-difficulty="${difficultyText}"
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
                            <span class="${difficultyClass}">${difficultyText}</span>
                            ${tagsHtml}
                        </div>
                    </div>
                    <div class="flashcard-face flashcard-back">
                        <div class="flashcard-label">背面</div>
                        <div class="flashcard-content">${back}</div>
                        <div class="flashcard-meta">
                            <span class="${difficultyClass}">${difficultyText}</span>
                            ${tagsHtml}
                        </div>
                    </div>
                </div>
            </article>
        `
    }).join("")

    return `
        <div class="flashcard-section">
            <div class="flashcard-section-header">
                <h3>记忆卡片</h3>
            </div>
            <p class="flashcard-hint">点击卡片翻转查看答案</p>
            <div class="flashcard-grid">${cardsHtml}</div>
        </div>
    `
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
        const testLine = line ? `${line}${normalizedText.includes(" ") ? " " : ""}${word}` : word
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
    drawRoundedRect(context, x, y, width, height, 18)
    context.fill()
    context.stroke()

    context.fillStyle = "#64748b"
    context.font = "700 22px Arial"
    context.fillText(title, x + 24, y + 36)

    context.fillStyle = "#0f172a"
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
        : "无"
    const meta = `标签：${tags}    难度：${card.difficulty || "medium"}`
    const isFront = side === "front"

    canvas.width = width * scale
    canvas.height = height * scale
    context.scale(scale, scale)
    context.fillStyle = "#f8fafc"
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
        isFront ? "#dbe3ef" : "#a7f3d0",
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

    let title = "学习内容："
    let noteHtml = ""

    if (ragPassed) {
        title = "知识库学习内容："
    } else if (ragFailed && fallbackUsed) {
        title = "普通模型学习内容："
        noteHtml = `
            <div class="fallback-note">
                知识库未找到可靠相关内容，本部分由普通模型生成。
            </div>
        `
    } else if (ragDisabled) {
        title = "学习内容："
    }

    const answerBody = rawAnswer.replace(/^知识内容：\s*/, "")

    return `
        <div class="answer">
            <h3 class="learning-answer-title">${title}</h3>
            ${noteHtml}
            <div>${escapeHtml(answerBody)}</div>
        </div>
    `
}


function appendChatResponse(data) {
    const sourcesHtml = renderSources(data.sources)
    const planHtml = renderPlan(data.plan)
    const flashcardsHtml = renderFlashcards(data.flashcards)
    const traceHtml = renderTrace("执行路径", data.trace)
    const answerHtml = renderAnswer(data)

    getElement("chatBox").innerHTML += `
        <div class="ai-message">
            <div class="response-meta">
                <span>模式：${escapeHtml(data.mode || "")}</span>
                <span>模型：${escapeHtml(data.model || "")}</span>
            </div>
            ${answerHtml}
            ${flashcardsHtml}
            ${sourcesHtml}
            ${planHtml}
            ${traceHtml}
        </div>
    `
}


async function handleChatBoxClick(event) {
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
        alert(data.message)
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


async function learnMode() {
    getElement("modeSelect").value = "learn"
    updateUseRagState()
    await sendMessage("learn")
}


async function ragMode() {
    getElement("modeSelect").value = "rag"
    getElement("useRagInput").checked = true
    updateUseRagState()
    await sendMessage("rag")
}


document.addEventListener("DOMContentLoaded", () => {
    getElement("modeSelect").addEventListener("change", updateUseRagState)
    getElement("chatBox").addEventListener("click", handleChatBoxClick)
    getElement("clearChatButton").addEventListener("click", clearConversation)
    updateUseRagState()
})
