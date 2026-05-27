const API_BASE_URL = "http://127.0.0.1:8000"


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


function renderTrace(title, items) {
    if (!items || items.length === 0) {
        return ""
    }

    const listItems = items
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


function traceIncludes(trace, text) {
    return Array.isArray(trace) && trace.some(item => String(item).includes(text))
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
    const traceHtml = renderTrace("执行路径", data.trace)
    const answerHtml = renderAnswer(data)

    getElement("chatBox").innerHTML += `
        <div class="ai-message">
            <div class="response-meta">
                <span>模式：${escapeHtml(data.mode || "")}</span>
                <span>模型：${escapeHtml(data.model || "")}</span>
            </div>
            ${answerHtml}
            ${sourcesHtml}
            ${planHtml}
            ${traceHtml}
        </div>
    `
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

        getElement("userInput").value = ""
        appendUserMessage(requestBody.message)
        appendChatResponse(data)
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
    updateUseRagState()
})
