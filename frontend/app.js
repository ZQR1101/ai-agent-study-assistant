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


function buildChatRequest(modeOverride) {
    const message = getElement("userInput").value.trim()

    return {
        message,
        mode: modeOverride || getElement("modeSelect").value,
        model: getElement("modelSelect").value,
        temperature: clampNumber(getElement("temperatureInput").value, 0.7, 0, 2),
        use_agent: getElement("useAgentInput").checked,
        use_rag: getElement("useRagInput").checked,
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


function renderList(title, items) {
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
    const sourcesHtml = renderList("参考来源", data.sources)
    const traceHtml = renderList("执行路径", data.trace)
    const answerHtml = renderAnswer(data)

    getElement("chatBox").innerHTML += `
        <div class="ai-message">
            <div class="response-meta">
                <span>模式：${escapeHtml(data.mode || "")}</span>
                <span>模型：${escapeHtml(data.model || "")}</span>
            </div>
            ${answerHtml}
            ${sourcesHtml}
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
    await sendMessage("learn")
}


async function ragMode() {
    getElement("modeSelect").value = "rag"
    getElement("useRagInput").checked = true
    await sendMessage("rag")
}
