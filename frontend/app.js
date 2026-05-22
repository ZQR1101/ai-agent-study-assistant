async function uploadPDF() {
    const fileInput = document.getElementById("pdfFile")
    const file = fileInput.files[0]

    if (!file) {
        alert("请先选择一个 PDF 文件")
        return
    }

    const formData = new FormData()
    formData.append("file", file)

    try {
        const response = await fetch(
            "http://127.0.0.1:8000/upload",
            {
                method: "POST",
                body: formData
            }
        )

        if (!response.ok) {
            throw new Error("上传失败：" + response.status)
        }

        const data = await response.json()
        alert(data.message)

    } catch (error) {
        alert("上传失败：" + error.message)
    }
}


async function sendMessage() {
    const userInput =
        document.getElementById("userInput").value

    if (!userInput.trim()) {
        alert("请输入内容")
        return
    }

    const loadingText =
        document.getElementById("loadingText")

    const chatBox =
        document.getElementById("chatBox")

    loadingText.style.display = "block"

    try {
        const response = await fetch(
            "http://127.0.0.1:8000/agent",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    text: userInput
                })
            }
        )

        if (!response.ok) {
            throw new Error("后端请求失败：" + response.status)
        }

        const data = await response.json()

        document.getElementById("userInput").value = ""

        chatBox.innerHTML += `
            <div class="user-message">
                <b>你：</b>${userInput}
            </div>
        `

        chatBox.innerHTML += `
            <div class="ai-message">
                <b>AI：</b>${data.result}
            </div>
        `

    } catch (error) {
        chatBox.innerHTML += `
            <div class="ai-message">
                <b>错误：</b>${error.message}
            </div>
        `
    } finally {
        loadingText.style.display = "none"
    }
}


async function learnMode() {
    const userInput =
        document.getElementById("userInput").value

    if (!userInput.trim()) {
        alert("请输入学习主题")
        return
    }

    const loadingText =
        document.getElementById("loadingText")

    const chatBox =
        document.getElementById("chatBox")

    loadingText.style.display = "block"

    try {
        const response = await fetch(
            "http://127.0.0.1:8000/learn",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    text: userInput
                })
            }
        )

        if (!response.ok) {
            throw new Error("学习模式请求失败：" + response.status)
        }

        const data = await response.json()

        document.getElementById("userInput").value = ""

        chatBox.innerHTML += `
            <div class="user-message">
                <b>你：</b>${userInput}
            </div>
        `

        chatBox.innerHTML += `
            <div class="learn-card">
                <h3>知识讲解</h3>
                <div>${data.knowledge}</div>

                <h3>总结</h3>
                <div>${data.summary}</div>

                <h3>练习题</h3>
                <div>${data.quiz}</div>

                <h3>下一步建议</h3>
                <div>${data.advice}</div>
            </div>
        `

    } catch (error) {
        chatBox.innerHTML += `
            <div class="ai-message">
                <b>错误：</b>${error.message}
            </div>
        `
    } finally {
        loadingText.style.display = "none"
    }
}

async function ragMode() {
    const userInput =
        document.getElementById("userInput").value

    if (!userInput.trim()) {
        alert("请输入知识库问题")
        return
    }

    const loadingText =
        document.getElementById("loadingText")

    const chatBox =
        document.getElementById("chatBox")

    loadingText.style.display = "block"

    try {
        const response = await fetch(
            "http://127.0.0.1:8000/rag",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    text: userInput
                })
            }
        )

        if (!response.ok) {
            throw new Error("知识库问答请求失败：" + response.status)
        }

        const data = await response.json()

        document.getElementById("userInput").value = ""

        const sourcesHtml = data.sources
            .map(source => `<li>📄 ${source}</li>`)
            .join("")

        chatBox.innerHTML += `
            <div class="user-message">
                <b>你：</b>${userInput}
            </div>
        `

        chatBox.innerHTML += `
            <div class="ai-message">
                <h3>知识库回答</h3>
                <div>${data.answer}</div>

                <h3>参考来源</h3>
                <ul>
                    ${sourcesHtml}
                </ul>
            </div>
        `

    } catch (error) {
        chatBox.innerHTML += `
            <div class="ai-message">
                <b>错误：</b>${error.message}
            </div>
        `
    } finally {
        loadingText.style.display = "none"
    }
}