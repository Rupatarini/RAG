const API_BASE = "http://localhost:5000";

const uploadForm = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const uploadStatus = document.getElementById("upload-status");

const chatForm = document.getElementById("chat-form");
const questionInput = document.getElementById("question");
const chatMessages = document.getElementById("chat-messages");
const askBtn = document.getElementById("ask-btn");
const newChatBtn = document.getElementById("new-chat-btn");

function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

/* Creates a user/assistant row */
function createMessageRow(role) {
    const row = document.createElement("div");
    row.classList.add("message-row", role);

    const avatar = document.createElement("div");
    avatar.classList.add("avatar");
    avatar.classList.add(role === "user" ? "user-avatar" : "assistant-avatar");
    avatar.textContent = role === "user" ? "U" : "A";

    const bubble = document.createElement("div");
    bubble.classList.add("message-bubble");

    const name = document.createElement("div");
    name.classList.add("message-name");
    name.textContent = role === "user" ? "You" : "Assistant";

    bubble.appendChild(name);
    row.appendChild(avatar);
    row.appendChild(bubble);
    return { row, bubble };
}

/* Append sources list */
function appendSourcesToBubble(bubble, sources) {
    const heading = document.createElement("div");
    heading.innerText = "Sources:";
    heading.style.marginTop = "8px";
    heading.style.fontSize = "12px";
    bubble.appendChild(heading);

    const ul = document.createElement("ul");
    ul.style.fontSize = "12px";
    ul.style.marginTop = "4px";

    sources.forEach(s => {
        const li = document.createElement("li");
        const fname = s.metadata?.filename || "doc";
        const idx = s.metadata?.chunk_index ?? "-";
        li.textContent = `${fname} (chunk ${idx})`;
        ul.appendChild(li);
    });

    bubble.appendChild(ul);
}

/* Normal message bubble */
function addMessage(role, text, sources = null) {
    const { row, bubble } = createMessageRow(role);

    const msg = document.createElement("div");
    msg.classList.add("message-text");
    msg.textContent = text;
    bubble.appendChild(msg);

    if (role === "assistant" && sources) {
        appendSourcesToBubble(bubble, sources);
    }

    chatMessages.appendChild(row);
    scrollToBottom();
    return row;
}

/* Typing indicator bubble */
function addTypingIndicator() {
    const { row, bubble } = createMessageRow("assistant");
    const typingDiv = document.createElement("div");
    typingDiv.classList.add("typing");
    for (let i = 0; i < 3; i++) {
        const dot = document.createElement("div");
        dot.classList.add("dot");
        typingDiv.appendChild(dot);
    }
    bubble.appendChild(typingDiv);
    chatMessages.appendChild(row);
    scrollToBottom();
    return row;
}

/* Typewriter effect */
function typeWriter(element, text, speed = 20, done) {
    let i = 0;
    function step() {
        if (i < text.length) {
            element.textContent += text.charAt(i);
            i++;
            scrollToBottom();
            setTimeout(step, speed);
        } else {
            if (done) done();
        }
    }
    step();
}

/* Assistant message with typing */
function addAssistantMessageTypewriter(text, sources = null) {
    const { row, bubble } = createMessageRow("assistant");
    const msg = document.createElement("div");
    msg.classList.add("message-text");
    bubble.appendChild(msg);
    chatMessages.appendChild(row);

    typeWriter(msg, text, 16, () => {
        if (sources) {
            appendSourcesToBubble(bubble, sources);
            scrollToBottom();
        }
    });
}

/* Welcome message */
function addWelcomeMessage() {
    addAssistantMessageTypewriter(
        "Hi 👋, upload a PDF or text file and then ask me anything about it."
    );
}

/* Upload handler */
uploadForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!fileInput.files[0]) return;

    uploadStatus.innerText = "Uploading & indexing...";

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    try {
        const res = await fetch(`${API_BASE}/upload`, {
            method: "POST",
            body: formData,
        });

        const data = await res.json();

        if (res.ok) {
            uploadStatus.innerText = `Indexed ${data.chunks_indexed} chunks.`;
            addAssistantMessageTypewriter(
                `Your document "${fileInput.files[0].name}" has been uploaded and indexed.`
            );
        } else {
            uploadStatus.innerText = data.error || "Upload failed.";
            addMessage("assistant", data.error);
        }
    } catch {
        uploadStatus.innerText = "Network error";
    }
});

/* Ask handler */
chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const question = questionInput.value.trim();
    if (!question) return;

    addMessage("user", question);
    askBtn.disabled = true;
    questionInput.value = "";

    const typingRow = addTypingIndicator();

    try {
        const res = await fetch(`${API_BASE}/ask`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question }),
        });

        const data = await res.json();

        if (typingRow.parentNode) chatMessages.removeChild(typingRow);

        if (res.ok) {
            addAssistantMessageTypewriter(data.answer, data.sources);
        } else {
            addAssistantMessageTypewriter(data.error);
        }
    } catch {
        if (typingRow.parentNode) chatMessages.removeChild(typingRow);
        addAssistantMessageTypewriter("Network error");
    } finally {
        askBtn.disabled = false;
    }
});

/* Enter = send, Shift+Enter = newline */
questionInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        chatForm.dispatchEvent(new Event("submit"));
    }
});

questionInput.addEventListener("input", () => {
    questionInput.style.height = "auto";
    questionInput.style.height = Math.min(questionInput.scrollHeight, 120) + "px";
});

/* New Chat */
newChatBtn.addEventListener("click", () => {
    chatMessages.innerHTML = "";
    addWelcomeMessage();
});

/* Initial */
addWelcomeMessage();
