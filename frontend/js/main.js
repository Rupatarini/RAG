const API_BASE = "https://vamsi1103-rag-backend.hf.space";

/* -------------------------------
   SESSION HANDLING (CRITICAL)
--------------------------------*/
let SESSION_ID = crypto.randomUUID();

function resetSession() {
    SESSION_ID = crypto.randomUUID();
    console.log("New session:", SESSION_ID);
}

/* -------------------------------
   DOM ELEMENTS
--------------------------------*/
const uploadForm = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const uploadStatus = document.getElementById("upload-status");

const chatForm = document.getElementById("chat-form");
const questionInput = document.getElementById("question");
const chatMessages = document.getElementById("chat-messages");
const askBtn = document.getElementById("ask-btn");
const newChatBtn = document.getElementById("new-chat-btn");

/* -------------------------------
   UI HELPERS
--------------------------------*/
function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function createMessageRow(role) {
    const row = document.createElement("div");
    row.classList.add("message-row", role);

    const avatar = document.createElement("div");
    avatar.classList.add("avatar", role === "user" ? "user-avatar" : "assistant-avatar");
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

function appendSourcesToBubble(bubble, sources) {
    if (!sources || sources.length === 0) return;

    const heading = document.createElement("div");
    heading.innerText = "Sources:";
    heading.style.marginTop = "8px";
    heading.style.fontSize = "12px";
    bubble.appendChild(heading);

    const ul = document.createElement("ul");
    ul.style.fontSize = "12px";

    sources.forEach(s => {
        const li = document.createElement("li");
        li.textContent = s.filename || "document";
        ul.appendChild(li);
    });

    bubble.appendChild(ul);
}

function addMessage(role, text, sources = null) {
    const { row, bubble } = createMessageRow(role);
    const msg = document.createElement("div");
    msg.classList.add("message-text");
    msg.textContent = text;
    bubble.appendChild(msg);

    if (role === "assistant") appendSourcesToBubble(bubble, sources);
    chatMessages.appendChild(row);
    scrollToBottom();
}

function addTypingIndicator() {
    const { row, bubble } = createMessageRow("assistant");
    const typing = document.createElement("div");
    typing.classList.add("typing");
    typing.innerHTML = "<span></span><span></span><span></span>";
    bubble.appendChild(typing);
    chatMessages.appendChild(row);
    scrollToBottom();
    return row;
}

function typeWriter(el, text, speed = 16, done) {
    let i = 0;
    function step() {
        if (i < text.length) {
            el.textContent += text.charAt(i++);
            scrollToBottom();
            setTimeout(step, speed);
        } else if (done) done();
    }
    step();
}

function addAssistantMessageTypewriter(text, sources = null) {
    const { row, bubble } = createMessageRow("assistant");
    const msg = document.createElement("div");
    msg.classList.add("message-text");
    bubble.appendChild(msg);
    chatMessages.appendChild(row);

    typeWriter(msg, text, 16, () => {
        appendSourcesToBubble(bubble, sources);
    });
}

/* -------------------------------
   WELCOME
--------------------------------*/
function addWelcomeMessage() {
    addAssistantMessageTypewriter(
        "Hi 👋 Upload one or more PDFs, then ask questions based only on them."
    );
}

/* -------------------------------
   UPLOAD HANDLER
--------------------------------*/
uploadForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!fileInput.files.length) return;

    uploadStatus.innerText = "Uploading & indexing...";

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    try {
        const res = await fetch(`${API_BASE}/upload`, {
            method: "POST",
            headers: {
                "X-Session-ID": SESSION_ID
            },
            body: formData
        });

        const data = await res.json();

        if (res.ok) {
            uploadStatus.innerText = `Indexed ${data.chunks_indexed} chunks`;
            addAssistantMessageTypewriter(
                `📄 "${fileInput.files[0].name}" indexed successfully.`
            );
        } else {
            uploadStatus.innerText = data.error;
        }
    } catch {
        uploadStatus.innerText = "Network error";
    }
});

/* -------------------------------
   ASK HANDLER
--------------------------------*/
chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const question = questionInput.value.trim();
    if (!question) return;

    addMessage("user", question);
    questionInput.value = "";
    askBtn.disabled = true;

    const typingRow = addTypingIndicator();

    try {
        const res = await fetch(`${API_BASE}/ask`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "X-Session-ID": SESSION_ID
            },
            body: JSON.stringify({ question })
        });

        const data = await res.json();
        typingRow.remove();

        if (res.ok) {
            addAssistantMessageTypewriter(data.answer, data.sources);
        } else {
            addAssistantMessageTypewriter(data.error);
        }
    } catch {
        typingRow.remove();
        addAssistantMessageTypewriter("Network error");
    } finally {
        askBtn.disabled = false;
    }
});

/* -------------------------------
   NEW CHAT (RESET SESSION)
--------------------------------*/
newChatBtn.addEventListener("click", () => {
    resetSession();              // 🔥 NEW SESSION
    chatMessages.innerHTML = ""; // Clear UI
    uploadStatus.innerText = "";
    addWelcomeMessage();
});

/* -------------------------------
   INIT
--------------------------------*/
addWelcomeMessage();
