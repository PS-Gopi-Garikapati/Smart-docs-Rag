/**
 * Smart Document Assistant - Main Client Application Logic.
 * Manages UI interactions, file uploads, parameter bindings, API calls,
 * and chat history rendering.
 */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Element Selectors
    const elements = {
        dropzone: document.getElementById("dropzone"),
        fileInput: document.getElementById("file-input"),
        btnSelectFiles: document.getElementById("btn-select-files"),
        uploadProgressBox: document.getElementById("upload-progress-box"),
        uploadProgressBar: document.getElementById("upload-progress-bar"),
        uploadProgressFilename: document.getElementById("upload-progress-filename"),
        uploadProgressPercent: document.getElementById("upload-progress-percent"),
        documentList: document.getElementById("document-list"),
        docCountBadge: document.getElementById("doc-count-badge"),
        
        // Hyperparameter inputs & API Key
        tempSlider: document.getElementById("temperature-slider"),
        tempVal: document.getElementById("temp-val"),
        topPSlider: document.getElementById("top-p-slider"),
        topPVal: document.getElementById("top-p-val"),
        topKInput: document.getElementById("top-k-input"),

        // Chat & Question elements

        askForm: document.getElementById("ask-form"),
        questionInput: document.getElementById("question-input"),
        btnAsk: document.getElementById("btn-ask"),
        chatHistory: document.getElementById("chat-history"),
        welcomeBanner: document.getElementById("welcome-banner"),
        btnExportChat: document.getElementById("btn-export-chat"),
        btnClearChat: document.getElementById("btn-clear-chat"),
        btnClearAll: document.getElementById("btn-clear-all"),

        // Toast container
        toastContainer: document.getElementById("toast-container")
    };

    // State Variables
    let isUploading = false;
    let isQuerying = false;

    // Initialize Event Listeners
    initEventListeners();
    fetchDocumentInventory();


    /**
     * Attaches DOM event listeners.
     */
    function initEventListeners() {
        // Slider value display updates
        elements.tempSlider.addEventListener("input", (e) => {
            elements.tempVal.textContent = parseFloat(e.target.value).toFixed(2);
        });

        elements.topPSlider.addEventListener("input", (e) => {
            elements.topPVal.textContent = parseFloat(e.target.value).toFixed(2);
        });



        // File Selection & Drag-and-Drop
        elements.btnSelectFiles.addEventListener("click", () => elements.fileInput.click());
        elements.fileInput.addEventListener("change", handleFileSelect);

        elements.dropzone.addEventListener("dragover", (e) => {
            e.preventDefault();
            elements.dropzone.classList.add("drag-over");
        });

        elements.dropzone.addEventListener("dragleave", () => {
            elements.dropzone.classList.remove("drag-over");
        });

        elements.dropzone.addEventListener("drop", (e) => {
            e.preventDefault();
            elements.dropzone.classList.remove("drag-over");
            if (e.dataTransfer.files.length > 0) {
                uploadFiles(e.dataTransfer.files);
            }
        });

        // Question Submission
        elements.askForm.addEventListener("submit", handleQuestionSubmit);

        // Enter key in text area triggers submission (Shift+Enter for new line)
        elements.questionInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                elements.askForm.dispatchEvent(new Event("submit"));
            }
        });

        // Action buttons
        if (elements.btnExportChat) {
            elements.btnExportChat.addEventListener("click", exportChatTranscript);
        }
        elements.btnClearChat.addEventListener("click", clearChatFeed);
        elements.btnClearAll.addEventListener("click", clearAllDocuments);
    }

    /**
     * Handles file input change event.
     */
    function handleFileSelect(e) {
        if (e.target.files.length > 0) {
            uploadFiles(e.target.files);
            elements.fileInput.value = "";
        }
    }

    /**
     * Helper to return dynamic FontAwesome icon based on file extension.
     */
    function getFileIconClass(filename) {
        const ext = (filename || "").split('.').pop().toLowerCase();
        switch (ext) {
            case 'pdf':
                return 'fa-solid fa-file-pdf text-danger';
            case 'docx':
            case 'doc':
                return 'fa-solid fa-file-word text-primary';
            case 'csv':
                return 'fa-solid fa-file-csv text-success';
            case 'txt':
            case 'md':
            case 'log':
            case 'text':
                return 'fa-solid fa-file-lines text-warning';
            case 'json':
                return 'fa-solid fa-file-code text-info';
            default:
                return 'fa-solid fa-file text-accent';
        }
    }

    /**
     * Uploads selected document files to backend API (/api/upload).
     */
    async function uploadFiles(files) {
        if (isUploading) return;

        const allowedExts = [".pdf", ".docx", ".doc", ".csv", ".txt", ".md", ".json", ".log", ".text"];
        const validFiles = Array.from(files).filter(f => {
            const ext = "." + f.name.split('.').pop().toLowerCase();
            return allowedExts.includes(ext);
        });

        if (validFiles.length === 0) {
            showToast("Please select supported documents (PDF, Word, CSV, TXT, MD, JSON).", "error");
            return;
        }

        isUploading = true;
        showUploadProgress(validFiles[0].name, 20);

        const formData = new FormData();
        validFiles.forEach(file => formData.append("files", file));

        try {
            updateUploadProgress(50);
            const response = await fetch("/api/upload", {
                method: "POST",
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || "Upload failed.");
            }

            updateUploadProgress(100);
            showToast(`Successfully indexed ${data.total_chunks_indexed} text chunks from ${validFiles.length} file(s)!`, "success");
            fetchDocumentInventory();

        } catch (error) {
            showToast(error.message, "error");
        } finally {
            setTimeout(hideUploadProgress, 1000);
            isUploading = false;
        }
    }

    /**
     * Fetches current document inventory from backend (/api/documents).
     */
    async function fetchDocumentInventory() {
        try {
            const response = await fetch("/api/documents");
            const data = await response.json();

            if (response.ok && data.status === "success") {
                renderDocumentList(data.documents);
            }
        } catch (error) {
            console.error("Failed to fetch documents:", error);
        }
    }

    /**
     * Renders document inventory list in left panel with individual delete icons.
     */
    function renderDocumentList(documents) {
        elements.documentList.innerHTML = "";

        if (!documents || documents.length === 0) {
            elements.docCountBadge.textContent = "0 Docs";
            elements.documentList.innerHTML = `
                <li class="empty-list-msg">No documents uploaded yet. Upload a file to start asking questions.</li>
            `;
            return;
        }

        elements.docCountBadge.textContent = `${documents.length} Doc${documents.length > 1 ? 's' : ''}`;

        documents.forEach(doc => {
            const li = document.createElement("li");
            li.className = "doc-item";
            const iconClass = getFileIconClass(doc.filename);
            li.innerHTML = `
                <div class="doc-name" title="${escapeHtml(doc.filename)}">
                    <i class="${iconClass}"></i>
                    <span>${escapeHtml(doc.filename)}</span>
                </div>
                <div class="doc-item-meta">
                    <span class="doc-chunks">${doc.chunk_count} Chunks</span>
                    <button class="btn-delete-doc" title="Delete document" data-filename="${escapeHtml(doc.filename)}">
                        <i class="fa-solid fa-trash-can"></i>
                    </button>
                </div>
            `;
            
            // Attach single document delete event
            const btnDelete = li.querySelector(".btn-delete-doc");
            btnDelete.addEventListener("click", (e) => {
                e.stopPropagation();
                deleteSingleDocument(doc.filename);
            });

            elements.documentList.appendChild(li);
        });
    }

    /**
     * Deletes a single uploaded document file.
     */
    async function deleteSingleDocument(filename) {
        if (!confirm(`Are you sure you want to remove '${filename}' and delete its vector chunks?`)) {
            return;
        }

        try {
            const response = await fetch(`/api/documents/${encodeURIComponent(filename)}`, {
                method: "DELETE"
            });
            const data = await response.json();

            if (response.ok) {
                showToast(`Removed '${filename}' from vector database.`, "success");
                fetchDocumentInventory();
            } else {
                throw new Error(data.detail || "Failed to delete document.");
            }
        } catch (error) {
            showToast(error.message, "error");
        }
    }

    /**
     * Handles question submission form submit.
     */
    async function handleQuestionSubmit(e) {
        e.preventDefault();

        const questionText = elements.questionInput.value.trim();
        if (!questionText || isQuerying) return;

        // Hide welcome banner on first question
        if (elements.welcomeBanner) {
            elements.welcomeBanner.style.display = "none";
        }

        // Gather hyperparameters
        const temperature = parseFloat(elements.tempSlider.value);
        const top_p = parseFloat(elements.topPSlider.value);
        const top_k = parseInt(elements.topKInput.value) || 3;

        // Append User Message to Chat Feed
        appendUserMessage(questionText);
        elements.questionInput.value = "";

        // Append Assistant Loading Indicator
        const loadingMsgId = appendLoadingMessage();
        setQueryState(true);

        try {
            const response = await fetch("/api/query", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    question: questionText,
                    temperature: temperature,
                    top_p: top_p,
                    top_k: top_k
                })
            });

            const data = await response.json();
            removeLoadingMessage(loadingMsgId);

            if (!response.ok) {
                throw new Error(data.detail || "Query execution failed.");
            }

            // Append Assistant Answer Bubble
            appendAssistantMessage(data);

        } catch (error) {
            removeLoadingMessage(loadingMsgId);
            appendErrorMessage(error.message);
            showToast(error.message, "error");
        } finally {
            setQueryState(false);
        }
    }

    /**
     * Appends user message bubble to chat feed.
     */
    function appendUserMessage(text) {
        const msgDiv = document.createElement("div");
        msgDiv.className = "chat-msg user-msg";
        msgDiv.innerHTML = `
            <div class="msg-bubble">${escapeHtml(text)}</div>
            <div class="msg-meta">
                <span>You</span> • <span>${getCurrentTimestamp()}</span>
            </div>
        `;
        elements.chatHistory.appendChild(msgDiv);
        scrollToBottom();
    }

    /**
     * Appends assistant loading bubble.
     */
    function appendLoadingMessage() {
        const id = "loading-" + Date.now();
        const msgDiv = document.createElement("div");
        msgDiv.id = id;
        msgDiv.className = "chat-msg assistant-msg";
        msgDiv.innerHTML = `
            <div class="msg-bubble">
                <i class="fa-solid fa-spinner fa-spin text-accent"></i> Searching vector database & generating answer...
            </div>
        `;
        elements.chatHistory.appendChild(msgDiv);
        scrollToBottom();
        return id;
    }

    function removeLoadingMessage(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    /**
     * Appends assistant answer bubble with source citations accordion and Copy button.
     */
    function appendAssistantMessage(data) {
        const msgDiv = document.createElement("div");
        msgDiv.className = "chat-msg assistant-msg";

        const parsedContent = window.MarkdownFormatter.parse(data.answer);

        // Build citations HTML if sources exist
        let citationsHtml = "";
        if (data.retrieved_sources && data.retrieved_sources.length > 0) {
            const citationCards = data.retrieved_sources.map(s => `
                <div class="citation-card">
                    <div class="citation-title">
                        <span><i class="fa-solid fa-file-lines"></i> ${escapeHtml(s.source)} (Page ${s.page})</span>
                        <span>Sim: ${(s.similarity_score * 100).toFixed(1)}%</span>
                    </div>
                    <div class="citation-snippet">"${escapeHtml(s.snippet)}"</div>
                </div>
            `).join("");

            citationsHtml = `
                <div class="citations-wrapper">
                    <div class="citations-header" onclick="this.nextElementSibling.classList.toggle('hidden')">
                        <i class="fa-solid fa-square-poll-vertical"></i> View ${data.retrieved_sources.length} Context Sources <i class="fa-solid fa-chevron-down icon-sm"></i>
                    </div>
                    <div class="citations-list hidden">
                        ${citationCards}
                    </div>
                </div>
            `;
        }

        msgDiv.innerHTML = `
            <div class="msg-bubble">
                <div class="bubble-header-row">
                    <span class="assistant-label"><i class="fa-solid fa-robot"></i> Smart Assistant</span>
                    <button class="btn-copy-answer" title="Copy text to clipboard">
                        <i class="fa-solid fa-copy"></i> Copy
                    </button>
                </div>
                <div class="answer-text">${parsedContent}</div>
                ${citationsHtml}
            </div>
            <div class="msg-meta">
                <span>Assistant</span> • <span>${data.execution_time_seconds}s</span> • 
                <span>Temp: ${data.parameters_used.temperature} | Top-P: ${data.parameters_used.top_p} | Top-K: ${data.parameters_used.top_k}</span>
            </div>
        `;

        // Attach Copy Answer functionality
        const btnCopy = msgDiv.querySelector(".btn-copy-answer");
        btnCopy.addEventListener("click", () => {
            navigator.clipboard.writeText(data.answer).then(() => {
                btnCopy.innerHTML = `<i class="fa-solid fa-check"></i> Copied!`;
                setTimeout(() => {
                    btnCopy.innerHTML = `<i class="fa-solid fa-copy"></i> Copy`;
                }, 2000);
            }).catch(() => {
                showToast("Failed to copy text.", "error");
            });
        });

        elements.chatHistory.appendChild(msgDiv);
        scrollToBottom();
    }

    function appendErrorMessage(errorMsg) {
        const msgDiv = document.createElement("div");
        msgDiv.className = "chat-msg assistant-msg";
        msgDiv.innerHTML = `
            <div class="msg-bubble text-danger">
                <i class="fa-solid fa-triangle-exclamation"></i> ${escapeHtml(errorMsg)}
            </div>
        `;
        elements.chatHistory.appendChild(msgDiv);
        scrollToBottom();
    }

    /**
     * Clears vector store and deletes all documents.
     */
    async function clearAllDocuments() {
        if (!confirm("Are you sure you want to clear all uploaded PDF documents and reset the vector store?")) {
            return;
        }

        try {
            const response = await fetch("/api/clear", { method: "DELETE" });
            const data = await response.json();

            if (response.ok) {
                showToast("Vector store and uploaded files cleared.", "success");
                fetchDocumentInventory();
                clearChatFeed();
            } else {
                throw new Error(data.detail || "Failed to clear store.");
            }
        } catch (error) {
            showToast(error.message, "error");
        }
    }

    /**
     * Exports chat conversation history as a Markdown file.
     */
    function exportChatTranscript() {
        const messages = elements.chatHistory.querySelectorAll(".chat-msg");
        if (messages.length === 0) {
            showToast("No conversation to export.", "info");
            return;
        }

        let markdownContent = `# Smart Document Assistant - Conversation Transcript\n\n`;
        markdownContent += `*Exported on ${new Date().toLocaleString()}*\n\n---\n\n`;

        messages.forEach(msg => {
            const isUser = msg.classList.contains("user-msg");
            const bubble = msg.querySelector(".msg-bubble");
            if (!bubble) return;

            if (isUser) {
                markdownContent += `### 👤 User:\n${bubble.textContent.trim()}\n\n`;
            } else {
                const answerText = msg.querySelector(".answer-text");
                const text = answerText ? answerText.textContent.trim() : bubble.textContent.trim();
                markdownContent += `### 🤖 Assistant:\n${text}\n\n---\n\n`;
            }
        });

        const blob = new Blob([markdownContent], { type: "text/markdown" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `smart_docs_transcript_${Date.now()}.md`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        showToast("Transcript exported successfully!", "success");
    }

    function clearChatFeed() {
        elements.chatHistory.innerHTML = "";
        if (elements.welcomeBanner) {
            elements.welcomeBanner.style.display = "block";
            elements.chatHistory.appendChild(elements.welcomeBanner);
        }
    }

    // UI Helper Utilities
    function setQueryState(loading) {
        isQuerying = loading;
        elements.btnAsk.disabled = loading;
        elements.btnAsk.querySelector(".btn-text").textContent = loading ? "Thinking..." : "Ask Question";
    }

    function showUploadProgress(filename, percent) {
        elements.uploadProgressBox.classList.remove("hidden");
        elements.uploadProgressFilename.textContent = filename;
        updateUploadProgress(percent);
    }

    function updateUploadProgress(percent) {
        elements.uploadProgressBar.style.width = `${percent}%`;
        elements.uploadProgressPercent.textContent = `${percent}%`;
    }

    function hideUploadProgress() {
        elements.uploadProgressBox.classList.add("hidden");
        elements.uploadProgressBar.style.width = "0%";
    }

    function scrollToBottom() {
        elements.chatHistory.scrollTop = elements.chatHistory.scrollHeight;
    }

    function getCurrentTimestamp() {
        return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }

    function escapeHtml(str) {
        return (str || "")
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;");
    }

    function showToast(message, type = "info") {
        const toast = document.createElement("div");
        toast.className = `toast ${type}`;
        
        let icon = "fa-circle-info";
        if (type === "success") icon = "fa-circle-check";
        if (type === "error") icon = "fa-triangle-exclamation";

        toast.innerHTML = `<i class="fa-solid ${icon}"></i> <span>${escapeHtml(message)}</span>`;
        elements.toastContainer.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = "0";
            setTimeout(() => toast.remove(), 300);
        }, 3500);
    }
});

