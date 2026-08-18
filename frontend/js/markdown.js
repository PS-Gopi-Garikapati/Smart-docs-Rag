/**
 * Lightweight Client-Side Markdown Formatter.
 * Formats Markdown syntax into clean HTML safely without external library overhead.
 */

window.MarkdownFormatter = {
    parse: function(markdownText) {
        if (!markdownText) return "";

        let html = markdownText;

        // Escape dangerous HTML tags to prevent XSS
        html = html.replace(/&/g, "&amp;")
                   .replace(/</g, "&lt;")
                   .replace(/>/g, "&gt;");

        // Code blocks: ```code```
        html = html.replace(/```([\s\S]*?)```/g, function(match, code) {
            return `<pre class="code-block"><code>${code.trim()}</code></pre>`;
        });

        // Inline code: `code`
        html = html.replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>');

        // Bold text: **text** or __text__
        html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        html = html.replace(/__(.*?)__/g, '<strong>$1</strong>');

        // Italic text: *text* or _text_
        html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
        html = html.replace(/_(.*?)_/g, '<em>$1</em>');

        // Bullet lists: * item or - item
        html = html.replace(/^\s*[\*\-]\s+(.*)$/gim, '<li>$1</li>');
        html = html.replace(/(<li>.*<\/li>)/s, '<ul class="markdown-list">$1</ul>');

        // Line breaks to paragraphs
        html = html.replace(/\n\n/g, '</p><p>');
        html = html.replace(/\n/g, '<br>');

        return `<p>${html}</p>`;
    }
};
