/**
 * Shared utilities for visualization modules.
 */

/**
 * Escape HTML special characters for safe rendering.
 */
function escapeHtml(text) {
    if (!text) return '';
    return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

/**
 * Protect math blocks from markdown parser.
 * Markdown treats _ as emphasis, breaking LaTeX like $h_t$ -> $h<em>t$
 * @returns {{ markdown: string, blocks: string[] }}
 */
function protectMathBlocks(markdown) {
    const blocks = [];
    const placeholder = (match) => {
        blocks.push(match);
        return `MATH_BLOCK_${blocks.length - 1}`;
    };
    // Extract display math ($$...$$) then inline math ($...$)
    markdown = markdown.replace(/\$\$[\s\S]+?\$\$/g, placeholder);
    markdown = markdown.replace(/\$[^\$\n]+?\$/g, placeholder);
    return { markdown, blocks };
}

/**
 * Restore math blocks after markdown parsing.
 */
function restoreMathBlocks(html, blocks) {
    blocks.forEach((block, i) => {
        html = html.replace(`MATH_BLOCK_${i}`, block);
    });
    return html;
}

/**
 * Apply centered moving average to smooth data.
 * @param {number[]} data - Input array
 * @param {number} windowSize - Window size (should be odd for centered average)
 * @returns {number[]} Smoothed array (same length as input)
 */
function smoothData(data, windowSize = 3) {
    if (data.length < windowSize) return data;
    const half = Math.floor(windowSize / 2);
    const result = [];
    for (let i = 0; i < data.length; i++) {
        const start = Math.max(0, i - half);
        const end = Math.min(data.length, i + half + 1);
        const slice = data.slice(start, end);
        result.push(slice.reduce((a, b) => a + b, 0) / slice.length);
    }
    return result;
}

// Export to global scope
window.escapeHtml = escapeHtml;
window.protectMathBlocks = protectMathBlocks;
window.restoreMathBlocks = restoreMathBlocks;
window.smoothData = smoothData;
