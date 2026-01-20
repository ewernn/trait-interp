/**
 * Annotation utilities for converting text spans to character ranges.
 *
 * Input: Unified annotation format with text spans
 * Output: Character ranges for frontend highlighting
 *
 * Usage:
 *     const charRanges = window.annotations.spansToCharRanges(responseText, annotations);
 *     const html = window.annotations.applyHighlights(responseText, charRanges);
 */

/**
 * Convert text span annotations to character ranges.
 *
 * @param {string} response - Response text to search in
 * @param {Array<{span: string}>} annotations - List of annotation objects with "span" key
 * @returns {Array<[number, number]>} List of [start, end) character ranges
 */
function spansToCharRanges(response, annotations) {
    const ranges = [];

    for (const ann of annotations || []) {
        const span = ann.span;
        if (!span) continue;

        // Find first occurrence
        let start = response.indexOf(span);
        if (start !== -1) {
            ranges.push([start, start + span.length]);
        } else {
            // Try case-insensitive
            const lowerResponse = response.toLowerCase();
            const lowerSpan = span.toLowerCase();
            start = lowerResponse.indexOf(lowerSpan);
            if (start !== -1) {
                ranges.push([start, start + span.length]);
            }
        }
    }

    return ranges;
}

/**
 * Merge overlapping or adjacent ranges.
 *
 * @param {Array<[number, number]>} ranges - List of [start, end) tuples
 * @returns {Array<[number, number]>} Merged non-overlapping ranges
 */
function mergeRanges(ranges) {
    if (!ranges || ranges.length === 0) return [];

    const sorted = [...ranges].sort((a, b) => a[0] - b[0]);
    const merged = [[...sorted[0]]];

    for (const [start, end] of sorted.slice(1)) {
        const last = merged[merged.length - 1];
        if (start <= last[1]) {
            last[1] = Math.max(last[1], end);
        } else {
            merged.push([start, end]);
        }
    }

    return merged;
}

/**
 * Apply character range highlights to text, returning HTML.
 *
 * @param {string} text - Original text
 * @param {Array<[number, number]>} charRanges - Character ranges to highlight
 * @param {string} className - CSS class for highlight (default: "highlight")
 * @returns {string} HTML with <mark> tags
 */
function applyHighlights(text, charRanges, className = 'highlight') {
    if (!charRanges || charRanges.length === 0) {
        return window.escapeHtml(text).replace(/\n/g, '<br>');
    }

    const merged = mergeRanges(charRanges);
    let result = '';
    let pos = 0;

    for (const [start, end] of merged) {
        // Text before highlight
        if (start > pos) {
            result += window.escapeHtml(text.slice(pos, start)).replace(/\n/g, '<br>');
        }
        // Highlighted text
        result += `<mark class="${className}">` +
            window.escapeHtml(text.slice(start, end)).replace(/\n/g, '<br>') +
            '</mark>';
        pos = end;
    }

    // Remaining text
    if (pos < text.length) {
        result += window.escapeHtml(text.slice(pos)).replace(/\n/g, '<br>');
    }

    return result;
}

/**
 * Get spans for a specific response index from annotation data.
 *
 * @param {Object} annotations - Loaded annotation object with "annotations" array
 * @param {number} responseIdx - Response index to look up
 * @returns {Array<{span: string}>} Array of span objects, or empty array
 */
function getSpansForResponse(annotations, responseIdx) {
    for (const entry of annotations?.annotations || []) {
        if (entry.idx === responseIdx) {
            return entry.spans || [];
        }
    }
    return [];
}

/**
 * Load annotations from sibling file and convert to char ranges.
 *
 * @param {string} responsesPath - Path to responses JSON file
 * @returns {Promise<Array<Array<[number, number]>>>} Per-response char ranges (sparse map)
 */
async function loadAndConvertAnnotations(responsesPath) {
    // Derive annotations path: baseline.json -> baseline_annotations.json
    const annotationsPath = responsesPath.replace('.json', '_annotations.json');

    try {
        const [responsesResp, annotationsResp] = await Promise.all([
            fetch(responsesPath),
            fetch(annotationsPath)
        ]);

        if (!annotationsResp.ok) {
            return null; // No annotations file
        }

        const responses = await responsesResp.json();
        const annotations = await annotationsResp.json();

        // Convert each response's annotations to char ranges
        const allRanges = [];
        const responseList = Array.isArray(responses) ? responses : [responses];

        for (let i = 0; i < responseList.length; i++) {
            const response = responseList[i];
            const spans = getSpansForResponse(annotations, i);
            const charRanges = spansToCharRanges(response.response || '', spans);
            allRanges.push(charRanges);
        }

        return allRanges;
    } catch (e) {
        console.warn('Could not load annotations:', e);
        return null;
    }
}

// Export
window.annotations = {
    spansToCharRanges,
    getSpansForResponse,
    mergeRanges,
    applyHighlights,
    loadAndConvertAnnotations
};
