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
        // Support both "idx" (numeric) and "id" (string) annotation formats
        const entryId = entry.idx ?? entry.id;
        // eslint-disable-next-line eqeqeq
        if (entryId == responseIdx) {
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

/**
 * Convert a text span to a token index range [startTokenIdx, endTokenIdx).
 * Walks response tokens cumulatively tracking char positions to find overlap.
 *
 * @param {string[]} responseTokens - Array of token strings for the response
 * @param {string} responseText - Full decoded response text
 * @param {string} spanText - Span text to locate
 * @returns {[number, number]|null} [startTokenIdx, endTokenIdx) or null if not found
 */
function spanToTokenRange(responseTokens, responseText, spanText) {
    if (!responseTokens || !responseText || !spanText) return null;

    // Find char range of span in response text
    let charStart = responseText.indexOf(spanText);
    if (charStart === -1) {
        // Try case-insensitive
        charStart = responseText.toLowerCase().indexOf(spanText.toLowerCase());
        if (charStart === -1) return null;
    }
    const charEnd = charStart + spanText.length;

    // Map responseText positions to token-walk positions.
    // Individually-decoded tokens may have extra leading spaces that produce
    // double spaces when concatenated (e.g., ", " + " which" → ",  which"
    // vs ", which" in responseText). Align both strings to correct for drift.
    const joinedTokens = responseTokens.join('');
    let ri = 0, ji = 0;
    let mappedStart = null, mappedEnd = null;

    while (ri <= charEnd && ji <= joinedTokens.length) {
        if (ri === charStart && mappedStart === null) mappedStart = ji;
        if (ri === charEnd) { mappedEnd = ji; break; }

        if (ri < responseText.length && ji < joinedTokens.length &&
            responseText[ri] === joinedTokens[ji]) {
            ri++;
            ji++;
        } else if (ji < joinedTokens.length) {
            ji++; // extra char in joined tokens (double space from decode)
        } else {
            break;
        }
    }

    if (mappedStart === null || mappedEnd === null) return null;

    // Walk tokens using token-walk coordinates (consistent with token lengths)
    let pos = 0;
    let startToken = null;
    let endToken = null;

    for (let i = 0; i < responseTokens.length; i++) {
        const tokenLen = responseTokens[i].length;
        const tokenStart = pos;
        const tokenEnd = pos + tokenLen;

        if (tokenEnd > mappedStart && tokenStart < mappedEnd) {
            if (startToken === null) startToken = i;
            endToken = i + 1;
        }

        pos = tokenEnd;
        if (tokenStart >= mappedEnd) break;
    }

    if (startToken === null) return null;
    return [startToken, endToken];
}

/**
 * Fetch annotations for a prompt set, with caching.
 * Cache stored on window.state._annotationCache to avoid re-fetching within same set.
 *
 * @param {string} experiment - Experiment name
 * @param {string} modelVariant - Model variant (e.g., 'instruct')
 * @param {string} promptSet - Prompt set name
 * @returns {Promise<Object|null>} Annotation data or null
 */
async function fetchAnnotations(experiment, modelVariant, promptSet) {
    const cacheKey = `${experiment}/${promptSet}`;

    // Check cache (keyed by experiment+promptSet, variant-agnostic)
    if (window.state._annotationCache?.key === cacheKey) {
        return window.state._annotationCache.data;
    }

    // Try the specified variant first, then all other variants that have data
    const variants = [modelVariant, ...(window.state.variantsPerPromptSet?.[promptSet] || []).filter(v => v !== modelVariant)];

    for (const variant of variants) {
        const url = `/experiments/${experiment}/inference/${variant}/responses/${promptSet}_annotations.json`;
        try {
            const response = await fetch(url);
            if (!response.ok) continue;
            const data = await response.json();
            window.state._annotationCache = { key: cacheKey, data };
            return data;
        } catch (e) {
            continue;
        }
    }

    // Cache the miss to avoid re-fetching
    window.state._annotationCache = { key: cacheKey, data: null };
    return null;
}

/**
 * Get annotation token ranges for a specific prompt.
 * Combines fetching, span lookup, and token range conversion.
 *
 * @param {string} experiment - Experiment name
 * @param {string} modelVariant - Model variant
 * @param {string} promptSet - Prompt set name
 * @param {number} promptId - Prompt ID
 * @param {string[]} responseTokens - Response token strings
 * @param {string} responseText - Full response text
 * @returns {Promise<Array<[number, number]>>} Array of [startTokenIdx, endTokenIdx) ranges
 */
async function getAnnotationTokenRanges(experiment, modelVariant, promptSet, promptId, responseTokens, responseText) {
    const annotationData = await fetchAnnotations(experiment, modelVariant, promptSet);
    if (!annotationData) return [];

    const spans = getSpansForResponse(annotationData, promptId);
    if (spans.length === 0) return [];

    const ranges = [];
    for (const { span } of spans) {
        const range = spanToTokenRange(responseTokens, responseText, span);
        if (range) ranges.push(range);
    }
    return ranges;
}

// Export
window.annotations = {
    spansToCharRanges,
    getSpansForResponse,
    mergeRanges,
    applyHighlights,
    loadAndConvertAnnotations,
    spanToTokenRange,
    fetchAnnotations,
    getAnnotationTokenRanges
};
