/**
 * Methodology View - Renders docs/methodology.md with markdown, KaTeX, and custom directives
 * Supports: :::figure:::, :::responses:::, :::dataset:::, [@citations]
 */

function parseMethodologyFrontmatter(text) {
    const match = text.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);
    if (!match) return { frontmatter: {}, content: text };

    try {
        const frontmatter = jsyaml.load(match[1]);
        return { frontmatter, content: match[2] };
    } catch (e) {
        console.error('Failed to parse frontmatter:', e);
        return { frontmatter: {}, content: text };
    }
}

async function renderMethodology() {
    const contentArea = document.getElementById('content-area');
    contentArea.innerHTML = '<div class="loading">Loading methodology...</div>';

    try {
        const response = await fetch('/docs/methodology.md');
        if (!response.ok) throw new Error('Failed to load methodology.md');

        const text = await response.text();
        const { frontmatter, content } = parseMethodologyFrontmatter(text);
        const references = frontmatter.references || {};

        // Protect math blocks from markdown parser
        let { markdown, blocks: mathBlocks } = window.protectMathBlocks(content);

        // Extract :::responses path "label"::: blocks
        const responseBlocks = [];
        markdown = markdown.replace(/:::responses\s+([^\s:]+)(?:\s+"([^"]*)")?\s*:::/g, (match, path, label) => {
            responseBlocks.push({ path, label: label || 'View responses' });
            return `RESPONSE_BLOCK_${responseBlocks.length - 1}`;
        });

        // Extract :::dataset path "label"::: blocks
        const datasetBlocks = [];
        markdown = markdown.replace(/:::dataset\s+([^\s:]+)(?:\s+"([^"]*)")?\s*:::/g, (match, path, label) => {
            datasetBlocks.push({ path, label: label || 'View examples' });
            return `DATASET_BLOCK_${datasetBlocks.length - 1}`;
        });

        // Extract :::figure path "caption" size::: blocks
        const figureBlocks = [];
        markdown = markdown.replace(/:::figure\s+([^\s:]+)(?:\s+"([^"]*)")?(?:\s+(small|medium|large))?\s*:::/g, (match, path, caption, size) => {
            figureBlocks.push({ path, caption: caption || '', size: size || '' });
            return `FIGURE_BLOCK_${figureBlocks.length - 1}`;
        });

        // Extract :::placeholder "description"::: blocks (for TODO content)
        const placeholderBlocks = [];
        markdown = markdown.replace(/:::placeholder\s+"([^"]*)"\s*:::/g, (match, description) => {
            placeholderBlocks.push({ description });
            return `PLACEHOLDER_BLOCK_${placeholderBlocks.length - 1}`;
        });

        // Extract [@key] citations
        const citedKeys = [];
        markdown = markdown.replace(/\[@(\w+)\]/g, (match, key) => {
            if (!citedKeys.includes(key)) citedKeys.push(key);
            return `CITE_${key}`;
        });

        // Fix relative image paths
        markdown = markdown.replace(/!\[([^\]]*)\]\(assets\//g, '![$1](/docs/assets/');

        // Render markdown
        marked.setOptions({ gfm: true, breaks: false, headerIds: true });
        let html = marked.parse(markdown);

        // Restore math blocks
        html = window.restoreMathBlocks(html, mathBlocks);

        // Replace response block placeholders
        responseBlocks.forEach((block, i) => {
            const dropdownId = `methodology-responses-${i}`;
            const dropdownHtml = `
                <div class="responses-dropdown" id="${dropdownId}">
                    <div class="responses-header" onclick="toggleMethodologyResponses('${dropdownId}', '${block.path}')">
                        <span class="responses-toggle">+</span>
                        <span class="responses-label">${block.label}</span>
                    </div>
                    <div class="responses-content"></div>
                </div>
            `;
            html = html.replace(`<p>RESPONSE_BLOCK_${i}</p>`, dropdownHtml);
            html = html.replace(`RESPONSE_BLOCK_${i}`, dropdownHtml);
        });

        // Replace dataset block placeholders
        datasetBlocks.forEach((block, i) => {
            const dropdownId = `methodology-dataset-${i}`;
            const dropdownHtml = `
                <div class="responses-dropdown" id="${dropdownId}">
                    <div class="responses-header" onclick="toggleMethodologyDataset('${dropdownId}', '${block.path}')">
                        <span class="responses-toggle">+</span>
                        <span class="responses-label">${block.label}</span>
                    </div>
                    <div class="responses-content"></div>
                </div>
            `;
            html = html.replace(`<p>DATASET_BLOCK_${i}</p>`, dropdownHtml);
            html = html.replace(`DATASET_BLOCK_${i}`, dropdownHtml);
        });

        // Replace figure block placeholders
        figureBlocks.forEach((block, i) => {
            const imgPath = block.path.startsWith('assets/') ? `/docs/${block.path}` : block.path;
            const sizeClass = block.size ? ` fig-${block.size}` : '';
            const figureHtml = `
                <figure class="fig${sizeClass}">
                    <img src="${imgPath}" alt="${block.caption}">
                    ${block.caption ? `<figcaption>${block.caption}</figcaption>` : ''}
                </figure>
            `;
            html = html.replace(`<p>FIGURE_BLOCK_${i}</p>`, figureHtml);
            html = html.replace(`FIGURE_BLOCK_${i}`, figureHtml);
        });

        // Replace placeholder blocks with styled TODO boxes
        placeholderBlocks.forEach((block, i) => {
            const placeholderHtml = `
                <div class="methodology-placeholder">
                    <span class="placeholder-icon">[ ]</span>
                    <span class="placeholder-text">${block.description}</span>
                </div>
            `;
            html = html.replace(`<p>PLACEHOLDER_BLOCK_${i}</p>`, placeholderHtml);
            html = html.replace(`PLACEHOLDER_BLOCK_${i}`, placeholderHtml);
        });

        // Replace citation placeholders
        for (const key of citedKeys) {
            const ref = references[key];
            if (ref) {
                const tooltipText = `${ref.title}`;
                const citeHtml = ref.url
                    ? `<a href="${ref.url}" class="citation" target="_blank" data-tooltip="${tooltipText}">(${ref.authors}, ${ref.year})</a>`
                    : `<span class="citation" data-tooltip="${tooltipText}">(${ref.authors}, ${ref.year})</span>`;
                html = html.replaceAll(`CITE_${key}`, citeHtml);
            } else {
                console.warn(`Citation [@${key}] not found in references`);
                html = html.replaceAll(`CITE_${key}`, `<span class="citation citation-missing">[${key}]</span>`);
            }
        }

        // Append References section if any citations used
        if (citedKeys.length > 0) {
            const validRefs = citedKeys.filter(key => references[key]);
            if (validRefs.length > 0) {
                let refsHtml = '<section class="references"><h2>References</h2><ol>';
                for (const key of validRefs) {
                    const ref = references[key];
                    const link = ref.url ? `<a href="${ref.url}" target="_blank">${ref.url}</a>` : '';
                    refsHtml += `<li id="ref-${key}">${ref.authors} (${ref.year}). "${ref.title}". ${link}</li>`;
                }
                refsHtml += '</ol></section>';
                html += refsHtml;
            }
        }

        contentArea.innerHTML = `<div class="prose">${html}</div>`;

        // Render math
        if (window.renderMath) {
            window.renderMath(contentArea);
        }
    } catch (error) {
        console.error('Error loading methodology:', error);
        contentArea.innerHTML = '<div class="error">Failed to load methodology</div>';
    }
}

// Dropdown toggle handlers (similar to findings.js)
async function toggleMethodologyResponses(dropdownId, path) {
    const dropdown = document.getElementById(dropdownId);
    const content = dropdown.querySelector('.responses-content');
    const toggle = dropdown.querySelector('.responses-toggle');

    if (dropdown.classList.contains('expanded')) {
        dropdown.classList.remove('expanded');
        content.style.display = 'none';
        toggle.textContent = '+';
    } else {
        if (!content.innerHTML) {
            content.innerHTML = '<div class="loading">Loading...</div>';
            try {
                const response = await fetch(path);
                if (!response.ok) throw new Error('Failed to load');
                const data = await response.json();
                content.innerHTML = renderMethodologyTable(data);
            } catch (error) {
                content.innerHTML = `<div class="error">Failed to load responses</div>`;
            }
        }
        dropdown.classList.add('expanded');
        content.style.display = 'block';
        toggle.textContent = '−';
    }
}

async function toggleMethodologyDataset(dropdownId, path) {
    const dropdown = document.getElementById(dropdownId);
    const content = dropdown.querySelector('.responses-content');
    const toggle = dropdown.querySelector('.responses-toggle');

    if (dropdown.classList.contains('expanded')) {
        dropdown.classList.remove('expanded');
        content.style.display = 'none';
        toggle.textContent = '+';
    } else {
        if (!content.innerHTML) {
            content.innerHTML = '<div class="loading">Loading...</div>';
            try {
                const response = await fetch(path);
                if (!response.ok) throw new Error('Failed to load');
                const text = await response.text();
                content.innerHTML = renderMethodologyDataset(text);
            } catch (error) {
                content.innerHTML = `<div class="error">Failed to load dataset</div>`;
            }
        }
        dropdown.classList.add('expanded');
        content.style.display = 'block';
        toggle.textContent = '−';
    }
}

function renderMethodologyTable(data) {
    if (!Array.isArray(data) || data.length === 0) {
        return '<div class="error">No data found</div>';
    }

    // Auto-detect columns from first item
    const columns = Object.keys(data[0]);

    let html = '<table class="responses-table"><thead><tr>';
    columns.forEach(col => {
        html += `<th>${col}</th>`;
    });
    html += '</tr></thead><tbody>';

    for (const row of data.slice(0, 20)) {
        html += '<tr>';
        columns.forEach(col => {
            const val = row[col];
            const display = typeof val === 'number' ? val.toFixed(2) : escapeHtml(String(val || '')).replace(/\n/g, '<br>');
            html += `<td>${display}</td>`;
        });
        html += '</tr>';
    }

    html += '</tbody></table>';
    if (data.length > 20) {
        html += `<div class="dataset-more">...and ${data.length - 20} more</div>`;
    }
    return html;
}

function renderMethodologyDataset(text) {
    const lines = text.trim().split('\n').filter(line => line.trim());
    if (lines.length === 0) {
        return '<div class="error">No examples found</div>';
    }

    const examples = lines.slice(0, 20);
    let html = '<ul class="dataset-list">';
    for (const line of examples) {
        html += `<li>${escapeHtml(line)}</li>`;
    }
    html += '</ul>';
    if (lines.length > 20) {
        html += `<div class="dataset-more">...and ${lines.length - 20} more</div>`;
    }
    return html;
}

window.renderMethodology = renderMethodology;
window.toggleMethodologyResponses = toggleMethodologyResponses;
window.toggleMethodologyDataset = toggleMethodologyDataset;
