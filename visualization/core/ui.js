/**
 * Shared UI primitives for visualization views.
 * Pure functions that return HTML strings.
 *
 * Usage: ui.renderToggle({ ... }), ui.setLoading('container')
 */

// === Subsections ===

/**
 * Render a subsection header with expandable info panel.
 * @param {Object} options
 * @param {string|number} [options.num] - Section number (optional)
 * @param {string} options.title - Section title
 * @param {string} options.infoId - ID for the info panel
 * @param {string} options.infoText - Content for the info panel
 * @param {string} [options.level='h3'] - Header level (h2, h3, h4)
 * @returns {string} HTML string
 */
function renderSubsection({ num, title, infoId, infoText, level = 'h3' }) {
    const numHtml = num ? `<span class="subsection-num">${num}.</span>` : '';
    return `
        <${level} class="subsection-header">
            ${numHtml}
            <span class="subsection-title">${title}</span>
            <span class="subsection-info-toggle" data-target="${infoId}">►</span>
        </${level}>
        <div class="subsection-info" id="${infoId}">${infoText}</div>
    `;
}

// === Form Controls ===

/**
 * Render a select dropdown with label.
 * @param {Object} options
 * @param {string} options.id - Element ID
 * @param {string} options.label - Label text
 * @param {Array<string|{value: string, label?: string}>} options.options - Options array
 * @param {string} [options.selected] - Currently selected value
 * @param {string} [options.className] - Additional CSS class
 * @param {string} [options.placeholder] - Placeholder option text
 * @returns {string} HTML string
 */
function renderSelect({ id, label, options, selected, className, placeholder }) {
    const optionsHtml = options.map(opt => {
        const value = typeof opt === 'string' ? opt : opt.value;
        const text = typeof opt === 'string' ? opt : (opt.label || opt.value);
        return `<option value="${value}" ${value === selected ? 'selected' : ''}>${text}</option>`;
    }).join('');
    const placeholderHtml = placeholder ? `<option value="">${placeholder}</option>` : '';

    return `
        <div class="control-group">
            <label>${label}</label>
            <select id="${id}" class="${className || id}">${placeholderHtml}${optionsHtml}</select>
        </div>
    `;
}

/**
 * Render a checkbox toggle with label.
 * @param {Object} options
 * @param {string} [options.id] - Element ID (optional)
 * @param {string} options.label - Label text
 * @param {boolean} [options.checked=false] - Whether checked
 * @param {{key: string, value: string}} [options.dataAttr] - Data attribute
 * @param {string} [options.className] - Additional CSS class
 * @returns {string} HTML string
 */
function renderToggle({ id, label, checked, dataAttr, className }) {
    const idAttr = id ? `id="${id}"` : '';
    const checkedAttr = checked ? 'checked' : '';
    const dataHtml = dataAttr ? `data-${dataAttr.key}="${dataAttr.value}"` : '';
    return `
        <label class="toggle-row ${className || ''}">
            <input type="checkbox" ${idAttr} ${checkedAttr} ${dataHtml}>
            ${label}
        </label>
    `;
}

// === Chips ===

/**
 * Render a single chip/button.
 * @param {Object} options
 * @param {string} options.label - Button text
 * @param {boolean} [options.active=false] - Whether active/selected
 * @param {{key: string, value: string}} [options.dataAttr] - Data attribute
 * @param {string} [options.className] - Additional CSS class
 * @param {string} [options.onClick] - Inline onclick handler
 * @returns {string} HTML string
 */
function renderChip({ label, active, dataAttr, className, onClick }) {
    const activeClass = active ? 'active' : '';
    const dataHtml = dataAttr ? `data-${dataAttr.key}="${dataAttr.value}"` : '';
    const onClickHtml = onClick ? `onclick="${onClick}"` : '';
    return `<button class="btn btn-xs ${className || ''} ${activeClass}" ${dataHtml} ${onClickHtml}>${label}</button>`;
}

/**
 * Render a group of chip buttons.
 * @param {Object} options
 * @param {Array<string|{value: string, label?: string}>} options.items - Items to render
 * @param {string|Set<string>} [options.selected] - Selected value(s)
 * @param {string} options.dataKey - Data attribute key for each chip
 * @param {string} [options.className] - Additional CSS class for each chip
 * @param {boolean} [options.multi=false] - Allow multiple selection
 * @returns {string} HTML string
 */
function renderChipGroup({ items, selected, dataKey, className, multi = false }) {
    return items.map(item => {
        const value = typeof item === 'string' ? item : item.value;
        const label = typeof item === 'string' ? item : (item.label || item.value);
        const active = multi ? selected?.has(value) : selected === value;
        return renderChip({ label, active, dataAttr: { key: dataKey, value }, className });
    }).join('');
}

// === Tables ===

/**
 * Render a sortable table header cell.
 * @param {Object} options
 * @param {string} options.key - Sort key for this column
 * @param {string} options.label - Column header text
 * @param {string} [options.sortKey] - Currently active sort key
 * @param {'asc'|'desc'} [options.sortDir] - Current sort direction
 * @returns {string} HTML string
 */
function renderSortableHeader({ key, label, sortKey, sortDir }) {
    const isActive = key === sortKey;
    const indicator = isActive ? (sortDir === 'desc' ? '▼' : '▲') : '▼';
    return `
        <th class="sortable ${isActive ? 'sort-active' : ''}" data-sort="${key}">
            ${label} <span class="sort-indicator">${indicator}</span>
        </th>
    `;
}

/**
 * Sort an array of objects by a key.
 * @param {Array<Object>} data - Data to sort
 * @param {string} key - Property key to sort by
 * @param {'asc'|'desc'} dir - Sort direction
 * @returns {Array<Object>} Sorted copy of data
 */
function sortData(data, key, dir) {
    return [...data].sort((a, b) => {
        const aVal = a[key] ?? 0;
        const bVal = b[key] ?? 0;
        if (typeof aVal === 'string') {
            return dir === 'desc' ? bVal.localeCompare(aVal) : aVal.localeCompare(bVal);
        }
        return dir === 'desc' ? bVal - aVal : aVal - bVal;
    });
}

/**
 * Handle click on sortable table header.
 * Updates state.sortKey and state.sortDir, then calls rerender.
 * @param {Event} e - Click event
 * @param {{sortKey?: string, sortDir?: string}} state - State object to update
 * @param {Function} rerender - Callback to re-render the table
 */
function handleSortClick(e, state, rerender) {
    const th = e.target.closest('.sortable');
    if (!th) return;
    const key = th.dataset.sort;
    state.sortDir = (state.sortKey === key && state.sortDir === 'desc') ? 'asc' : 'desc';
    state.sortKey = key;
    rerender();
}

// === States ===

/**
 * Render a loading indicator.
 * @param {string} [message='Loading...'] - Loading message
 * @returns {string} HTML string
 */
function renderLoading(message = 'Loading...') {
    return `<div class="loading">${message}</div>`;
}

/**
 * Set loading state on an element.
 * @param {string|HTMLElement} elementOrId - Element or ID
 * @param {string} [message='Loading...'] - Loading message
 */
function setLoading(elementOrId, message = 'Loading...') {
    const el = typeof elementOrId === 'string' ? document.getElementById(elementOrId) : elementOrId;
    if (el) el.innerHTML = renderLoading(message);
}

/**
 * Render an error message.
 * @param {string} message - Error message
 * @param {string} [details] - Additional details (will be escaped)
 * @returns {string} HTML string
 */
function renderError(message, details) {
    const escape = window.escapeHtml || (s => s);
    const detailsHtml = details ? `<div class="error-details">${escape(details)}</div>` : '';
    return `<div class="error">${escape(message)}${detailsHtml}</div>`;
}

// === Export ===

window.ui = {
    renderSubsection,
    renderSelect,
    renderToggle,
    renderChip,
    renderChipGroup,
    renderSortableHeader,
    sortData,
    handleSortClick,
    renderLoading,
    setLoading,
    renderError
};
