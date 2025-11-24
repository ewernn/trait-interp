// Trait Dashboard View - Compact 4-per-row layout with mini heatmaps and quality summaries

let evaluationData = null;

async function loadEvaluationData() {
    if (evaluationData) return evaluationData;
    try {
        const url = window.paths.extractionEvaluation();
        const response = await fetch(url);
        if (!response.ok) {
            evaluationData = { all_results: [] };
            return evaluationData;
        }
        evaluationData = await response.json();
        return evaluationData;
    } catch (error) {
        console.error('Error loading evaluation data:', error);
        evaluationData = { all_results: [] };
        return evaluationData;
    }
}


async function renderTraitDashboard() {
    const contentArea = document.getElementById('content-area');
    const filteredTraits = window.getFilteredTraits();

    if (filteredTraits.length === 0) {
        contentArea.innerHTML = `<div class="info" style="margin:16px;padding:8px;">Select at least one trait to view.</div>`;
        return;
    }

    contentArea.innerHTML = '<div class="loading">Loading...</div>';
    window.paths.setExperiment(window.state.experimentData.name);

    const allEvaluationData = await loadEvaluationData();

    // Create compact grid layout - 4 per row
    contentArea.innerHTML = `<div class="trait-dashboard-grid"></div>`;
    const grid = contentArea.querySelector('.trait-dashboard-grid');

    for (const trait of filteredTraits) {
        const card = document.createElement('div');
        card.className = 'trait-compact-card';
        card.id = `dashboard-${trait.name}`;
        grid.appendChild(card);
        renderCompactTraitCard(trait, allEvaluationData, card);
    }
}


async function renderCompactTraitCard(trait, allEvaluationData, container) {
    const displayName = window.getDisplayName(trait.name);
    container.innerHTML = `<div class="trait-compact-name">${displayName}</div><div class="trait-compact-loading">...</div>`;

    try {
        const nLayers = trait.metadata?.n_layers || 26;
        const methods = ['mean_diff', 'probe', 'ica', 'gradient'];
        const layers = Array.from({ length: nLayers }, (_, i) => i);

        // Fetch vector metadata
        const fetchPromises = methods.flatMap(method =>
            layers.map(layer => {
                const url = window.paths.vectorMetadata(trait, method, layer);
                return fetch(url)
                    .then(r => r.ok ? r.json() : null)
                    .catch(() => null);
            })
        );
        const statsResults = (await Promise.all(fetchPromises)).filter(r => r !== null);
        const qualityResults = allEvaluationData.all_results.filter(r => r.trait === trait.name);

        // Build compact card
        const heatmapId = `heatmap-${trait.name.replace(/\//g, '-')}`;
        container.innerHTML = `
            <div class="trait-compact-name">${displayName}</div>
            <div class="trait-compact-heatmap" id="${heatmapId}"></div>
            <div class="trait-compact-quality" id="quality-${trait.name.replace(/\//g, '-')}"></div>
        `;

        // Render mini heatmap
        if (statsResults.length > 0) {
            renderMiniHeatmap(trait, statsResults, heatmapId, nLayers);
        } else {
            document.getElementById(heatmapId).innerHTML = `<span class="trait-compact-no-data">No vectors</span>`;
        }

        // Render compact quality
        if (qualityResults.length > 0) {
            renderCompactQuality(qualityResults, `quality-${trait.name.replace(/\//g, '-')}`);
        }

    } catch (error) {
        console.error(`Failed to render ${trait.name}:`, error);
        container.innerHTML = `<div class="trait-compact-name">${displayName}</div><span class="trait-compact-no-data">Error</span>`;
    }
}


function renderMiniHeatmap(trait, statsResults, containerId, nLayers) {
    const methods = ['mean_diff', 'probe', 'ica', 'gradient'];
    const layers = Array.from({ length: nLayers }, (_, i) => i);

    const vectorData = {};
    methods.forEach(m => vectorData[m] = {});
    statsResults.forEach(r => {
        if (r && r.method && r.layer !== undefined) {
            vectorData[r.method][r.layer] = r;
        }
    });

    const normalizedData = layers.map(layer => {
        return methods.map(method => {
            const metadata = vectorData[method] ? vectorData[method][layer] : null;
            if (!metadata) return null;
            if (method === 'probe') return metadata.vector_norm ? (1.0 / metadata.vector_norm) : null;
            if (method === 'gradient') return metadata.final_separation || null;
            return metadata.vector_norm;
        });
    });

    const maxPerMethod = methods.map((_, methodIdx) => {
        const values = normalizedData.map(row => row[methodIdx]).filter(v => v !== null && !isNaN(v));
        return values.length > 0 ? Math.max(...values) : 1;
    });

    const heatmapData = normalizedData.map(row =>
        row.map((value, methodIdx) =>
            value === null ? null : (value / maxPerMethod[methodIdx]) * 100
        )
    );

    const trace = {
        z: heatmapData,
        x: methods.map(m => m === 'mean_diff' ? 'md' : m === 'gradient' ? 'grad' : m),
        y: layers,
        type: 'heatmap',
        colorscale: window.ASYMB_COLORSCALE,
        hovertemplate: '%{x} L%{y}: %{z:.0f}%<extra></extra>',
        zmin: 0,
        zmax: 100,
        showscale: false
    };

    // Get computed text color for axis labels
    const textColor = getComputedStyle(document.documentElement).getPropertyValue('--text-primary').trim();

    Plotly.newPlot(containerId, [trace], {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        margin: { l: 22, r: 4, t: 4, b: 28 },
        xaxis: {
            tickfont: { size: 10, color: textColor },
            tickangle: 0,
            fixedrange: true
        },
        yaxis: {
            tickfont: { size: 9, color: textColor },
            tickvals: [0, 12, 25],
            fixedrange: true
        },
        height: 160,
        width: null
    }, { displayModeBar: false, responsive: true });
}


function renderCompactQuality(qualityResults, containerId) {
    const maxEffect = Math.max(...qualityResults.map(r => r.val_effect_size || 0));
    const max_d = maxEffect > 0 ? maxEffect : 1;

    const calculateScore = (r) => {
        if (r.val_accuracy === null || r.val_effect_size === null) return 0;
        return (0.5 * r.val_accuracy) + (0.5 * (r.val_effect_size || 0) / max_d);
    };

    const sorted = [...qualityResults].sort((a, b) => calculateScore(b) - calculateScore(a));
    const top3 = sorted.slice(0, 3);

    if (top3.length === 0) {
        document.getElementById(containerId).innerHTML = '';
        return;
    }

    const rows = top3.map(r => {
        const acc = (r.val_accuracy * 100).toFixed(0);
        const accClass = r.val_accuracy >= 0.9 ? 'quality-good' : r.val_accuracy >= 0.75 ? 'quality-ok' : 'quality-bad';
        const method = r.method === 'mean_diff' ? 'md' : r.method === 'gradient' ? 'grad' : r.method;
        return `<div class="quality-row"><span class="quality-method">${method}</span><span class="quality-layer">L${r.layer}</span><span class="quality-acc ${accClass}">${acc}%</span><span class="quality-effect">${r.val_effect_size?.toFixed(1) || '-'}</span></div>`;
    }).join('');

    document.getElementById(containerId).innerHTML = `<div class="quality-header"><span>Method</span><span>Lyr</span><span>Acc</span><span>d</span></div>${rows}`;
}

window.renderTraitDashboard = renderTraitDashboard;
