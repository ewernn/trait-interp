/**
 * Model Analysis View - Understanding model internals and comparing variants
 *
 * Sections:
 * 1. Activation Diagnostics: Magnitude by layer, massive activations (Sun et al. 2024)
 * 2. Variant Comparison: Effect size (Cohen's d) for same prompts across model variants
 */

// Cache for loaded projections to avoid re-fetching
const projectionCache = {};

/**
 * Load all projection files for a variant/trait/prompt-set combination
 */
async function loadProjections(experiment, modelVariant, trait, promptSet) {
    const cacheKey = `${experiment}/${modelVariant}/${trait}/${promptSet}`;
    if (projectionCache[cacheKey]) {
        return projectionCache[cacheKey];
    }

    const pb = window.PathBuilder;
    const projectionsDir = pb.get('inference.projections', {
        experiment,
        model_variant: modelVariant,
        trait,
        prompt_set: promptSet
    });

    // Get list of prompt IDs from responses directory
    const responsesDir = pb.get('inference.responses', {
        experiment,
        model_variant: modelVariant,
        prompt_set: promptSet
    });

    try {
        // Fetch a sample response to get metadata
        const metadataPath = `${responsesDir}/metadata.json`;
        const metadata = await fetch(metadataPath).then(r => r.json()).catch(() => null);

        // Try to list responses
        const responsesPath = `${responsesDir}/`;
        // Since we can't list directory, we'll try loading sequentially until we fail
        const projections = {};
        let promptId = 1;
        let consecutiveFailures = 0;

        while (consecutiveFailures < 5) {
            const projPath = `${projectionsDir}/${promptId}.json`;
            try {
                const data = await fetch(projPath).then(r => r.json());
                projections[promptId] = data;
                promptId++;
                consecutiveFailures = 0;
            } catch (e) {
                consecutiveFailures++;
                promptId++;
            }
        }

        const result = { projections, metadata };
        projectionCache[cacheKey] = result;
        return result;
    } catch (error) {
        console.error(`Failed to load projections for ${cacheKey}:`, error);
        return { projections: {}, metadata: null };
    }
}

/**
 * Aggregate projection scores by layer (mean across tokens per response)
 */
function aggregateByLayer(projections, numLayers) {
    const byLayer = {};

    for (let layer = 0; layer < numLayers; layer++) {
        byLayer[layer] = [];
    }

    // For each response
    for (const [promptId, data] of Object.entries(projections)) {
        // For each layer
        for (let layer = 0; layer < numLayers; layer++) {
            const layerKey = `layer_${layer}`;
            if (data[layerKey] && Array.isArray(data[layerKey])) {
                // Mean across tokens for this response
                const scores = data[layerKey];
                const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
                byLayer[layer].push(mean);
            }
        }
    }

    return byLayer;
}

/**
 * Compute Cohen's d effect size between two distributions
 */
function computeCohenD(baseline, compare) {
    if (baseline.length === 0 || compare.length === 0) {
        return { effectSize: 0, meanDiff: 0, pValue: 1, n: 0 };
    }

    const mean1 = baseline.reduce((a, b) => a + b, 0) / baseline.length;
    const mean2 = compare.reduce((a, b) => a + b, 0) / compare.length;
    const meanDiff = mean2 - mean1;

    const variance1 = baseline.reduce((sum, x) => sum + Math.pow(x - mean1, 2), 0) / baseline.length;
    const variance2 = compare.reduce((sum, x) => sum + Math.pow(x - mean2, 2), 0) / compare.length;
    const pooledStd = Math.sqrt((variance1 + variance2) / 2);

    const effectSize = pooledStd > 0 ? meanDiff / pooledStd : 0;

    // Simple t-test (assumes equal variances)
    const n1 = baseline.length;
    const n2 = compare.length;
    const pooledVar = ((n1 - 1) * variance1 + (n2 - 1) * variance2) / (n1 + n2 - 2);
    const tStat = meanDiff / Math.sqrt(pooledVar * (1/n1 + 1/n2));

    // Approximate p-value (two-tailed)
    // For simplicity, we'll mark as significant if |t| > 2 (roughly p < 0.05 for large n)
    const pValue = Math.abs(tStat) > 2 ? 0.01 : 0.1; // Rough approximation

    return {
        effectSize,
        meanDiff,
        pValue,
        n: Math.min(n1, n2),
        baseline: { mean: mean1, std: Math.sqrt(variance1), n: n1 },
        compare: { mean: mean2, std: Math.sqrt(variance2), n: n2 }
    };
}

/**
 * Main render function
 */
async function renderModelAnalysis() {
    const contentArea = document.getElementById('content-area');

    // Guard: require experiment selection
    if (!window.state.currentExperiment) {
        contentArea.innerHTML = `
            <div class="tool-view">
                <div class="no-data">
                    <p>Please select an experiment from the sidebar</p>
                    <small>Analysis views require an experiment to be selected. Choose one from the "Experiment" section in the sidebar.</small>
                </div>
            </div>
        `;
        return;
    }

    const experiment = window.state.currentExperiment;
    const config = window.state.experimentData?.experimentConfig;

    // Check if variant comparison is available (need 2+ variants)
    const hasVariants = config?.model_variants && Object.keys(config.model_variants).length >= 2;
    const variants = hasVariants ? Object.keys(config.model_variants) : [];
    const defaultBaseline = hasVariants ? (config.defaults?.application || variants[0]) : '';
    const defaultCompare = hasVariants ? (variants.find(v => v !== defaultBaseline) || variants[1] || variants[0]) : '';

    // Render UI with both sections
    contentArea.innerHTML = `
        <div class="tool-view">
            <div class="page-intro">
                <div class="page-intro-text">Understanding model internals and comparing model variants.</div>
            </div>

            <!-- Section 1: Activation Diagnostics -->
            <section>
                ${ui.renderSubsection({
                    num: 1,
                    title: 'Activation Diagnostics',
                    infoId: 'info-activation-diagnostics',
                    infoText: 'Understanding model internals: activation magnitude growth and massive activation dimensions (Sun et al. 2024). Run <code>python analysis/massive_activations.py</code> to generate data.'
                })}

                <h4 class="subsection-header" style="margin-top: 16px;">
                    <span class="subsection-title">Activation Magnitude by Layer</span>
                    <span class="subsection-info-toggle" data-target="info-act-magnitude">►</span>
                </h4>
                <div id="info-act-magnitude" class="info-text">
                    How the residual stream grows in magnitude as each layer adds information to the hidden state.
                </div>
                <div id="activation-magnitude-plot"></div>

                <h4 class="subsection-header" style="margin-top: 24px;">
                    <span class="subsection-title">Massive Activations</span>
                    <span class="subsection-info-toggle" data-target="info-massive-acts">►</span>
                </h4>
                <div id="info-massive-acts" class="info-text">
                    Massive activation dimensions (Sun et al. 2024) - specific dimensions with values 100-1000x larger than median. These act as fixed biases.
                </div>
                <div id="massive-activations-container"></div>

                <h4 class="subsection-header" style="margin-top: 24px;">
                    <span class="subsection-title">Massive Dims Across Layers</span>
                    <span class="subsection-info-toggle" data-target="info-massive-dims-layers">►</span>
                </h4>
                <div id="info-massive-dims-layers" class="info-text">
                    Shows how each massive dimension's magnitude changes across layers (normalized by layer average).
                </div>
                <div class="projection-toggle" style="margin-bottom: 12px;">
                    <span class="projection-toggle-label">Criteria:</span>
                    <select id="massive-dims-criteria">
                        <option value="top5-3layers">Top 5, 3+ layers</option>
                        <option value="top3-any">Top 3, any layer</option>
                        <option value="top5-any">Top 5, any layer</option>
                    </select>
                </div>
                <div id="massive-dims-layers-plot"></div>
            </section>

            <!-- Section 2: Variant Comparison -->
            <section>
                ${ui.renderSubsection({
                    num: 2,
                    title: 'Variant Comparison',
                    infoId: 'info-variant-comparison',
                    infoText: 'Effect size (Cohen\\'s d) between model variants on trait projections. Positive = variant B projects higher. Run <code>python analysis/model_diff/compare_variants.py</code> to generate data.'
                })}

                <div id="model-diff-container" style="margin-top: 16px;">
                    <div class="loading">Loading model diff data...</div>
                </div>
            </section>
        </div>
    `;

    // Setup info toggles
    window.setupSubsectionInfoToggles?.();

    // Render activation diagnostics (always)
    await renderActivationMagnitudePlot();
    await renderMassiveActivations();
    await renderMassiveDimsAcrossLayers();

    // Render model diff comparison
    await renderModelDiffComparison(experiment);
}

/**
 * Render model diff comparison using pre-computed results from compare_variants.py
 */
async function renderModelDiffComparison(experiment) {
    const container = document.getElementById('model-diff-container');
    if (!container) return;

    try {
        // Fetch available comparisons
        const response = await fetch(`/api/experiments/${experiment}/model-diff`);
        const data = await response.json();
        const comparisons = data.comparisons || [];

        if (comparisons.length === 0) {
            container.innerHTML = `
                <div class="info">
                    No model diff data available.
                    <br><br>
                    Run: <code>python analysis/model_diff/compare_variants.py --experiment ${experiment} --variant-a instruct --variant-b rm_lora --prompt-set {prompt_set}</code>
                </div>
            `;
            return;
        }

        // For now, use the first comparison (typically only one)
        const comparison = comparisons[0];
        const { variant_a, variant_b, prompt_sets } = comparison;

        // Load results for all prompt sets
        const allResults = {};
        for (const promptSet of prompt_sets) {
            const resultsPath = `experiments/${experiment}/model_diff/${comparison.variant_pair}/${promptSet}/results.json`;
            try {
                const res = await fetch('/' + resultsPath);
                if (res.ok) {
                    allResults[promptSet] = await res.json();
                }
            } catch (e) {
                console.warn(`Failed to load ${resultsPath}:`, e);
            }
        }

        if (Object.keys(allResults).length === 0) {
            container.innerHTML = `<div class="info">Failed to load model diff results.</div>`;
            return;
        }

        // Build summary table
        const summaryRows = [];
        const allTraits = new Set();
        for (const [promptSet, results] of Object.entries(allResults)) {
            for (const trait of Object.keys(results.traits || {})) {
                allTraits.add(trait);
            }
        }

        for (const trait of allTraits) {
            const row = { trait: trait.split('/').pop() };
            for (const [promptSet, results] of Object.entries(allResults)) {
                const traitData = results.traits?.[trait];
                if (traitData) {
                    const setName = promptSet.split('/').pop();
                    row[setName] = {
                        peak_layer: traitData.peak_layer,
                        peak_effect: traitData.peak_effect_size
                    };
                }
            }
            summaryRows.push(row);
        }

        // Render summary
        const promptSetNames = prompt_sets.map(ps => ps.split('/').pop());
        container.innerHTML = `
            <div class="model-diff-header">
                <strong>${variant_b}</strong> vs <strong>${variant_a}</strong>
                <span style="color: var(--text-tertiary); margin-left: 8px;">(${Object.values(allResults)[0]?.n_prompts || '?'} prompts)</span>
            </div>

            <table class="data-table" style="margin: 16px 0;">
                <thead>
                    <tr>
                        <th>Trait</th>
                        ${promptSetNames.map(ps => `<th>${ps}</th>`).join('')}
                    </tr>
                </thead>
                <tbody>
                    ${summaryRows.map(row => `
                        <tr>
                            <td>${row.trait}</td>
                            ${promptSetNames.map(ps => {
                                const data = row[ps];
                                if (data) {
                                    const color = data.peak_effect > 1.5 ? 'var(--success-color)' :
                                                  data.peak_effect > 0.5 ? 'var(--warning-color)' :
                                                  'var(--text-secondary)';
                                    return `<td style="color: ${color};">${data.peak_effect.toFixed(2)}σ @ L${data.peak_layer}</td>`;
                                }
                                return '<td>—</td>';
                            }).join('')}
                        </tr>
                    `).join('')}
                </tbody>
            </table>

            <div id="model-diff-chart"></div>
        `;

        // Plot all traits × prompt sets
        renderModelDiffChart(allResults, comparison);

    } catch (error) {
        console.error('Model diff error:', error);
        container.innerHTML = `<div class="info">Error loading model diff data: ${error.message}</div>`;
    }
}

/**
 * Render model diff chart with all traits and prompt sets
 */
function renderModelDiffChart(allResults, comparison) {
    const chartDiv = document.getElementById('model-diff-chart');
    if (!chartDiv) return;

    const colors = window.getChartColors?.() || ['#4ecdc4', '#ff6b6b', '#ffe66d', '#95e1d3'];
    const traces = [];
    let colorIdx = 0;

    // Collect all traits
    const allTraits = new Set();
    for (const results of Object.values(allResults)) {
        for (const trait of Object.keys(results.traits || {})) {
            allTraits.add(trait);
        }
    }

    // Create traces for each trait × prompt set combination
    for (const trait of allTraits) {
        const traitName = trait.split('/').pop();
        const color = colors[colorIdx % colors.length];
        let dashIdx = 0;

        for (const [promptSet, results] of Object.entries(allResults)) {
            const traitData = results.traits?.[trait];
            if (!traitData || !traitData.per_layer_effect_size) continue;

            const setName = promptSet.split('/').pop();
            const dash = dashIdx === 0 ? 'solid' : 'dash';

            traces.push({
                x: traitData.layers,
                y: traitData.per_layer_effect_size,
                type: 'scatter',
                mode: 'lines+markers',
                name: `${traitName} ${setName} (peak: ${traitData.peak_effect_size.toFixed(2)}σ @ L${traitData.peak_layer})`,
                line: { color, width: 2, dash },
                marker: { size: 3 },
                hovertemplate: `${traitName} ${setName}<br>L%{x}: %{y:.2f}σ<extra></extra>`
            });

            dashIdx++;
        }
        colorIdx++;
    }

    const layout = {
        title: `Trait Detection: ${comparison.variant_b} vs ${comparison.variant_a}`,
        xaxis: {
            title: 'Layer',
            dtick: 10,
            showgrid: true,
            gridcolor: 'rgba(128,128,128,0.2)'
        },
        yaxis: {
            title: 'Effect Size (σ)',
            zeroline: true,
            zerolinewidth: 1,
            zerolinecolor: 'rgba(128,128,128,0.5)',
            showgrid: true,
            gridcolor: 'rgba(128,128,128,0.2)'
        },
        hovermode: 'closest',
        showlegend: true,
        legend: {
            x: 1,
            y: 1,
            xanchor: 'right',
            bgcolor: 'rgba(0,0,0,0.5)'
        },
        template: 'plotly_dark',
        plot_bgcolor: 'var(--bg-primary)',
        paper_bgcolor: 'var(--bg-primary)',
        font: { color: 'var(--text-primary)' },
        height: 400
    };

    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false
    };

    Plotly.newPlot(chartDiv, traces, layout, config);
}


// ============================================================================
// Activation Diagnostics Functions
// ============================================================================

/**
 * Fetch massive activations data, using calibration.json as canonical source.
 * Calibration contains model-wide massive dims computed from neutral prompts.
 */
async function fetchMassiveActivationsData() {
    const modelVariant = window.state.experimentData?.experimentConfig?.defaults?.application || 'instruct';
    const calibrationPath = window.paths.get('inference.massive_activations', { prompt_set: 'calibration', model_variant: modelVariant });
    const response = await fetch('/' + calibrationPath);
    if (!response.ok) return null;
    return response.json();
}


/**
 * Render Activation Magnitude plot showing ||h|| by layer.
 * Uses data from massive activations calibration file.
 */
async function renderActivationMagnitudePlot() {
    const plotDiv = document.getElementById('activation-magnitude-plot');
    if (!plotDiv) return;

    try {
        const data = await fetchMassiveActivationsData();
        if (!data || !data.aggregate?.layer_norms) {
            plotDiv.innerHTML = `
                <div class="info">
                    Activation magnitude data not available.
                    <br><br>
                    Run: <code>python analysis/massive_activations.py --experiment ${window.paths.getExperiment()}</code>
                </div>
            `;
            return;
        }

        const layerNorms = data.aggregate.layer_norms;
        const layers = Object.keys(layerNorms).map(Number).sort((a, b) => a - b);
        const norms = layers.map(l => layerNorms[l]);

        // Show model info if available
        const modelInfo = data.model ? `<div class="model-label">Model: <code>${data.model}</code></div>` : '';
        plotDiv.innerHTML = modelInfo;
        const chartDiv = document.createElement('div');
        plotDiv.appendChild(chartDiv);

        const traces = [{
            x: layers,
            y: norms,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Mean ||h||',
            line: { color: window.getChartColors()[0], width: 2 },
            marker: { size: 4 },
            hovertemplate: '<b>Layer %{x}</b><br>||h|| = %{y:.1f}<extra></extra>'
        }];

        const layout = window.buildChartLayout({
            preset: 'layerChart',
            traces,
            height: 250,
            legendPosition: 'none',
            xaxis: { title: 'Layer', tickmode: 'linear', tick0: 0, dtick: 5, showgrid: true },
            yaxis: { title: '||h|| (L2 norm)', showgrid: true }
        });
        window.renderChart(chartDiv, traces, layout);

    } catch (error) {
        plotDiv.innerHTML = `<div class="info">Error loading activation data: ${error.message}</div>`;
    }
}


/**
 * Render Massive Activations section.
 * Shows mean alignment plot - how much tokens point in a common direction.
 */
async function renderMassiveActivations() {
    const container = document.getElementById('massive-activations-container');
    if (!container) return;

    try {
        const data = await fetchMassiveActivationsData();
        if (!data) {
            container.innerHTML = `
                <div class="info">
                    No massive activation calibration data.
                    <br><br>
                    Run: <code>python analysis/massive_activations.py --experiment ${window.paths.getExperiment()}</code>
                </div>
            `;
            return;
        }

        const aggregate = data.aggregate || {};
        const meanAlignment = aggregate.mean_alignment_by_layer || {};

        if (Object.keys(meanAlignment).length === 0) {
            container.innerHTML = `<div class="info">No mean alignment data available.</div>`;
            return;
        }

        container.innerHTML = `<div id="mean-alignment-plot"></div>`;

        // Plot mean alignment by layer
        const layers = Object.keys(meanAlignment).map(Number).sort((a, b) => a - b);
        const alignments = layers.map(l => meanAlignment[l]);

        const alignTrace = {
            x: layers,
            y: alignments.map(v => v * 100),
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Mean Alignment',
            line: { color: window.getChartColors()[0], width: 2 },
            marker: { size: 4 },
            hovertemplate: 'L%{x}<br>Alignment: %{y:.1f}%<extra></extra>'
        };

        const alignLayout = window.buildChartLayout({
            preset: 'layerChart',
            traces: [alignTrace],
            height: 200,
            legendPosition: 'none',
            xaxis: { title: 'Layer', dtick: 5, showgrid: true },
            yaxis: { title: 'Mean Alignment (%)', range: [0, 100], showgrid: true }
        });
        window.renderChart('mean-alignment-plot', [alignTrace], alignLayout);

    } catch (error) {
        container.innerHTML = `<div class="info">Error loading massive activation data: ${error.message}</div>`;
    }
}


/**
 * Render Massive Dims Across Layers section.
 * Shows how each massive dim's normalized magnitude changes across layers.
 */
async function renderMassiveDimsAcrossLayers() {
    const container = document.getElementById('massive-dims-layers-plot');
    if (!container) return;

    try {
        const data = await fetchMassiveActivationsData();
        if (!data) {
            container.innerHTML = `<div class="info">No massive activation data. Run <code>python analysis/massive_activations.py --experiment ${window.paths.getExperiment()}</code></div>`;
            return;
        }
        const aggregate = data.aggregate || {};
        const topDimsByLayer = aggregate.top_dims_by_layer || {};
        const dimMagnitude = aggregate.dim_magnitude_by_layer || {};

        if (Object.keys(dimMagnitude).length === 0) {
            container.innerHTML = `<div class="info">No per-layer magnitude data. Re-run <code>python analysis/massive_activations.py</code> to generate.</div>`;
            return;
        }

        // Get criteria from dropdown
        const criteriaSelect = document.getElementById('massive-dims-criteria');
        const criteria = criteriaSelect?.value || 'top5-3layers';

        // Filter dims based on criteria
        const filteredDims = filterDimsByCriteria(topDimsByLayer, criteria);

        if (filteredDims.length === 0) {
            container.innerHTML = `<div class="info">No dims match criteria "${criteria}".</div>`;
            return;
        }

        // Show model info if available
        const modelInfo = data.model ? `<div class="model-label">Model: <code>${data.model}</code></div>` : '';
        container.innerHTML = modelInfo;
        const chartDiv = document.createElement('div');
        container.appendChild(chartDiv);

        // Build traces
        const colors = window.getChartColors();
        const nLayers = Object.keys(topDimsByLayer).length;
        const layers = Array.from({ length: nLayers }, (_, i) => i);

        const traces = filteredDims.map((dim, idx) => {
            const magnitudes = dimMagnitude[dim] || [];
            return {
                x: layers,
                y: magnitudes,
                type: 'scatter',
                mode: 'lines+markers',
                name: `dim ${dim}`,
                line: { color: colors[idx % colors.length], width: 2 },
                marker: { size: 4 },
                hovertemplate: `dim ${dim}<br>L%{x}<br>Normalized: %{y:.2f}x<extra></extra>`
            };
        });

        const layout = window.buildChartLayout({
            preset: 'layerChart',
            traces,
            height: 300,
            legendPosition: 'above',
            xaxis: { title: 'Layer', dtick: 5, showgrid: true },
            yaxis: { title: 'Normalized Magnitude', showgrid: true }
        });
        window.renderChart(chartDiv, traces, layout);

        // Setup dropdown change handler
        if (criteriaSelect && !criteriaSelect.dataset.bound) {
            criteriaSelect.dataset.bound = 'true';
            criteriaSelect.addEventListener('change', () => {
                renderMassiveDimsAcrossLayers();
            });
        }

    } catch (error) {
        container.innerHTML = `<div class="info">Error loading data: ${error.message}</div>`;
    }
}


/**
 * Filter dims based on selected criteria.
 */
function filterDimsByCriteria(topDimsByLayer, criteria) {
    const dimAppearances = {};  // {dim: count of layers it appears in}

    // Count appearances based on criteria
    for (const [layer, dims] of Object.entries(topDimsByLayer)) {
        const topK = criteria === 'top3-any' ? 3 : 5;
        const dimsToCount = dims.slice(0, topK);
        for (const dim of dimsToCount) {
            dimAppearances[dim] = (dimAppearances[dim] || 0) + 1;
        }
    }

    // Filter based on min layers
    const minLayers = criteria === 'top5-3layers' ? 3 : 1;
    const filtered = Object.entries(dimAppearances)
        .filter(([dim, count]) => count >= minLayers)
        .map(([dim]) => parseInt(dim))
        .sort((a, b) => a - b);

    return filtered;
}


// Export (both old and new names for compatibility)
window.renderModelAnalysis = renderModelAnalysis;
window.renderModelComparison = renderModelAnalysis;  // Backwards compatibility
