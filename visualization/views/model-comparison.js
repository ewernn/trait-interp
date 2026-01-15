/**
 * Model Comparison View - Compare trait activations across model variants
 *
 * Computes effect size (Cohen's d) for same prompts processed by different models.
 * Used for analyzing fine-tuning effects, LoRA impacts, etc.
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
async function renderModelComparison() {
    const contentArea = document.getElementById('content-area');

    // Guard: require experiment selection
    if (!window.state.currentExperiment) {
        contentArea.innerHTML = `
            <div class="tool-view">
                <div class="no-data">
                    <p>Please select an experiment from the sidebar</p>
                </div>
            </div>
        `;
        return;
    }

    const experiment = window.state.currentExperiment;
    const config = window.state.experimentData?.config;

    if (!config || !config.model_variants) {
        contentArea.innerHTML = `
            <div class="tool-view">
                <div class="error">No model variants defined in experiment config</div>
            </div>
        `;
        return;
    }

    // Get available variants
    const variants = Object.keys(config.model_variants);
    const defaultBaseline = config.defaults?.application || variants[0];
    const defaultCompare = variants.find(v => v !== defaultBaseline) || variants[1] || variants[0];

    // Render UI
    contentArea.innerHTML = `
        <div class="tool-view">
            <div class="tool-header">
                <h2>Model Comparison</h2>
                <p class="tool-description">
                    Compare how different model variants activate on the same trait when processing identical text.
                    Uses prefilled responses (same prompts → variant A generates → both variants process same text).
                </p>
            </div>

            <div class="tool-controls">
                <div class="control-row">
                    ${ui.renderSelect({ id: 'baseline-variant', label: 'Baseline Variant', options: variants, selected: defaultBaseline, className: 'variant-select' })}
                    ${ui.renderSelect({ id: 'compare-variant', label: 'Compare Variant', options: variants, selected: defaultCompare, className: 'variant-select' })}
                </div>

                <div class="control-row">
                    ${ui.renderSelect({ id: 'trait-select', label: 'Trait', options: [], placeholder: 'Loading traits...', className: 'trait-select' })}
                    ${ui.renderSelect({ id: 'prompt-set-select', label: 'Prompt Set', options: [], placeholder: 'Loading prompt sets...', className: 'prompt-set-select' })}
                </div>

                <button id="compute-btn" class="primary-btn">Compute Effect Size</button>
            </div>

            <div id="comparison-results" class="comparison-results" style="display: none;">
                <div class="results-summary" id="results-summary"></div>
                <div id="effect-size-chart"></div>
            </div>

            <div id="comparison-loading" class="loading" style="display: none;">
                Computing effect sizes across layers...
            </div>

            <div id="comparison-error" class="error" style="display: none;"></div>
        </div>
    `;

    // Populate trait selector
    await populateTraitSelector(experiment);

    // Populate prompt set selector
    await populatePromptSetSelector(experiment, defaultBaseline);

    // Add event listeners
    document.getElementById('compute-btn').addEventListener('click', () => {
        runComparison(experiment);
    });

    // Auto-run if we have stored selections
    const storedTrait = sessionStorage.getItem('modelComparison.trait');
    const storedPromptSet = sessionStorage.getItem('modelComparison.promptSet');
    if (storedTrait && storedPromptSet) {
        document.getElementById('trait-select').value = storedTrait;
        document.getElementById('prompt-set-select').value = storedPromptSet;
    }
}

/**
 * Populate trait selector with available traits from experiment
 */
async function populateTraitSelector(experiment) {
    const select = document.getElementById('trait-select');

    try {
        const response = await fetch(`/api/experiments/${experiment}/traits`);
        const data = await response.json();
        const traits = data.traits || [];

        if (traits.length === 0) {
            select.innerHTML = '<option value="">No traits available</option>';
            return;
        }

        select.innerHTML = traits.map(trait =>
            `<option value="${trait}">${trait}</option>`
        ).join('');

    } catch (error) {
        console.error('Failed to load traits:', error);
        select.innerHTML = '<option value="">Failed to load traits</option>';
    }
}

/**
 * Populate prompt set selector - for now just hardcode common ones
 * TODO: Auto-discover from inference/responses directory
 */
async function populatePromptSetSelector(experiment, variant) {
    const select = document.getElementById('prompt-set-select');

    // Hardcoded common prompt sets - ideally we'd auto-discover these
    const commonSets = [
        'train_100',
        'test_150',
        'benign',
        'harmful',
        'single_trait',
        'multi_trait',
        'dynamic',
        'adversarial'
    ];

    select.innerHTML = commonSets.map(set =>
        `<option value="${set}">${set}</option>`
    ).join('');
}

/**
 * Run the comparison analysis
 */
async function runComparison(experiment) {
    const baselineVariant = document.getElementById('baseline-variant').value;
    const compareVariant = document.getElementById('compare-variant').value;
    const trait = document.getElementById('trait-select').value;
    const promptSet = document.getElementById('prompt-set-select').value;

    if (!trait || !promptSet) {
        showError('Please select a trait and prompt set');
        return;
    }

    if (baselineVariant === compareVariant) {
        showError('Baseline and compare variants must be different');
        return;
    }

    // Store selections
    sessionStorage.setItem('modelComparison.trait', trait);
    sessionStorage.setItem('modelComparison.promptSet', promptSet);

    // Show loading
    document.getElementById('comparison-loading').style.display = 'block';
    document.getElementById('comparison-results').style.display = 'none';
    document.getElementById('comparison-error').style.display = 'none';

    try {
        // Load projections for both variants
        const baselineData = await loadProjections(experiment, baselineVariant, trait, promptSet);
        const compareData = await loadProjections(experiment, compareVariant, trait, promptSet);

        if (Object.keys(baselineData.projections).length === 0) {
            throw new Error(`No projections found for ${baselineVariant}/${promptSet}. Run: python inference/project_raw_activations_onto_traits.py --experiment ${experiment} --model-variant ${baselineVariant} --prompt-set ${promptSet}`);
        }

        if (Object.keys(compareData.projections).length === 0) {
            throw new Error(`No projections found for ${compareVariant}/${promptSet}. Run: python inference/project_raw_activations_onto_traits.py --experiment ${experiment} --model-variant ${compareVariant} --prompt-set ${promptSet}`);
        }

        // Get number of layers from model config
        const modelConfig = await window.getModelConfig(experiment);
        const numLayers = modelConfig.n_layers;

        // Aggregate by layer
        const baselineByLayer = aggregateByLayer(baselineData.projections, numLayers);
        const compareByLayer = aggregateByLayer(compareData.projections, numLayers);

        // Compute effect sizes per layer
        const results = [];
        for (let layer = 0; layer < numLayers; layer++) {
            const stats = computeCohenD(baselineByLayer[layer], compareByLayer[layer]);
            results.push({
                layer,
                ...stats
            });
        }

        // Find peak
        const peak = results.reduce((max, r) =>
            Math.abs(r.effectSize) > Math.abs(max.effectSize) ? r : max
        );

        // Display results
        displayResults(results, peak, {
            baselineVariant,
            compareVariant,
            trait,
            promptSet,
            baselineMetadata: baselineData.metadata,
            compareMetadata: compareData.metadata
        });

    } catch (error) {
        showError(error.message);
        console.error('Comparison error:', error);
    } finally {
        document.getElementById('comparison-loading').style.display = 'none';
    }
}

/**
 * Display comparison results
 */
function displayResults(results, peak, context) {
    const resultsDiv = document.getElementById('comparison-results');
    const summaryDiv = document.getElementById('results-summary');
    const chartDiv = document.getElementById('effect-size-chart');

    resultsDiv.style.display = 'block';

    // Summary
    const traitName = context.trait.split('/').pop();
    summaryDiv.innerHTML = `
        <div class="summary-grid">
            <div class="summary-item">
                <div class="summary-label">Baseline</div>
                <div class="summary-value">${context.baselineVariant}</div>
            </div>
            <div class="summary-item">
                <div class="summary-label">Compare</div>
                <div class="summary-value">${context.compareVariant}</div>
            </div>
            <div class="summary-item">
                <div class="summary-label">Trait</div>
                <div class="summary-value">${traitName}</div>
            </div>
            <div class="summary-item">
                <div class="summary-label">Prompt Set</div>
                <div class="summary-value">${context.promptSet}</div>
            </div>
            <div class="summary-item">
                <div class="summary-label">Peak Effect</div>
                <div class="summary-value">${peak.effectSize.toFixed(2)}σ @ L${peak.layer}</div>
            </div>
            <div class="summary-item">
                <div class="summary-label">Responses</div>
                <div class="summary-value">${peak.n}</div>
            </div>
        </div>

        ${context.compareMetadata?.prefilled_from ?
            `<div class="metadata-note">
                <strong>Note:</strong> ${context.compareVariant} was prefilled with responses from ${context.compareMetadata.prefilled_from}
            </div>` : ''
        }
    `;

    // Plot effect size by layer
    renderEffectSizePlot(chartDiv, results, context);
}

/**
 * Render effect size plot using Plotly
 */
function renderEffectSizePlot(container, results, context) {
    const layers = results.map(r => r.layer);
    const effectSizes = results.map(r => r.effectSize);

    const trace = {
        x: layers,
        y: effectSizes,
        type: 'scatter',
        mode: 'lines+markers',
        name: `${context.trait.split('/').pop()}`,
        line: { width: 2 },
        marker: { size: 4 }
    };

    const layout = {
        title: `Effect Size by Layer: ${context.compareVariant} vs ${context.baselineVariant}`,
        xaxis: {
            title: 'Layer',
            dtick: 10
        },
        yaxis: {
            title: 'Effect Size (Cohen\'s d)',
            zeroline: true,
            zerolinewidth: 2,
            zerolinecolor: '#888'
        },
        hovermode: 'closest',
        showlegend: true,
        template: 'plotly_dark',
        plot_bgcolor: 'var(--bg-primary)',
        paper_bgcolor: 'var(--bg-primary)',
        font: { color: 'var(--text-primary)' }
    };

    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false
    };

    Plotly.newPlot(container, [trace], layout, config);
}

/**
 * Show error message
 */
function showError(message) {
    const errorDiv = document.getElementById('comparison-error');
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
    document.getElementById('comparison-results').style.display = 'none';
}

// Export
window.renderModelComparison = renderModelComparison;
