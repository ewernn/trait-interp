// Trait Extraction - Comprehensive view of extraction quality, methods, and vector properties

async function renderTraitExtraction() {
    const contentArea = document.getElementById('content-area');

    contentArea.innerHTML = '<div class="loading">Loading extraction evaluation data...</div>';

    // Load extraction evaluation data
    const evalData = await loadExtractionEvaluation();

    if (!evalData || !evalData.all_results || evalData.all_results.length === 0) {
        contentArea.innerHTML = `
            <div class="info" style="margin: 16px; padding: 12px;">
                <h3>No Extraction Evaluation Data</h3>
                <p>Run the extraction evaluation script to generate quality metrics:</p>
                <pre style="margin-top: 8px; padding: 8px; background: var(--bg-secondary); border-radius: 4px;">python analysis/vectors/extraction_evaluation.py --experiment ${window.state.experimentData?.name || 'your_experiment'}</pre>
            </div>
        `;
        return;
    }

    // Build the comprehensive view
    contentArea.innerHTML = `
        <div class="tool-view">
            <!-- Section 1: Visualizations -->
            <section>
                <h2>Quality Analysis</h2>

                <h3>All Vectors - Sortable Quality Table</h3>
                <div id="quality-table-container"></div>

                <h3>Per-Trait Quality Heatmaps (Layer × Method)</h3>
                <div id="trait-heatmaps-container"></div>

                <h3>Best Vector Per Trait</h3>
                <div id="best-vectors-container"></div>

                <h3>Method Comparison</h3>
                <div id="method-comparison-container" style="height: 300px;"></div>

                <h3>Best-Vector Similarity Matrix (Trait Independence)</h3>
                <div id="best-vector-similarity-container"></div>

                <h3>Cross-Layer Consistency</h3>
                <div id="cross-layer-consistency-container"></div>
            </section>

            <!-- Section 2: Notation -->
            <section>
                <h2>Notation & Definitions <span class="info-icon" data-info="notation">ⓘ</span></h2>
                ${renderNotation()}
            </section>

            <!-- Section 3: Extraction Techniques -->
            <section>
                <h2>Extraction Techniques <span class="info-icon" data-info="techniques">ⓘ</span></h2>
                ${renderExtractionTechniques()}
            </section>

            <!-- Section 4: Metrics Definitions -->
            <section>
                <h2>Quality Metrics <span class="info-icon" data-info="metrics">ⓘ</span></h2>
                ${renderMetricsDefinitions()}
            </section>

            <!-- Section 5: Scoring Method -->
            <section>
                <h2>Scoring & Ranking <span class="info-icon" data-info="scoring">ⓘ</span></h2>
                ${renderScoringExplanation(evalData)}
            </section>

            <!-- Tooltip container -->
            <div id="section-info-tooltip" class="tooltip"></div>
        </div>
    `;

    // Render each visualization
    renderQualityTable(evalData);
    renderTraitHeatmaps(evalData);
    renderBestVectors(evalData);
    renderMethodComparison(evalData);
    renderBestVectorSimilarity(evalData);
    renderCrossLayerConsistency(evalData);

    // Render math after all content is in DOM
    if (window.MathJax) {
        MathJax.typesetPromise();
    }

    // Setup info tooltips
    setupSectionInfoTooltips();
}


// Info tooltip content for each section
const SECTION_INFO_CONTENT = {
    notation: `
        <h4>About This Notation</h4>
        <p>These symbols are used consistently throughout the extraction pipeline and evaluation metrics.</p>
        <p><strong>Key insight:</strong> Each example's activation is the <em>average</em> across all response tokens, giving a single d-dimensional vector per example.</p>
    `,
    techniques: `
        <h4>Choosing an Extraction Method</h4>
        <ul>
            <li><strong>Mean Diff:</strong> Fast baseline. Use for initial exploration.</li>
            <li><strong>Probe:</strong> Best for high-separability traits (>80% accuracy). Optimized for classification.</li>
            <li><strong>ICA:</strong> Use when traits overlap or interfere. Finds independent directions.</li>
            <li><strong>Gradient:</strong> Best for low-separability traits. Can find subtle directions that other methods miss.</li>
        </ul>
        <p><em>Tip: Compare methods in the heatmaps above to see which works best for each trait.</em></p>
    `,
    metrics: `
        <h4>Interpreting Quality Metrics</h4>
        <p>All metrics are computed on <strong>held-out validation data</strong> (20% of examples) to measure generalization.</p>
        <ul>
            <li><strong>Accuracy:</strong> Can the vector classify unseen examples? >90% is good.</li>
            <li><strong>Effect Size (d):</strong> How separated are the distributions? >1.5 is large effect.</li>
            <li><strong>Norm:</strong> Vector magnitude. Typical range: 15-40.</li>
            <li><strong>Margin:</strong> Gap between distributions. Positive = no overlap.</li>
        </ul>
    `,
    scoring: `
        <h4>Why This Scoring Formula?</h4>
        <p>The combined score balances two goals:</p>
        <ul>
            <li><strong>Accuracy (50%):</strong> Practical utility—can we use this vector?</li>
            <li><strong>Effect Size (50%):</strong> Robustness—how confident is the separation?</li>
        </ul>
        <p>Effect size is normalized per-trait because scales vary (0.5–5.0). This makes cross-trait comparison fair.</p>
        <p><em>A vector with 95% accuracy but tiny effect size may be overfitting. This score catches that.</em></p>
    `
};


function setupSectionInfoTooltips() {
    const tooltip = document.getElementById('section-info-tooltip');
    if (!tooltip) return;

    const icons = document.querySelectorAll('.info-icon');

    icons.forEach(icon => {
        icon.addEventListener('click', (e) => {
            e.stopPropagation();
            const key = icon.dataset.info;
            const content = SECTION_INFO_CONTENT[key];

            if (tooltip.classList.contains('show') && tooltip.dataset.activeKey === key) {
                // Toggle off if clicking same icon
                tooltip.classList.remove('show');
                return;
            }

            tooltip.innerHTML = content;
            tooltip.dataset.activeKey = key;

            // Position near the icon
            const rect = icon.getBoundingClientRect();
            const containerRect = document.querySelector('.tool-view').getBoundingClientRect();

            tooltip.style.top = `${rect.bottom - containerRect.top + 8}px`;
            tooltip.style.left = `${rect.left - containerRect.left}px`;

            tooltip.classList.add('show');
        });
    });

    // Close tooltip when clicking elsewhere
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.info-icon') && !e.target.closest('.tooltip')) {
            tooltip.classList.remove('show');
        }
    });
}


async function loadExtractionEvaluation() {
    try {
        const url = window.paths.extractionEvaluation();
        const response = await fetch(url);
        if (!response.ok) return null;
        return await response.json();
    } catch (error) {
        console.error('Failed to load extraction evaluation:', error);
        return null;
    }
}


function renderNotation() {
    return `
        <div class="grid">
            <div class="card">
                <h4>Input Shapes</h4>
                <table class="def-table">
                    <tr><td>$$n$$</td><td>Number of examples (train or validation split)</td></tr>
                    <tr><td>$$d$$</td><td>Hidden dimension (2304 for Gemma 2B)</td></tr>
                    <tr><td>$$L$$</td><td>Number of layers (26 for Gemma 2B)</td></tr>
                    <tr><td>$$\\mathbf{A} \\in \\mathbb{R}^{n \\times d}$$</td><td>Activation matrix (token-averaged per example)</td></tr>
                </table>
            </div>

            <div class="card">
                <h4>Variables</h4>
                <table class="def-table">
                    <tr><td>$$\\vec{v} \\in \\mathbb{R}^d$$</td><td>Trait vector (direction in activation space)</td></tr>
                    <tr><td>$$\\vec{a}_i \\in \\mathbb{R}^d$$</td><td>Single example's activation (row of A)</td></tr>
                    <tr><td>$$y_i \\in \\{+1, -1\\}$$</td><td>Binary label (positive/negative trait)</td></tr>
                    <tr><td>$$\\text{pos}, \\text{neg}$$</td><td>Subscripts for positive/negative example subsets</td></tr>
                </table>
            </div>

            <div class="card">
                <h4>Key Quantities</h4>
                <table class="def-table">
                    <tr><td>$$\\vec{a} \\cdot \\vec{v}$$</td><td>Projection score (dot product)</td></tr>
                    <tr><td>$$\\mu_{\\text{pos}}, \\mu_{\\text{neg}}$$</td><td>Mean projection for pos/neg examples</td></tr>
                    <tr><td>$$\\sigma_{\\text{pooled}}$$</td><td>Pooled standard deviation</td></tr>
                    <tr><td>$$||\\vec{v}||_2$$</td><td>L2 norm (vector magnitude)</td></tr>
                </table>
            </div>

            <div class="card">
                <h4>Pipeline Context</h4>
                <table class="def-table">
                    <tr><td><strong>Train split</strong></td><td>80% of examples → used to extract vectors</td></tr>
                    <tr><td><strong>Val split</strong></td><td>20% of examples → used to evaluate vectors</td></tr>
                    <tr><td><strong>Per-layer</strong></td><td>Vectors extracted independently for each layer</td></tr>
                    <tr><td><strong>Per-method</strong></td><td>4 extraction methods × 26 layers = 104 vectors/trait</td></tr>
                </table>
            </div>
        </div>
    `;
}


function renderExtractionTechniques() {
    return `
        <div class="grid">
            <div class="card">
                <h4>Mean Difference</h4>
                <p>$$\\vec{v} = \\text{mean}(\\mathbf{A}_{\\text{pos}}) - \\text{mean}(\\mathbf{A}_{\\text{neg}})$$</p>
                <p>Simple baseline: average positive activations minus average negative activations.</p>
                <p><strong>Use:</strong> Quick baseline, interpretable direction.</p>
            </div>

            <div class="card">
                <h4>Linear Probe</h4>
                <p>$$\\min_\\vec{w} \\sum_i \\log(1 + e^{-y_i (\\vec{w} \\cdot \\vec{a}_i)})$$</p>
                <p>Train logistic regression classifier, use weights as vector.</p>
                <p><strong>Use:</strong> Best for high-separability traits. Optimized for classification.</p>
            </div>

            <div class="card">
                <h4>ICA (Independent Component Analysis)</h4>
                <p>$$\\mathbf{A} = \\mathbf{S} \\mathbf{M}, \\quad \\text{maximize independence of } \\mathbf{S}$$</p>
                <p>Separate mixed signals into independent components, select component with best separation.</p>
                <p><strong>Use:</strong> When traits overlap or interfere. Finds independent directions.</p>
            </div>

            <div class="card">
                <h4>Gradient Optimization</h4>
                <p>$$\\max_\\vec{v} \\left( \\text{mean}(\\mathbf{A}_{\\text{pos}} \\cdot \\vec{v}) - \\text{mean}(\\mathbf{A}_{\\text{neg}} \\cdot \\vec{v}) \\right)$$</p>
                <p>Directly optimize vector to maximize separation between positive/negative projections.</p>
                <p><strong>Use:</strong> Best for low-separability traits. Adaptive optimization.</p>
            </div>
        </div>
    `;
}


function renderMetricsDefinitions() {
    return `
        <div class="grid">
            <div class="card">
                <h4>Accuracy</h4>
                <p>$$\\text{acc} = \\frac{\\text{correct classifications}}{\\text{total examples}}$$</p>
                <p>Percentage of validation examples correctly classified as positive/negative.</p>
                <p><strong>Range:</strong> 0-1 (50% = random, 100% = perfect). <strong class="quality-good">Good: &gt; 0.90</strong></p>
            </div>

            <div class="card">
                <h4>AUC-ROC</h4>
                <p>$$\\text{AUC} = \\int_0^1 \\text{TPR}(\\text{FPR}^{-1}(t)) \\, dt$$</p>
                <p>Area Under ROC Curve. Threshold-independent measure of classification quality.</p>
                <p><strong>Range:</strong> 0.5-1 (0.5 = random, 1 = perfect). <strong class="quality-good">Good: &gt; 0.90</strong></p>
            </div>

            <div class="card">
                <h4>Effect Size (Cohen's d)</h4>
                <p>$$d = \\frac{\\mu_{\\text{pos}} - \\mu_{\\text{neg}}}{\\sigma_{\\text{pooled}}}$$</p>
                <p>Magnitude of separation between positive/negative distributions in standard deviation units.</p>
                <p><strong>Range:</strong> 0-∞ (0 = no separation, &gt;2 = large effect). <strong class="quality-good">Good: &gt; 1.5</strong></p>
            </div>

            <div class="card">
                <h4>Vector Norm</h4>
                <p>$$||\\vec{v}||_2 = \\sqrt{\\sum_i v_i^2}$$</p>
                <p>L2 norm of the vector. Indicates magnitude/strength.</p>
                <p><strong>Range:</strong> 0-∞. <strong>Typical:</strong> 15-40 for normalized vectors</p>
            </div>

            <div class="card">
                <h4>Separation Margin</h4>
                <p>$$(\\mu_{\\text{pos}} - \\sigma_{\\text{pos}}) - (\\mu_{\\text{neg}} + \\sigma_{\\text{neg}})$$</p>
                <p>Gap between distributions. Positive = good separation, negative = overlap.</p>
                <p><strong>Range:</strong> -∞ to +∞. <strong class="quality-good">Good: &gt; 0</strong></p>
            </div>

            <div class="card">
                <h4>Sparsity</h4>
                <p>$$\\text{sparsity} = \\frac{|\\{i : |v_i| < 0.01\\}|}{d}$$</p>
                <p>Percentage of near-zero components. High sparsity = interpretable, focused vector.</p>
                <p><strong>Range:</strong> 0-1 (0 = dense, 1 = sparse)</p>
            </div>

            <div class="card">
                <h4>Overlap Coefficient</h4>
                <p>$$\\text{overlap} \\approx 1 - \\frac{|\\mu_{\\text{pos}} - \\mu_{\\text{neg}}|}{4\\sigma_{\\text{pooled}}}$$</p>
                <p>Estimate of distribution overlap (0 = no overlap, 1 = complete overlap).</p>
                <p><strong>Range:</strong> 0-1. <strong class="quality-good">Good: &lt; 0.2</strong></p>
            </div>
        </div>
    `;
}


function renderScoringExplanation(evalData) {
    return `
        <div class="card">
            <h4>Combined Score Formula</h4>
            <p>$$\\text{score} = 0.5 \\times \\text{accuracy} + 0.5 \\times \\frac{\\text{effect\\_size}}{\\text{max\\_effect\\_size}}$$</p>

            <h4>Rationale</h4>
            <ul>
                <li><strong>Accuracy (50%):</strong> Measures classification performance. Essential for practical use.</li>
                <li><strong>Normalized Effect Size (50%):</strong> Measures separation magnitude. Prevents overfitting to binary classification.</li>
                <li><strong>Why normalize effect size?</strong> Scale varies across traits (0.5-5.0). Normalization makes comparison fair.</li>
                <li><strong>Why 50/50?</strong> Balances classification accuracy with separation strength. Both matter for vector quality.</li>
            </ul>

            <h4>Alternative Scoring Methods</h4>
            <ul>
                <li><strong>Accuracy-only:</strong> rank by <code>val_accuracy</code></li>
                <li><strong>Effect-size-only:</strong> rank by <code>val_effect_size</code></li>
                <li><strong>Weighted custom:</strong> adjust weights interactively (future feature)</li>
            </ul>
        </div>
    `;
}


function renderQualityTable(evalData) {
    const container = document.getElementById('quality-table-container');
    if (!container) return;

    const results = evalData.all_results || [];
    if (results.length === 0) {
        container.innerHTML = '<p>No results to display.</p>';
        return;
    }

    // Compute combined score for each result
    const maxEffectPerTrait = {};
    results.forEach(r => {
        if (!maxEffectPerTrait[r.trait] || r.val_effect_size > maxEffectPerTrait[r.trait]) {
            maxEffectPerTrait[r.trait] = r.val_effect_size || 0;
        }
    });

    const augmentedResults = results.map(r => {
        const max_d = maxEffectPerTrait[r.trait] || 1;
        const score = 0.5 * (r.val_accuracy || 0) + 0.5 * ((r.val_effect_size || 0) / max_d);
        return { ...r, combined_score: score };
    });

    // Build table HTML
    const tableHTML = `
        <div style="max-height: 600px; overflow-y: auto;">
            <table class="data-table" id="extraction-quality-table">
                <thead>
                    <tr>
                        <th class="sortable" data-column="trait">Trait<span class="sort-indicator">↕</span></th>
                        <th class="sortable" data-column="method">Method<span class="sort-indicator">↕</span></th>
                        <th class="sortable" data-column="layer">Layer<span class="sort-indicator">↕</span></th>
                        <th class="sortable sort-active" data-column="combined_score">Score<span class="sort-indicator">↓</span></th>
                        <th class="sortable" data-column="val_accuracy">Accuracy<span class="sort-indicator">↕</span></th>
                        <th class="sortable" data-column="val_auc_roc">AUC<span class="sort-indicator">↕</span></th>
                        <th class="sortable" data-column="val_effect_size">Effect Size (d)<span class="sort-indicator">↕</span></th>
                        <th class="sortable" data-column="vector_norm">Norm<span class="sort-indicator">↕</span></th>
                    </tr>
                </thead>
                <tbody>
                    ${augmentedResults
                        .sort((a, b) => b.combined_score - a.combined_score)
                        .map(r => {
                            const accClass = (r.val_accuracy >= 0.9) ? 'quality-good' : (r.val_accuracy >= 0.75) ? 'quality-ok' : 'quality-bad';
                            const aucClass = (r.val_auc_roc >= 0.9) ? 'quality-good' : (r.val_auc_roc >= 0.75) ? 'quality-ok' : 'quality-bad';
                            return `
                                <tr>
                                    <td>${window.getDisplayName(r.trait)}</td>
                                    <td>${r.method}</td>
                                    <td>${r.layer}</td>
                                    <td><strong>${r.combined_score.toFixed(3)}</strong></td>
                                    <td class="${accClass}">${(r.val_accuracy * 100).toFixed(1)}%</td>
                                    <td class="${aucClass}">${(r.val_auc_roc * 100).toFixed(1)}%</td>
                                    <td>${r.val_effect_size?.toFixed(2) ?? 'N/A'}</td>
                                    <td>${r.vector_norm?.toFixed(1) ?? 'N/A'}</td>
                                </tr>
                            `;
                        }).join('')}
                </tbody>
            </table>
        </div>
    `;

    container.innerHTML = tableHTML;

    // Add sort functionality
    container.querySelectorAll('.sortable').forEach(th => {
        th.addEventListener('click', () => {
            const column = th.dataset.column;
            const direction = sortDirection[column] === 'asc' ? 'desc' : 'asc';
            sortDirection[column] = direction;

            // Update visual indicators
            container.querySelectorAll('.sortable').forEach(header => {
                header.classList.remove('sort-active');
                header.querySelector('.sort-indicator').textContent = '↕';
            });
            th.classList.add('sort-active');
            th.querySelector('.sort-indicator').textContent = direction === 'asc' ? '↑' : '↓';

            sortQualityTable(augmentedResults, column, container);
        });
    });
}


let sortDirection = { combined_score: 'desc' };
function sortQualityTable(results, column, container) {
    const direction = sortDirection[column];

    const sorted = [...results].sort((a, b) => {
        let valA = a[column];
        let valB = b[column];

        if (column === 'trait' || column === 'method') {
            valA = valA || '';
            valB = valB || '';
            return direction === 'asc' ? valA.localeCompare(valB) : valB.localeCompare(valA);
        } else {
            valA = valA || 0;
            valB = valB || 0;
            return direction === 'asc' ? valA - valB : valB - valA;
        }
    });

    // Re-render tbody
    const tbody = container.querySelector('tbody');
    tbody.innerHTML = sorted.map(r => {
        const accClass = (r.val_accuracy >= 0.9) ? 'quality-good' : (r.val_accuracy >= 0.75) ? 'quality-ok' : 'quality-bad';
        const aucClass = (r.val_auc_roc >= 0.9) ? 'quality-good' : (r.val_auc_roc >= 0.75) ? 'quality-ok' : 'quality-bad';
        return `
            <tr>
                <td>${window.getDisplayName(r.trait)}</td>
                <td>${r.method}</td>
                <td>${r.layer}</td>
                <td><strong>${r.combined_score.toFixed(3)}</strong></td>
                <td class="${accClass}">${(r.val_accuracy * 100).toFixed(1)}%</td>
                <td class="${aucClass}">${(r.val_auc_roc * 100).toFixed(1)}%</td>
                <td>${r.val_effect_size?.toFixed(2) ?? 'N/A'}</td>
                <td>${r.vector_norm?.toFixed(1) ?? 'N/A'}</td>
            </tr>
        `;
    }).join('');
}


function renderTraitHeatmaps(evalData) {
    const container = document.getElementById('trait-heatmaps-container');
    if (!container) return;

    const results = evalData.all_results || [];
    if (results.length === 0) {
        container.innerHTML = '<p>No results to display.</p>';
        return;
    }

    // Group by trait
    const traitGroups = {};
    results.forEach(r => {
        if (!traitGroups[r.trait]) traitGroups[r.trait] = [];
        traitGroups[r.trait].push(r);
    });

    const traits = Object.keys(traitGroups).sort();

    // Create header with shared legend
    container.innerHTML = `
        <div class="heatmap-legend-header">
            <span style="font-size: 12px; color: var(--text-secondary);">${traits.length} traits</span>
            <div class="heatmap-legend">
                <span>Accuracy:</span>
                <div>
                    <div class="heatmap-legend-bar"></div>
                    <div class="heatmap-legend-labels">
                        <span>50%</span>
                        <span>75%</span>
                        <span>100%</span>
                    </div>
                </div>
            </div>
        </div>
        <div class="trait-heatmaps-grid" id="heatmaps-grid"></div>
    `;

    const grid = document.getElementById('heatmaps-grid');

    // Create compact heatmap for each trait
    traits.forEach(trait => {
        const traitResults = traitGroups[trait];
        const traitId = trait.replace(/\//g, '-');
        const displayName = window.getDisplayName(trait);

        const traitDiv = document.createElement('div');
        traitDiv.className = 'trait-heatmap-item';
        traitDiv.innerHTML = `
            <h4 title="${displayName}">${displayName}</h4>
            <div id="heatmap-${traitId}" style="width: 100%; height: 120px;"></div>
        `;

        grid.appendChild(traitDiv);

        renderSingleTraitHeatmap(traitResults, `heatmap-${traitId}`, true);
    });
}


function renderSingleTraitHeatmap(traitResults, containerId, compact = false) {
    const methods = ['mean_diff', 'probe', 'ica', 'gradient'];
    const layers = Array.from(new Set(traitResults.map(r => r.layer))).sort((a, b) => a - b);

    // Build matrix: layers × methods, value = accuracy
    const matrix = [];
    layers.forEach(layer => {
        const row = methods.map(method => {
            const result = traitResults.find(r => r.layer === layer && r.method === method);
            return result ? result.val_accuracy * 100 : null;
        });
        matrix.push(row);
    });

    const trace = {
        z: matrix,
        x: compact ? ['MD', 'Pr', 'ICA', 'Gr'] : methods,
        y: layers,
        type: 'heatmap',
        colorscale: window.ASYMB_COLORSCALE,
        hovertemplate: '%{x} L%{y}: %{z:.1f}%<extra></extra>',
        zmin: 50,
        zmax: 100,
        showscale: !compact
    };

    if (!compact) {
        trace.colorbar = {
            title: { text: 'Accuracy %', font: { size: 11 } },
            tickvals: [50, 75, 90, 100],
            ticktext: ['50%', '75%', '90%', '100%']
        };
    }

    const layout = compact ? {
        margin: { l: 25, r: 5, t: 5, b: 25 },
        xaxis: { tickfont: { size: 8 }, tickangle: 0 },
        yaxis: { tickfont: { size: 8 }, title: '' },
        height: 120
    } : {
        margin: { l: 40, r: 80, t: 20, b: 60 },
        xaxis: { title: 'Method', tickfont: { size: 11 } },
        yaxis: { title: 'Layer', tickfont: { size: 10 } },
        height: 400
    };

    Plotly.newPlot(containerId, [trace], window.getPlotlyLayout(layout), { displayModeBar: false, responsive: true });
}


function renderBestVectors(evalData) {
    const container = document.getElementById('best-vectors-container');
    if (!container) return;

    const bestPerTrait = evalData.best_per_trait || [];
    if (bestPerTrait.length === 0) {
        container.innerHTML = '<p>No best vectors found.</p>';
        return;
    }

    const tableHTML = `
        <table class="data-table">
            <thead>
                <tr>
                    <th>Trait</th>
                    <th>Best Method</th>
                    <th>Best Layer</th>
                    <th>Accuracy</th>
                    <th>Effect Size (d)</th>
                    <th>Norm</th>
                </tr>
            </thead>
            <tbody>
                ${bestPerTrait.map(r => `
                    <tr>
                        <td><strong>${window.getDisplayName(r.trait)}</strong></td>
                        <td>${r.method}</td>
                        <td>${r.layer}</td>
                        <td>${(r.val_accuracy * 100).toFixed(1)}%</td>
                        <td>${r.val_effect_size?.toFixed(2) ?? 'N/A'}</td>
                        <td>${r.vector_norm?.toFixed(1) ?? 'N/A'}</td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;

    container.innerHTML = tableHTML;
}


function renderMethodComparison(evalData) {
    const container = document.getElementById('method-comparison-container');
    if (!container) return;

    const methodSummary = evalData.method_summary || {};
    if (Object.keys(methodSummary).length === 0) {
        container.innerHTML = '<p>No method summary available.</p>';
        return;
    }

    // Extract mean accuracy per method
    const accMean = methodSummary['val_accuracy_mean'] || {};
    const methods = Object.keys(accMean);
    const meanAccuracies = methods.map(m => accMean[m] * 100);

    const trace = {
        x: methods,
        y: meanAccuracies,
        type: 'bar',
        marker: { color: getCssVar('--primary-color', '#a09f6c') },
        text: meanAccuracies.map(v => v.toFixed(1) + '%'),
        textposition: 'outside'
    };

    Plotly.newPlot(container, [trace], window.getPlotlyLayout({
        margin: { l: 60, r: 20, t: 20, b: 60 },
        xaxis: { title: 'Method' },
        yaxis: { title: 'Mean Validation Accuracy (%)', range: [0, 100] },
        height: 300
    }), { displayModeBar: false, responsive: true });
}


function renderBestVectorSimilarity(evalData) {
    const container = document.getElementById('best-vector-similarity-container');
    if (!container) return;

    const similarityMatrix = evalData.best_vector_similarity || {};
    const traits = Object.keys(similarityMatrix);

    if (traits.length === 0) {
        container.innerHTML = '<p>No similarity matrix available.</p>';
        return;
    }

    // Convert to 2D array
    const matrix = traits.map(t1 =>
        traits.map(t2 => similarityMatrix[t1][t2])
    );

    const displayNames = traits.map(t => window.getDisplayName(t));

    const trace = {
        z: matrix,
        x: displayNames,
        y: displayNames,
        type: 'heatmap',
        colorscale: window.ASYMB_COLORSCALE,
        zmid: 0,
        zmin: -1,
        zmax: 1,
        colorbar: {
            title: { text: 'Similarity', font: { size: 11 } },
            tickvals: [-1, -0.5, 0, 0.5, 1]
        },
        hovertemplate: '%{y} ↔ %{x}<br>sim = %{z:.3f}<extra></extra>',
        texttemplate: '%{z:.2f}',
        textfont: { size: 9 }
    };

    Plotly.newPlot(container, [trace], window.getPlotlyLayout({
        margin: { l: 150, r: 80, t: 100, b: 150 },
        xaxis: { side: 'top', tickangle: -45, tickfont: { size: 10 } },
        yaxis: { tickfont: { size: 10 } },
        height: 600
    }), { displayModeBar: false, responsive: true });
}


function renderCrossLayerConsistency(evalData) {
    const container = document.getElementById('cross-layer-consistency-container');
    if (!container) return;

    const consistencyData = evalData.cross_layer_consistency || {};
    const traits = Object.keys(consistencyData);

    if (traits.length === 0) {
        container.innerHTML = '<p>No cross-layer consistency data available.</p>';
        return;
    }

    // Extract mean consistency per trait
    const traitNames = traits.map(t => window.getDisplayName(t));
    const meanValues = traits.map(t => consistencyData[t]?.mean || 0);

    // Sort by consistency (descending)
    const sortedIndices = meanValues.map((v, i) => i).sort((a, b) => meanValues[b] - meanValues[a]);
    const sortedTraits = sortedIndices.map(i => traitNames[i]);
    const sortedValues = sortedIndices.map(i => meanValues[i]);

    const trace = {
        x: sortedTraits,
        y: sortedValues,
        type: 'bar',
        marker: {
            color: sortedValues.map(v => v >= 0.8 ? getCssVar('--success', '#4a9') : v >= 0.5 ? getCssVar('--warning', '#a94') : getCssVar('--danger', '#a44'))
        },
        text: sortedValues.map(v => v.toFixed(2)),
        textposition: 'outside'
    };

    Plotly.newPlot(container, [trace], window.getPlotlyLayout({
        margin: { l: 60, r: 20, t: 20, b: 120 },
        xaxis: { tickangle: -45, tickfont: { size: 10 } },
        yaxis: { title: 'Mean Cosine Similarity', range: [0, 1] },
        height: 350
    }), { displayModeBar: false, responsive: true });
}


// CSS helper
function getCssVar(name, fallback = '') {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
}


// Export
window.renderTraitExtraction = renderTraitExtraction;
