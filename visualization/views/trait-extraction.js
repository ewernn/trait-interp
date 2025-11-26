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
        <div class="extraction-overview-container">
            <!-- Section 1: Extraction Techniques -->
            <div class="extraction-section">
                <h2 class="section-heading">Extraction Techniques</h2>
                <div class="techniques-grid">
                    ${renderExtractionTechniques()}
                </div>
            </div>

            <!-- Section 2: Metrics Definitions -->
            <div class="extraction-section">
                <h2 class="section-heading">Quality Metrics</h2>
                <div class="metrics-definitions">
                    ${renderMetricsDefinitions()}
                </div>
            </div>

            <!-- Section 3: Scoring Method -->
            <div class="extraction-section">
                <h2 class="section-heading">Scoring & Ranking</h2>
                <div class="scoring-explanation">
                    ${renderScoringExplanation(evalData)}
                </div>
            </div>

            <!-- Section 4: Visualizations -->
            <div class="extraction-section">
                <h2 class="section-heading">Quality Analysis</h2>

                <!-- Overall Quality Table -->
                <div class="card" style="margin-bottom: 24px;">
                    <div class="card-title">All Vectors - Sortable Quality Table</div>
                    <div id="quality-table-container"></div>
                </div>

                <!-- Per-Trait Layer×Method Heatmaps -->
                <div class="card" style="margin-bottom: 24px;">
                    <div class="card-title">Per-Trait Quality Heatmaps (Layer × Method)</div>
                    <div id="trait-heatmaps-container"></div>
                </div>

                <!-- Best Vectors Summary -->
                <div class="card" style="margin-bottom: 24px;">
                    <div class="card-title">Best Vector Per Trait</div>
                    <div id="best-vectors-container"></div>
                </div>

                <!-- Method Comparison -->
                <div class="card" style="margin-bottom: 24px;">
                    <div class="card-title">Method Comparison</div>
                    <div id="method-comparison-container"></div>
                </div>

                <!-- Cross-Trait Independence -->
                <div class="card">
                    <div class="card-title">Best-Vector Similarity Matrix (Trait Independence)</div>
                    <div id="best-vector-similarity-container"></div>
                </div>
            </div>
        </div>
    `;

    // Render each visualization
    renderQualityTable(evalData);
    renderTraitHeatmaps(evalData);
    renderBestVectors(evalData);
    renderMethodComparison(evalData);
    renderBestVectorSimilarity(evalData);
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


function renderExtractionTechniques() {
    return `
        <div class="technique-card">
            <h4>Mean Difference</h4>
            <div class="technique-math">$$\\vec{v} = \\text{mean}(\\mathbf{A}_{\\text{pos}}) - \\text{mean}(\\mathbf{A}_{\\text{neg}})$$</div>
            <p class="technique-desc">Simple baseline: average positive activations minus average negative activations.</p>
            <p class="technique-use"><strong>Use:</strong> Quick baseline, interpretable direction.</p>
            <div class="technique-shape">$$\\mathbf{A} \\in \\mathbb{R}^{n \\times d} \\rightarrow \\vec{v} \\in \\mathbb{R}^d$$</div>
        </div>

        <div class="technique-card">
            <h4>Linear Probe</h4>
            <div class="technique-math">$$\\min_\\vec{w} \\sum_i \\log(1 + e^{-y_i (\\vec{w} \\cdot \\vec{a}_i)})$$</div>
            <p class="technique-desc">Train logistic regression classifier, use weights as vector.</p>
            <p class="technique-use"><strong>Use:</strong> Best for high-separability traits. Optimized for classification.</p>
            <div class="technique-shape">$$\\text{Labels: } y_i \\in \\{+1, -1\\}, \\quad \\vec{w} \\in \\mathbb{R}^d$$</div>
        </div>

        <div class="technique-card">
            <h4>ICA (Independent Component Analysis)</h4>
            <div class="technique-math">$$\\mathbf{A} = \\mathbf{S} \\mathbf{M}, \\quad \\text{maximize independence of } \\mathbf{S}$$</div>
            <p class="technique-desc">Separate mixed signals into independent components, select component with best separation.</p>
            <p class="technique-use"><strong>Use:</strong> When traits overlap or interfere. Finds independent directions.</p>
            <div class="technique-shape">$$\\mathbf{S} \\in \\mathbb{R}^{n \\times k}, \\quad \\vec{v} = \\text{best component}$$</div>
        </div>

        <div class="technique-card">
            <h4>Gradient Optimization</h4>
            <div class="technique-math">$$\\max_\\vec{v} \\left( \\text{mean}(\\mathbf{A}_{\\text{pos}} \\cdot \\vec{v}) - \\text{mean}(\\mathbf{A}_{\\text{neg}} \\cdot \\vec{v}) \\right)$$</div>
            <p class="technique-desc">Directly optimize vector to maximize separation between positive/negative projections.</p>
            <p class="technique-use"><strong>Use:</strong> Best for low-separability traits. Adaptive optimization.</p>
            <div class="technique-shape">$$\\text{Adam optimizer, 100 steps, lr=0.01}$$</div>
        </div>
    `;
}


function renderMetricsDefinitions() {
    return `
        <div class="metrics-grid">
            <div class="metric-card">
                <h4>Accuracy</h4>
                <div class="metric-formula">$$\\text{acc} = \\frac{\\text{correct classifications}}{\\text{total examples}}$$</div>
                <p>Percentage of validation examples correctly classified as positive/negative.</p>
                <p class="metric-range"><strong>Range:</strong> 0-1 (50% = random, 100% = perfect)</p>
                <p class="metric-good"><strong>Good:</strong> &gt; 0.90</p>
            </div>

            <div class="metric-card">
                <h4>Effect Size (Cohen's d)</h4>
                <div class="metric-formula">$$d = \\frac{\\mu_{\\text{pos}} - \\mu_{\\text{neg}}}{\\sigma_{\\text{pooled}}}$$</div>
                <p>Magnitude of separation between positive/negative distributions in standard deviation units.</p>
                <p class="metric-range"><strong>Range:</strong> 0-∞ (0 = no separation, &gt;2 = large effect)</p>
                <p class="metric-good"><strong>Good:</strong> &gt; 1.5</p>
            </div>

            <div class="metric-card">
                <h4>Vector Norm</h4>
                <div class="metric-formula">$$||\\vec{v}||_2 = \\sqrt{\\sum_i v_i^2}$$</div>
                <p>L2 norm of the vector. Indicates magnitude/strength.</p>
                <p class="metric-range"><strong>Range:</strong> 0-∞</p>
                <p class="metric-good"><strong>Typical:</strong> 15-40 for normalized vectors</p>
            </div>

            <div class="metric-card">
                <h4>Separation Margin</h4>
                <div class="metric-formula">$$(\\mu_{\\text{pos}} - \\sigma_{\\text{pos}}) - (\\mu_{\\text{neg}} + \\sigma_{\\text{neg}})$$</div>
                <p>Gap between distributions. Positive = good separation, negative = overlap.</p>
                <p class="metric-range"><strong>Range:</strong> -∞ to +∞</p>
                <p class="metric-good"><strong>Good:</strong> &gt; 0</p>
            </div>

            <div class="metric-card">
                <h4>Sparsity</h4>
                <div class="metric-formula">$$\\text{sparsity} = \\frac{|\\{i : |v_i| < 0.01\\}|}{d}$$</div>
                <p>Percentage of near-zero components. High sparsity = interpretable, focused vector.</p>
                <p class="metric-range"><strong>Range:</strong> 0-1 (0 = dense, 1 = sparse)</p>
            </div>

            <div class="metric-card">
                <h4>Overlap Coefficient</h4>
                <div class="metric-formula">$$\\text{overlap} \\approx 1 - \\frac{|\\mu_{\\text{pos}} - \\mu_{\\text{neg}}|}{4\\sigma_{\\text{pooled}}}$$</div>
                <p>Estimate of distribution overlap (0 = no overlap, 1 = complete overlap).</p>
                <p class="metric-range"><strong>Range:</strong> 0-1</p>
                <p class="metric-good"><strong>Good:</strong> &lt; 0.2</p>
            </div>
        </div>
    `;
}


function renderScoringExplanation(evalData) {
    return `
        <div class="scoring-content">
            <h3>Combined Score Formula</h3>
            <div class="score-formula">
                $$\\text{score} = 0.5 \\times \\text{accuracy} + 0.5 \\times \\frac{\\text{effect\\_size}}{\\text{max\\_effect\\_size}}$$
            </div>

            <h4>Rationale</h4>
            <ul class="scoring-rationale">
                <li><strong>Accuracy (50%):</strong> Measures classification performance. Essential for practical use.</li>
                <li><strong>Normalized Effect Size (50%):</strong> Measures separation magnitude. Prevents overfitting to binary classification.</li>
                <li><strong>Why normalize effect size?</strong> Scale varies across traits (0.5-5.0). Normalization makes comparison fair.</li>
                <li><strong>Why 50/50?</strong> Balances classification accuracy with separation strength. Both matter for vector quality.</li>
            </ul>

            <h4>Empirical Validation</h4>
            <p><em>(This section will be populated after analyzing the data distribution - showing how this scoring separates good/bad vectors better than alternatives.)</em></p>

            <div class="scoring-alternatives" style="margin-top: 16px;">
                <h4>Alternative Scoring Methods (for comparison)</h4>
                <ul>
                    <li><strong>Accuracy-only:</strong> rank by <code>val_accuracy</code></li>
                    <li><strong>Effect-size-only:</strong> rank by <code>val_effect_size</code></li>
                    <li><strong>Weighted custom:</strong> adjust weights interactively (future feature)</li>
                </ul>
            </div>
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
            <table class="quality-table">
                <thead>
                    <tr>
                        <th class="sortable" data-column="trait">Trait</th>
                        <th class="sortable" data-column="method">Method</th>
                        <th class="sortable" data-column="layer">Layer</th>
                        <th class="sortable" data-column="combined_score">Score ↓</th>
                        <th class="sortable" data-column="val_accuracy">Accuracy</th>
                        <th class="sortable" data-column="val_effect_size">Effect Size (d)</th>
                        <th class="sortable" data-column="vector_norm">Norm</th>
                        <th class="sortable" data-column="separation_margin">Margin</th>
                    </tr>
                </thead>
                <tbody>
                    ${augmentedResults
                        .sort((a, b) => b.combined_score - a.combined_score)
                        .map(r => {
                            const accClass = (r.val_accuracy >= 0.9) ? 'quality-good' : (r.val_accuracy >= 0.75) ? 'quality-ok' : 'quality-bad';
                            return `
                                <tr>
                                    <td>${window.getDisplayName(r.trait)}</td>
                                    <td>${r.method}</td>
                                    <td>${r.layer}</td>
                                    <td><strong>${r.combined_score.toFixed(3)}</strong></td>
                                    <td class="${accClass}">${(r.val_accuracy * 100).toFixed(1)}%</td>
                                    <td>${r.val_effect_size?.toFixed(2) || 'N/A'}</td>
                                    <td>${r.vector_norm?.toFixed(1) || 'N/A'}</td>
                                    <td>${r.separation_margin?.toFixed(2) || 'N/A'}</td>
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
            sortQualityTable(augmentedResults, column, container);
        });
    });
}


let sortDirection = {};
function sortQualityTable(results, column, container) {
    const direction = sortDirection[column] === 'asc' ? 'desc' : 'asc';
    sortDirection[column] = direction;

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
        return `
            <tr>
                <td>${window.getDisplayName(r.trait)}</td>
                <td>${r.method}</td>
                <td>${r.layer}</td>
                <td><strong>${r.combined_score.toFixed(3)}</strong></td>
                <td class="${accClass}">${(r.val_accuracy * 100).toFixed(1)}%</td>
                <td>${r.val_effect_size?.toFixed(2) || 'N/A'}</td>
                <td>${r.vector_norm?.toFixed(1) || 'N/A'}</td>
                <td>${r.separation_margin?.toFixed(2) || 'N/A'}</td>
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

    // Create heatmap for each trait
    container.innerHTML = '';
    const traits = Object.keys(traitGroups).sort();

    traits.forEach(trait => {
        const traitResults = traitGroups[trait];
        const traitDiv = document.createElement('div');
        traitDiv.style.marginBottom = '32px';

        const traitId = trait.replace(/\//g, '-');
        const displayName = window.getDisplayName(trait);

        traitDiv.innerHTML = `
            <h4 style="margin-bottom: 12px;">${displayName}</h4>
            <div id="heatmap-${traitId}" style="width: 100%; height: 400px;"></div>
        `;

        container.appendChild(traitDiv);

        renderSingleTraitHeatmap(traitResults, `heatmap-${traitId}`);
    });
}


function renderSingleTraitHeatmap(traitResults, containerId) {
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
        x: methods,
        y: layers,
        type: 'heatmap',
        colorscale: window.ASYMB_COLORSCALE,
        hovertemplate: '%{x} L%{y}: %{z:.1f}%<extra></extra>',
        zmin: 50,
        zmax: 100,
        colorbar: {
            title: { text: 'Accuracy %', font: { size: 11 } },
            tickvals: [50, 75, 90, 100],
            ticktext: ['50%', '75%', '90%', '100%']
        }
    };

    Plotly.newPlot(containerId, [trace], window.getPlotlyLayout({
        margin: { l: 40, r: 80, t: 20, b: 60 },
        xaxis: { title: 'Method', tickfont: { size: 11 } },
        yaxis: { title: 'Layer', tickfont: { size: 10 } },
        height: 400
    }), { displayModeBar: false, responsive: true });
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
        <table class="best-vectors-table">
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
                        <td>${r.val_effect_size?.toFixed(2) || 'N/A'}</td>
                        <td>${r.vector_norm?.toFixed(1) || 'N/A'}</td>
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

    // Render math
    if (window.MathJax) {
        MathJax.typesetPromise();
    }
}


// CSS helper
function getCssVar(name, fallback = '') {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
}


// Export
window.renderTraitExtraction = renderTraitExtraction;
