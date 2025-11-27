// Analysis Gallery View - Unified live-rendered analysis with token slider support
// Replaces static PNGs with interactive Plotly visualizations

let galleryData = null;
let galleryCache = { experiment: null, promptSet: null, promptId: null };

// =============================================================================
// DATA LOADING
// =============================================================================

async function loadGalleryData() {
    const experiment = window.state.experimentData?.name;
    const promptSet = window.state.currentPromptSet;
    const promptId = window.state.currentPromptId;

    if (!experiment || !promptSet || !promptId) return null;

    // Check cache
    if (galleryCache.experiment === experiment &&
        galleryCache.promptSet === promptSet &&
        galleryCache.promptId === promptId &&
        galleryData) {
        return galleryData;
    }

    const url = window.paths.analysisPerToken(promptSet, promptId);

    try {
        const response = await fetch(url);
        if (!response.ok) {
            console.error('Failed to load per-token data:', response.status);
            return null;
        }
        galleryData = await response.json();
        galleryCache = { experiment, promptSet, promptId };
        return galleryData;
    } catch (error) {
        console.error('Error loading per-token data:', error);
        return null;
    }
}

// =============================================================================
// MAIN RENDER
// =============================================================================

async function renderAnalysisGallery() {
    const contentArea = document.getElementById('content-area');
    const experiment = window.state.experimentData?.name;

    if (!experiment) {
        contentArea.innerHTML = '<div class="error">No experiment selected</div>';
        return;
    }

    // Check if DOM exists and data is cached (avoid scroll reset on slider move)
    const existingGallery = contentArea.querySelector('.analysis-gallery');
    const dataIsCached = galleryCache.experiment === experiment &&
                         galleryCache.promptSet === window.state.currentPromptSet &&
                         galleryCache.promptId === window.state.currentPromptId &&
                         galleryData;

    if (!dataIsCached) {
        contentArea.innerHTML = '<div class="loading">Loading analysis data...</div>';
    }

    const data = await loadGalleryData();

    if (!data) {
        contentArea.innerHTML = `
            <div class="info" style="margin: 16px; padding: 16px;">
                <h3>No per-token data available</h3>
                <p>Run the per-token analysis script first:</p>
                <code>python experiments/${experiment}/analysis/compute_per_token_all_sets.py</code>
            </div>
        `;
        return;
    }

    const tokenIdx = Math.min(window.state.currentTokenIndex || 0, data.n_total_tokens - 1);
    const tokenData = data.per_token[tokenIdx];

    // If DOM exists and data cached, just update visualizations
    if (existingGallery && dataIsCached) {
        updateGalleryVisualizations(data, tokenIdx, tokenData);
        return;
    }

    // Full render
    contentArea.innerHTML = `
        <div class="tool-view analysis-gallery">
            <section>
                <h3>All Tokens</h3>
                <p class="section-desc">Slider highlights the selected token's column/row</p>
                <div class="grid" style="grid-template-columns: 1fr 1fr;">
                    <div class="card">
                        <h4>Trait Scores <span class="layer-badge">Layer 16</span></h4>
                        <div id="trait-heatmap-container"></div>
                    </div>
                    <div class="card">
                        <h4>Normalized Velocity</h4>
                        <div id="velocity-heatmap-container"></div>
                    </div>
                </div>
            </section>

            <section>
                <h3>Selected Token: <span class="current-token-label">"${escapeHtml(tokenData.token)}"</span></h3>
                <p class="section-desc">Updates as you move the slider</p>
                <div class="grid" style="grid-template-columns: 1fr 1fr;">
                    <div class="card">
                        <h4>Trait Scores <span class="layer-badge">Layer 16</span></h4>
                        <div id="trait-scores-container"></div>
                    </div>
                    <div class="card">
                        <h4>Attention Pattern <span class="layer-badge">Layer 16</span></h4>
                        <div id="attention-container"></div>
                    </div>
                </div>
            </section>

            <section>
                <h3>Aggregate</h3>
                <p class="section-desc">Computed across all tokens (slider ignored)</p>
                <div class="grid" style="grid-template-columns: 1fr 1fr;">
                    <div class="card">
                        <h4>Trait Emergence</h4>
                        <div id="trait-emergence-container"></div>
                    </div>
                    <div class="card">
                        <h4>Trait-Dynamics Correlation</h4>
                        <div id="dynamics-correlation-container"></div>
                    </div>
                </div>
            </section>

            ${getCategoryReference()}
        </div>
    `;

    // Render all visualizations
    renderAllVisualizations(data, tokenIdx, tokenData);
}

function updateGalleryVisualizations(data, tokenIdx, tokenData) {
    // Update current token label
    const label = document.querySelector('.current-token-label');
    if (label) label.textContent = `"${escapeHtml(tokenData.token)}"`;

    // Re-render all (Plotly handles updates efficiently)
    renderAllVisualizations(data, tokenIdx, tokenData);
}

function renderAllVisualizations(data, tokenIdx, tokenData) {
    renderTraitHeatmap(data, tokenIdx);
    renderVelocityHeatmap(data, tokenIdx);
    renderTraitScoresBar(tokenData);
    renderAttentionPattern(tokenData, data);
    renderTraitEmergence(data);
    renderDynamicsCorrelation(data);
}

// =============================================================================
// ALL TOKENS VISUALIZATIONS (slider highlights)
// =============================================================================

function renderTraitHeatmap(data, currentTokenIdx) {
    const container = document.getElementById('trait-heatmap-container');
    if (!container) return;

    // Get traits from first token with data
    const firstToken = data.per_token.find(t => t.trait_scores_per_layer);
    if (!firstToken) {
        container.innerHTML = '<div class="no-data">No trait data</div>';
        return;
    }

    const traits = Object.keys(firstToken.trait_scores_per_layer);
    const layer = 16;

    // Build matrix [traits × tokens]
    const zData = traits.map(trait =>
        data.per_token.map(t => t.trait_scores_per_layer?.[trait]?.[layer] ?? 0)
    );

    const trace = {
        z: zData,
        x: data.tokens.map((t, i) => i),
        y: traits,
        type: 'heatmap',
        colorscale: 'RdBu',
        zmid: 0,
        hovertemplate: '%{y}<br>Token %{x}: %{z:.2f}<extra></extra>',
        showscale: true,
        colorbar: { thickness: 15, len: 0.8 }
    };

    // Highlight current token
    const shapes = [{
        type: 'rect',
        x0: currentTokenIdx - 0.5,
        x1: currentTokenIdx + 0.5,
        y0: -0.5,
        y1: traits.length - 0.5,
        line: { color: '#ffff00', width: 2 },
        fillcolor: 'rgba(0,0,0,0)'
    }];

    const layout = {
        margin: { l: 100, r: 50, t: 10, b: 40 },
        height: 250,
        xaxis: { title: 'Token', dtick: 10 },
        yaxis: { tickfont: { size: 10 } },
        shapes
    };

    Plotly.newPlot(container, [trace], layout, { responsive: true });
}

function renderVelocityHeatmap(data, currentTokenIdx) {
    const container = document.getElementById('velocity-heatmap-container');
    if (!container) return;

    // Build matrix [tokens × layer_transitions]
    const nLayers = 25; // transitions
    const zData = data.per_token.map(t => t.normalized_velocity_per_layer || new Array(nLayers).fill(0));

    const trace = {
        z: zData,
        x: Array.from({ length: nLayers }, (_, i) => i),
        y: data.tokens.map((t, i) => i),
        type: 'heatmap',
        colorscale: 'Viridis',
        hovertemplate: 'Token %{y}, Layer %{x}→%{x+1}<br>Velocity: %{z:.3f}<extra></extra>',
        showscale: true,
        colorbar: { thickness: 15, len: 0.8 }
    };

    // Highlight current token row
    const shapes = [{
        type: 'rect',
        x0: -0.5,
        x1: nLayers - 0.5,
        y0: currentTokenIdx - 0.5,
        y1: currentTokenIdx + 0.5,
        line: { color: '#ffff00', width: 2 },
        fillcolor: 'rgba(0,0,0,0)'
    }];

    const layout = {
        margin: { l: 50, r: 50, t: 10, b: 40 },
        height: 250,
        xaxis: { title: 'Layer Transition', dtick: 5 },
        yaxis: { title: 'Token', dtick: 10 },
        shapes
    };

    Plotly.newPlot(container, [trace], layout, { responsive: true });
}

// =============================================================================
// SELECTED TOKEN VISUALIZATIONS (slider updates)
// =============================================================================

function renderTraitScoresBar(tokenData) {
    const container = document.getElementById('trait-scores-container');
    if (!container || !tokenData.trait_scores_per_layer) {
        if (container) container.innerHTML = '<div class="no-data">No trait data for this token</div>';
        return;
    }

    const layer = 16;
    const traits = Object.keys(tokenData.trait_scores_per_layer);
    const scores = traits.map(t => tokenData.trait_scores_per_layer[t][layer]);

    // Sort by absolute value
    const sorted = traits.map((t, i) => ({ trait: t, score: scores[i] }))
        .sort((a, b) => Math.abs(b.score) - Math.abs(a.score));

    const trace = {
        x: sorted.map(s => s.score),
        y: sorted.map(s => s.trait),
        type: 'bar',
        orientation: 'h',
        marker: {
            color: sorted.map(s => s.score > 0 ? '#4ecdc4' : '#ff6b6b')
        },
        hovertemplate: '%{y}: %{x:.3f}<extra></extra>'
    };

    const layout = {
        margin: { l: 100, r: 20, t: 10, b: 40 },
        height: 250,
        xaxis: { title: 'Score', zeroline: true, zerolinecolor: '#888' },
        yaxis: { tickfont: { size: 10 } }
    };

    Plotly.newPlot(container, [trace], layout, { responsive: true });
}

function renderAttentionPattern(tokenData, data) {
    const container = document.getElementById('attention-container');
    if (!container) return;

    if (!tokenData.attention_pattern_L16) {
        container.innerHTML = `
            <div class="no-data">
                No attention data<br>
                <small>(Only for dynamic prompts)</small>
            </div>
        `;
        return;
    }

    const attnPattern = tokenData.attention_pattern_L16;
    const contextSize = tokenData.attention_context_size || attnPattern[0]?.length || 0;
    const nHeads = attnPattern.length;

    // Average across heads
    const avgAttn = [];
    for (let i = 0; i < contextSize; i++) {
        let sum = 0;
        for (let h = 0; h < nHeads; h++) {
            sum += attnPattern[h]?.[i] || 0;
        }
        avgAttn.push(sum / nHeads);
    }

    const trace = {
        x: Array.from({ length: contextSize }, (_, i) => i),
        y: avgAttn,
        type: 'bar',
        marker: { color: '#9b59b6' },
        hovertemplate: 'Pos %{x}: %{y:.3f}<extra></extra>'
    };

    const layout = {
        margin: { l: 50, r: 20, t: 10, b: 40 },
        height: 250,
        xaxis: { title: 'Context Position', dtick: 10 },
        yaxis: { title: 'Attention' }
    };

    Plotly.newPlot(container, [trace], layout, { responsive: true });
}

// =============================================================================
// AGGREGATE VISUALIZATIONS (slider ignored)
// =============================================================================

function renderTraitEmergence(data) {
    const container = document.getElementById('trait-emergence-container');
    if (!container) return;

    // Get traits
    const firstToken = data.per_token.find(t => t.trait_scores_per_layer);
    if (!firstToken) {
        container.innerHTML = '<div class="no-data">No trait data</div>';
        return;
    }

    const traits = Object.keys(firstToken.trait_scores_per_layer);
    const nLayers = 26;

    // For each trait, find emergence layer (first layer where |score| > 0.5 * max)
    const emergence = traits.map(trait => {
        // Get max absolute score across all tokens and layers
        let maxAbs = 0;
        data.per_token.forEach(t => {
            if (!t.trait_scores_per_layer?.[trait]) return;
            t.trait_scores_per_layer[trait].forEach(score => {
                maxAbs = Math.max(maxAbs, Math.abs(score));
            });
        });

        if (maxAbs === 0) return { trait, layer: nLayers - 1 };

        const threshold = 0.5 * maxAbs;

        // Find first layer where average |score| exceeds threshold
        for (let layer = 0; layer < nLayers; layer++) {
            let sum = 0, count = 0;
            data.per_token.forEach(t => {
                if (t.trait_scores_per_layer?.[trait]?.[layer] !== undefined) {
                    sum += Math.abs(t.trait_scores_per_layer[trait][layer]);
                    count++;
                }
            });
            if (count > 0 && (sum / count) > threshold) {
                return { trait, layer };
            }
        }
        return { trait, layer: nLayers - 1 };
    });

    // Sort by emergence layer
    emergence.sort((a, b) => a.layer - b.layer);

    const trace = {
        x: emergence.map(e => e.layer),
        y: emergence.map(e => e.trait),
        type: 'bar',
        orientation: 'h',
        marker: { color: '#3498db' },
        hovertemplate: '%{y}: Layer %{x}<extra></extra>'
    };

    const layout = {
        margin: { l: 100, r: 20, t: 10, b: 40 },
        height: 250,
        xaxis: { title: 'Emergence Layer', range: [0, 26], dtick: 5 },
        yaxis: { tickfont: { size: 10 } },
        shapes: [
            // "Stable computation" region markers
            { type: 'line', x0: 8, x1: 8, y0: -0.5, y1: traits.length - 0.5, line: { color: 'green', width: 1, dash: 'dash' } },
            { type: 'line', x0: 19, x1: 19, y0: -0.5, y1: traits.length - 0.5, line: { color: 'red', width: 1, dash: 'dash' } }
        ]
    };

    Plotly.newPlot(container, [trace], layout, { responsive: true });
}

function renderDynamicsCorrelation(data) {
    const container = document.getElementById('dynamics-correlation-container');
    if (!container) return;

    const firstToken = data.per_token.find(t => t.trait_scores_per_layer);
    if (!firstToken) {
        container.innerHTML = '<div class="no-data">No trait data</div>';
        return;
    }

    const traits = Object.keys(firstToken.trait_scores_per_layer);

    // For each trait, compute correlation between normalized velocity and |trait velocity|
    const correlations = traits.map(trait => {
        const velocities = [];
        const traitVelocities = [];

        data.per_token.forEach(t => {
            if (!t.normalized_velocity_per_layer || !t.trait_scores_per_layer?.[trait]) return;

            const traitScores = t.trait_scores_per_layer[trait];
            // Trait velocity = diff of trait scores across layers
            for (let i = 0; i < traitScores.length - 1; i++) {
                velocities.push(t.normalized_velocity_per_layer[i] || 0);
                traitVelocities.push(Math.abs(traitScores[i + 1] - traitScores[i]));
            }
        });

        if (velocities.length < 2) return { trait, corr: 0 };

        // Pearson correlation
        const n = velocities.length;
        const sumX = velocities.reduce((a, b) => a + b, 0);
        const sumY = traitVelocities.reduce((a, b) => a + b, 0);
        const sumXY = velocities.reduce((sum, x, i) => sum + x * traitVelocities[i], 0);
        const sumX2 = velocities.reduce((sum, x) => sum + x * x, 0);
        const sumY2 = traitVelocities.reduce((sum, y) => sum + y * y, 0);

        const num = n * sumXY - sumX * sumY;
        const den = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
        const corr = den === 0 ? 0 : num / den;

        return { trait, corr };
    });

    // Sort by correlation (descending)
    correlations.sort((a, b) => b.corr - a.corr);

    const trace = {
        x: correlations.map(c => c.corr),
        y: correlations.map(c => c.trait),
        type: 'bar',
        orientation: 'h',
        marker: {
            color: correlations.map(c => c.corr > 0.3 ? '#27ae60' : c.corr > 0.1 ? '#f39c12' : '#95a5a6')
        },
        hovertemplate: '%{y}: r = %{x:.3f}<extra></extra>'
    };

    const layout = {
        margin: { l: 100, r: 20, t: 10, b: 40 },
        height: 250,
        xaxis: { title: 'Correlation (r)', range: [-0.2, 1] },
        yaxis: { tickfont: { size: 10 } }
    };

    Plotly.newPlot(container, [trace], layout, { responsive: true });
}

// =============================================================================
// REFERENCE SECTION
// =============================================================================

function getCategoryReference() {
    return `
        <div class="category-reference">
            <h3>Reference</h3>

            <details>
                <summary>Trait Scores Heatmap</summary>
                <p>Shows trait activation for every token at layer 16.</p>
                <p><strong>Math:</strong> score = hidden[token, L16] · normalized_trait_vector</p>
                <p><strong>Read:</strong> Red = positive (expresses trait), Blue = negative. Yellow box = selected token.</p>
            </details>

            <details>
                <summary>Velocity Heatmap</summary>
                <p>How fast each token's representation changes at each layer transition.</p>
                <p><strong>Math:</strong> velocity[L] = ||hidden[L+1] - hidden[L]|| / ||hidden[L]||</p>
                <p><strong>Read:</strong> Bright = major transformation. Typical: high (L0-6) → low (L7-22) → high (L23-24).</p>
            </details>

            <details>
                <summary>Trait Scores Bar</summary>
                <p>Selected token's trait activations at layer 16, sorted by strength.</p>
                <p><strong>Read:</strong> Teal = positive, Red = negative. Longest bars = strongest traits.</p>
            </details>

            <details>
                <summary>Attention Pattern</summary>
                <p>Where the selected token "looks" in the context (layer 16).</p>
                <p><strong>Math:</strong> Average attention weights across all heads.</p>
                <p><strong>Read:</strong> Tall bars = positions this token attends to strongly.</p>
            </details>

            <details>
                <summary>Trait Emergence</summary>
                <p>Which layer each trait first becomes significant.</p>
                <p><strong>Math:</strong> First layer where avg|score| > 0.5 × max|score|.</p>
                <p><strong>Read:</strong> Shorter bars = emerges earlier. Green line = L8, Red = L19 (stable region).</p>
            </details>

            <details>
                <summary>Trait-Dynamics Correlation</summary>
                <p>Do trait changes happen when the model is most "active"?</p>
                <p><strong>Math:</strong> Pearson correlation between normalized velocity and |trait velocity|.</p>
                <p><strong>Read:</strong> Green = high (trait tied to computation). Gray = low (independent).</p>
            </details>
        </div>
    `;
}

// =============================================================================
// UTILITIES
// =============================================================================

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
