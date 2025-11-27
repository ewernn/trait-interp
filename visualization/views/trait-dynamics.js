// Trait Dynamics View - Watch the model's internal state evolve token-by-token
// Core insight: "See how the model is thinking" by projecting onto trait vectors
//
// Two complementary views:
// 1. Token Trajectory: X=tokens, Y=activation (layer-averaged) - how traits evolve during generation
// 2. Layer Evolution: X=layers, Y=projection (token-averaged) - how traits emerge through the network

// Color palette for traits (distinct, colorblind-friendly)
const TRAIT_COLORS = [
    '#4a9eff',  // blue
    '#ff6b6b',  // red
    '#51cf66',  // green
    '#ffd43b',  // yellow
    '#cc5de8',  // purple
    '#ff922b',  // orange
    '#20c997',  // teal
    '#f06595',  // pink
    '#748ffc',  // indigo
    '#a9e34b',  // lime
];

// Derivative overlay colors (position, velocity, acceleration)
const DERIVATIVE_COLORS = {
    position: '#2E86AB',     // blue
    velocity: '#A23B72',     // magenta
    acceleration: '#F18F01'  // orange
};

/**
 * Compute layer-averaged position, velocity, and acceleration for a trait.
 * Position = average projection across all tokens per layer
 * Velocity = first derivative (diff between layers)
 * Acceleration = second derivative (diff of velocity)
 *
 * @param {Object} data - Trait projection data with projections.prompt and projections.response
 * @returns {Object} { position: [26], velocity: [25], acceleration: [24] }
 */
function computeLayerDerivatives(data) {
    const promptProj = data.projections.prompt;
    const responseProj = data.projections.response;
    const allProj = [...promptProj, ...responseProj];
    const nLayers = promptProj[0].length;
    const nSublayers = promptProj[0][0].length;

    // Compute layer-averaged position (average across all tokens and sublayers)
    const position = [];
    for (let layer = 0; layer < nLayers; layer++) {
        let sum = 0;
        let count = 0;
        for (let token = 0; token < allProj.length; token++) {
            for (let sublayer = 0; sublayer < nSublayers; sublayer++) {
                sum += allProj[token][layer][sublayer];
                count++;
            }
        }
        position.push(sum / count);
    }

    // Compute velocity (first derivative)
    const velocity = [];
    for (let i = 0; i < position.length - 1; i++) {
        velocity.push(position[i + 1] - position[i]);
    }

    // Compute acceleration (second derivative)
    const acceleration = [];
    for (let i = 0; i < velocity.length - 1; i++) {
        acceleration.push(velocity[i + 1] - velocity[i]);
    }

    return { position, velocity, acceleration };
}

async function renderTraitDynamics() {
    const contentArea = document.getElementById('content-area');
    const filteredTraits = window.getFilteredTraits();

    if (filteredTraits.length === 0) {
        // Show education sections even without data
        contentArea.innerHTML = `
            <div class="tool-view">
                ${renderEducationSections()}
                <section>
                    <h2>Token Trajectory</h2>
                    <div class="info">Select at least one trait from the sidebar to view activation trajectories.</div>
                </section>
            </div>
        `;
        if (window.MathJax) MathJax.typesetPromise();
        return;
    }

    // Show loading state
    contentArea.innerHTML = `
        <div class="tool-view">
            ${renderEducationSections()}
            <section>
                <h2>Token Trajectory</h2>
                <div class="info">Loading data for ${filteredTraits.length} trait(s)...</div>
            </section>
        </div>
    `;
    if (window.MathJax) MathJax.typesetPromise();

    const traitData = {};
    const failedTraits = [];
    const promptSet = window.state.currentPromptSet;
    const promptId = window.state.currentPromptId;

    if (!promptSet || !promptId) {
        renderNoDataMessage(contentArea, filteredTraits, promptSet, promptId);
        return;
    }

    // Load data for ALL selected traits
    for (const trait of filteredTraits) {
        try {
            const fetchPath = window.paths.residualStreamData(trait, promptSet, promptId);
            console.log(`[${trait.name}] Fetching prompt activation data for ${promptSet}/${promptId}`);
            const response = await fetch(fetchPath);

            if (!response.ok) {
                console.log(`[${trait.name}] No data available for ${promptSet}/${promptId} (${response.status})`);
                failedTraits.push(trait.name);
                continue;
            }

            const data = await response.json();
            console.log(`[${trait.name}] Data loaded successfully for ${promptSet}/${promptId}`);
            traitData[trait.name] = data;
        } catch (error) {
            console.log(`[${trait.name}] Load failed for ${promptSet}/${promptId}:`, error.message);
            failedTraits.push(trait.name);
        }
    }

    // Check if we have any data
    const loadedTraits = Object.keys(traitData);
    if (loadedTraits.length === 0) {
        renderNoDataMessage(contentArea, filteredTraits, promptSet, promptId);
        return;
    }

    // Render education + combined graph
    renderCombinedGraph(contentArea, traitData, loadedTraits, failedTraits, promptSet, promptId);
}


function renderEducationSections() {
    return `
        <!-- Section 1: What is Projection? -->
        <section>
            <h2>What is Trait Projection?</h2>
            <div class="grid">
                <div class="card">
                    <h4>The Core Idea</h4>
                    <p>Each token's hidden state lives in a high-dimensional space (2304 dims for Gemma 2B). We project onto trait vectors to measure "how much of this trait is present."</p>
                    <p>$$\\text{score} = \\frac{\\vec{h} \\cdot \\vec{v}}{||\\vec{v}||}$$</p>
                    <p><strong>Where:</strong> \\(\\vec{h}\\) = hidden state, \\(\\vec{v}\\) = trait vector</p>
                </div>

                <div class="card">
                    <h4>Interpretation</h4>
                    <p>The projection score tells you how aligned the model's internal state is with a particular trait direction.</p>
                    <ul>
                        <li><strong>Positive:</strong> Model expressing trait</li>
                        <li><strong>Negative:</strong> Model avoiding trait</li>
                        <li><strong>Near zero:</strong> Neutral/unrelated</li>
                        <li><strong>Magnitude:</strong> Strength of expression</li>
                    </ul>
                </div>

                <div class="card">
                    <h4>Why This Works</h4>
                    <p>Trait vectors are extracted by finding directions that separate positive/negative examples during training. These directions capture semantic meaning.</p>
                    <p><strong>Validation:</strong> Vectors achieve 90%+ classification accuracy on held-out examples.</p>
                </div>
            </div>
        </section>

        <!-- Section 2: How to Read the Graphs -->
        <section>
            <h2>Reading the Graphs</h2>
            <div class="grid">
                <div class="card">
                    <h4>Token Trajectory</h4>
                    <p>X = tokens, Y = activation (layer-averaged). Shows how traits evolve during generation. Dashed line separates prompt/response.</p>
                </div>

                <div class="card">
                    <h4>Layer Evolution</h4>
                    <p>X = layers, Y = projection (token-averaged). Shows how traits emerge through the network with position/velocity/acceleration.</p>
                </div>

                <div class="card">
                    <h4>Colors & Lines</h4>
                    <p>Each color = one trait. Solid = position, dashed = velocity, dotted = acceleration. Hover to highlight a trait.</p>
                </div>

                <div class="card">
                    <h4>Vertical Line</h4>
                    <p>Shows the currently selected token (from the global slider). Syncs across all views.</p>
                </div>
            </div>
        </section>

        <!-- Section 3: Patterns to Look For -->
        <section>
            <h2>Key Patterns</h2>
            <div class="grid">
                <div class="card">
                    <h4>Commitment Points</h4>
                    <p>Sharp rises or falls indicate "decision moments" where the model commits to a behavior. Often occurs early in response.</p>
                    <p><strong>Example:</strong> Refusal spike at "I cannot" tokens</p>
                </div>

                <div class="card">
                    <h4>Trait Crossings</h4>
                    <p>When two trait lines cross, the model is transitioning from one mode to another. Watch for correlated inversions.</p>
                    <p><strong>Example:</strong> Uncertainty dropping as confidence rises</p>
                </div>

                <div class="card">
                    <h4>Persistence</h4>
                    <p>Traits that stay elevated across many tokens indicate sustained behavioral modes (not just local patterns).</p>
                    <p><strong>Example:</strong> Sustained sycophancy throughout a response</p>
                </div>

                <div class="card">
                    <h4>Prompt vs Response</h4>
                    <p>Compare trait levels in prompt (context processing) vs response (generation). Dramatic shifts reveal how context influences behavior.</p>
                    <p><strong>Example:</strong> Low refusal during prompt, high during harmful response</p>
                </div>
            </div>
        </section>
    `;
}

function renderNoDataMessage(container, traits, promptSet, promptId) {
    const promptLabel = promptSet && promptId ? `${promptSet}/${promptId}` : 'none selected';
    container.innerHTML = `
        <div class="tool-view">
            ${renderEducationSections()}
            <section>
                <h2>Token Trajectory</h2>
                <div class="info">
                    No data available for prompt ${promptLabel} for any selected trait.
                </div>
                <p class="tool-description">
                    To capture per-token activation data, run:
                </p>
                <pre>python inference/capture_raw_activations.py --experiment ${window.paths.getExperiment()} --prompt-set ${promptSet || 'PROMPT_SET'}</pre>
            </section>
        </div>
    `;
    if (window.MathJax) MathJax.typesetPromise();
}

function renderCombinedGraph(container, traitData, loadedTraits, failedTraits, promptSet, promptId) {
    // Use first trait's data as reference for tokens (they should all be the same)
    const refData = traitData[loadedTraits[0]];
    const promptTokens = refData.prompt.tokens;
    const responseTokens = refData.response.tokens;
    const allTokens = [...promptTokens, ...responseTokens];
    const nPromptTokens = promptTokens.length;  // Use actual array length
    const nTotalTokens = allTokens.length;

    // Build HTML
    let failedHtml = '';
    if (failedTraits.length > 0) {
        failedHtml = `
            <div class="tool-description">
                No data for: ${failedTraits.map(t => window.getDisplayName(t)).join(', ')}
            </div>
        `;
    }

    container.innerHTML = `
        <div class="tool-view">
            ${renderEducationSections()}
            <section>
                <h2>Token Trajectory</h2>
                <p class="tool-description">
                    How traits evolve as the model generates each token (layer-averaged)
                </p>
                ${failedHtml}
                <div id="combined-activation-plot"></div>
            </section>
            <section>
                <h2>Layer Evolution</h2>
                <p class="tool-description">
                    How traits emerge through network layers (token-averaged). Shows position, velocity (1st derivative), and acceleration (2nd derivative).
                </p>
                <div id="layer-evolution-plot"></div>
            </section>
        </div>
    `;

    // Render MathJax
    if (window.MathJax) MathJax.typesetPromise();

    // Prepare traces for each trait
    const traces = [];
    const startIdx = 1;  // Skip BOS token

    loadedTraits.forEach((traitName, idx) => {
        const data = traitData[traitName];
        const promptProj = data.projections.prompt;
        const responseProj = data.projections.response;
        const allProj = [...promptProj, ...responseProj];
        const nLayers = promptProj[0].length;

        // Calculate activation strength for each token (average across all layers and sublayers)
        const activations = [];
        const displayTokens = [];

        for (let t = startIdx; t < allProj.length; t++) {
            let sum = 0;
            let count = 0;
            for (let l = 0; l < nLayers; l++) {
                for (let s = 0; s < 3; s++) {
                    sum += allProj[t][l][s];
                    count++;
                }
            }
            activations.push(sum / count);
            displayTokens.push(allTokens[t]);
        }

        const color = TRAIT_COLORS[idx % TRAIT_COLORS.length];

        traces.push({
            x: Array.from({length: activations.length}, (_, i) => i),
            y: activations,
            type: 'scatter',
            mode: 'lines+markers',
            name: window.getDisplayName(traitName),
            line: {
                color: color,
                width: 1
            },
            marker: {
                size: 1,
                color: color
            },
            text: displayTokens,
            hovertemplate: `<b>${window.getDisplayName(traitName)}</b><br>Token %{x}: %{text}<br>Activation: %{y:.3f}<extra></extra>`
        });
    });

    // Get display tokens from first trait for x-axis labels
    const displayTokens = [];
    for (let t = startIdx; t < allTokens.length; t++) {
        displayTokens.push(allTokens[t]);
    }

    // Get colors from CSS variables
    const textSecondary = window.getCssVar('--text-secondary', '#a4a4a4');
    const bgTertiary = window.getCssVar('--bg-tertiary', '#3a3a3a');
    const primaryColor = window.getCssVar('--primary-color', '#a09f6c');

    // Get current token index from global state (absolute index across prompt+response)
    // The graph skips BOS (startIdx=1), so token at absolute index N = x position (N - startIdx)
    const currentTokenIdx = window.state.currentTokenIndex || 0;
    const highlightX = currentTokenIdx - startIdx;

    // Add subtle vertical line separator between prompt and response
    const shapes = [
        {
            type: 'line',
            x0: (nPromptTokens - startIdx) - 0.5,
            x1: (nPromptTokens - startIdx) - 0.5,
            y0: 0,
            y1: 1,
            yref: 'paper',
            line: {
                color: textSecondary,
                width: 2,
                dash: 'dash'
            }
        },
        // Current token highlight line from global slider
        {
            type: 'line',
            x0: highlightX,
            x1: highlightX,
            y0: 0,
            y1: 1,
            yref: 'paper',
            line: {
                color: primaryColor,
                width: 2
            }
        }
    ];

    // Add annotations for prompt/response regions
    const annotations = [
        {
            x: (nPromptTokens - startIdx) / 2 - 0.5,
            y: 1.08,
            yref: 'paper',
            text: 'PROMPT',
            showarrow: false,
            font: {
                size: 11,
                color: textSecondary
            }
        },
        {
            x: (nPromptTokens - startIdx) + (displayTokens.length - (nPromptTokens - startIdx)) / 2 - 0.5,
            y: 1.08,
            yref: 'paper',
            text: 'RESPONSE',
            showarrow: false,
            font: {
                size: 11,
                color: textSecondary
            }
        }
    ];

    const layout = window.getPlotlyLayout({
        xaxis: {
            title: 'Token Position',
            tickmode: 'array',
            tickvals: Array.from({length: displayTokens.length}, (_, i) => i),
            ticktext: displayTokens,
            tickangle: -45,
            showgrid: false,
            tickfont: { size: 9 }
        },
        yaxis: {
            title: 'Activation (avg all layers)',
            zeroline: true,
            zerolinewidth: 1,
            showgrid: true
        },
        shapes: shapes,
        annotations: annotations,
        margin: { l: 60, r: 20, t: 40, b: 100 },
        height: 500,
        font: { size: 11 },
        hovermode: 'closest',
        legend: {
            orientation: 'v',
            yanchor: 'top',
            y: 1,
            xanchor: 'left',
            x: 1.02,
            font: { size: 11 },
            bgcolor: 'transparent'
        },
        showlegend: true
    });

    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['lasso2d', 'select2d']
    };

    Plotly.newPlot('combined-activation-plot', traces, layout, config);

    // Hover-to-highlight: dim other traces when hovering
    const plotDiv = document.getElementById('combined-activation-plot');
    plotDiv.on('plotly_hover', (d) =>
        Plotly.restyle(plotDiv, {'opacity': traces.map((_, i) => i === d.points[0].curveNumber ? 1.0 : 0.2)})
    );
    plotDiv.on('plotly_unhover', () => Plotly.restyle(plotDiv, {'opacity': 1.0}));

    // Render Layer Evolution plot (derivative overlay)
    renderLayerEvolutionPlot(traitData, loadedTraits);
}


/**
 * Render the Layer Evolution plot showing position, velocity, and acceleration
 * across layers for each selected trait.
 */
function renderLayerEvolutionPlot(traitData, loadedTraits) {
    const nLayers = 26;
    const layerIndices = Array.from({length: nLayers}, (_, i) => i);
    const velocityIndices = Array.from({length: nLayers - 1}, (_, i) => i + 0.5);  // Midpoints
    const accelIndices = Array.from({length: nLayers - 2}, (_, i) => i + 1);  // Midpoints

    const traces = [];
    const textSecondary = window.getCssVar('--text-secondary', '#a4a4a4');

    // For each trait, compute derivatives and add traces
    loadedTraits.forEach((traitName, idx) => {
        const data = traitData[traitName];
        const derivatives = computeLayerDerivatives(data);
        const traitColor = TRAIT_COLORS[idx % TRAIT_COLORS.length];
        const displayName = window.getDisplayName(traitName);

        // Position trace (solid line)
        traces.push({
            x: layerIndices,
            y: derivatives.position,
            type: 'scatter',
            mode: 'lines+markers',
            name: `${displayName} (pos)`,
            line: { color: traitColor, width: 2 },
            marker: { size: 4, symbol: 'circle' },
            legendgroup: traitName,
            hovertemplate: `<b>${displayName}</b><br>Layer %{x}<br>Position: %{y:.2f}<extra></extra>`
        });

        // Velocity trace (dashed line)
        traces.push({
            x: velocityIndices,
            y: derivatives.velocity,
            type: 'scatter',
            mode: 'lines+markers',
            name: `${displayName} (vel)`,
            line: { color: traitColor, width: 1.5, dash: 'dash' },
            marker: { size: 3, symbol: 'square' },
            legendgroup: traitName,
            showlegend: false,
            hovertemplate: `<b>${displayName}</b><br>Layer %{x:.1f}<br>Velocity: %{y:.2f}<extra></extra>`
        });

        // Acceleration trace (dotted line)
        traces.push({
            x: accelIndices,
            y: derivatives.acceleration,
            type: 'scatter',
            mode: 'lines+markers',
            name: `${displayName} (acc)`,
            line: { color: traitColor, width: 1, dash: 'dot' },
            marker: { size: 2, symbol: 'diamond' },
            legendgroup: traitName,
            showlegend: false,
            hovertemplate: `<b>${displayName}</b><br>Layer %{x:.1f}<br>Acceleration: %{y:.2f}<extra></extra>`
        });
    });

    // Add derivative type legend entries (shown once)
    traces.push({
        x: [null], y: [null],
        type: 'scatter',
        mode: 'lines',
        name: '── Position',
        line: { color: textSecondary, width: 2 },
        legendgroup: 'legend-position',
        hoverinfo: 'skip'
    });
    traces.push({
        x: [null], y: [null],
        type: 'scatter',
        mode: 'lines',
        name: '╌╌ Velocity',
        line: { color: textSecondary, width: 1.5, dash: 'dash' },
        legendgroup: 'legend-velocity',
        hoverinfo: 'skip'
    });
    traces.push({
        x: [null], y: [null],
        type: 'scatter',
        mode: 'lines',
        name: '··· Acceleration',
        line: { color: textSecondary, width: 1, dash: 'dot' },
        legendgroup: 'legend-accel',
        hoverinfo: 'skip'
    });

    const layout = window.getPlotlyLayout({
        xaxis: {
            title: 'Layer',
            tickmode: 'linear',
            tick0: 0,
            dtick: 2,
            showgrid: true,
            gridcolor: 'rgba(128,128,128,0.2)'
        },
        yaxis: {
            title: 'Value',
            zeroline: true,
            zerolinewidth: 1,
            zerolinecolor: textSecondary,
            showgrid: true
        },
        margin: { l: 60, r: 20, t: 20, b: 50 },
        height: 400,
        font: { size: 11 },
        hovermode: 'closest',
        legend: {
            orientation: 'v',
            yanchor: 'top',
            y: 1,
            xanchor: 'left',
            x: 1.02,
            font: { size: 10 },
            bgcolor: 'transparent'
        },
        showlegend: true
    });

    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['lasso2d', 'select2d']
    };

    Plotly.newPlot('layer-evolution-plot', traces, layout, config);

    // Hover-to-highlight by legendgroup
    const layerPlotDiv = document.getElementById('layer-evolution-plot');
    layerPlotDiv.on('plotly_hover', (d) => {
        const hoveredGroup = d.points[0].data.legendgroup;
        if (hoveredGroup && !hoveredGroup.startsWith('legend-')) {
            const opacities = traces.map(t =>
                t.legendgroup === hoveredGroup || t.legendgroup?.startsWith('legend-') ? 1.0 : 0.15
            );
            Plotly.restyle(layerPlotDiv, {'opacity': opacities});
        }
    });
    layerPlotDiv.on('plotly_unhover', () => Plotly.restyle(layerPlotDiv, {'opacity': 1.0}));
}

// Export to global scope
window.renderTraitDynamics = renderTraitDynamics;
