// Trait Dynamics View - Watch the model's internal state evolve token-by-token
// Core insight: "See how the model is thinking" by projecting onto trait vectors

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

async function renderPerTokenActivation() {
    const contentArea = document.getElementById('content-area');
    const filteredTraits = window.getFilteredTraits();

    if (filteredTraits.length === 0) {
        // Show education sections even without data
        contentArea.innerHTML = `
            <div class="extraction-overview-container">
                ${renderEducationSections()}
                <div class="extraction-section">
                    <h2 class="section-heading">Trait Trajectory</h2>
                    <div class="card">
                        <div class="info">Select at least one trait from the sidebar to view activation trajectories.</div>
                    </div>
                </div>
            </div>
        `;
        if (window.MathJax) MathJax.typesetPromise();
        return;
    }

    // Show loading state
    contentArea.innerHTML = `
        <div class="extraction-overview-container">
            ${renderEducationSections()}
            <div class="extraction-section">
                <h2 class="section-heading">Trait Trajectory</h2>
                <div class="card">
                    <div class="info">Loading data for ${filteredTraits.length} trait(s)...</div>
                </div>
            </div>
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
        <div class="extraction-section">
            <h2 class="section-heading">What is Trait Projection?</h2>
            <div class="techniques-grid" style="grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));">
                <div class="technique-card">
                    <h4>The Core Idea</h4>
                    <p class="technique-desc">Each token's hidden state lives in a high-dimensional space (2304 dims for Gemma 2B). We project onto trait vectors to measure "how much of this trait is present."</p>
                    <div class="technique-math">$$\\text{score} = \\frac{\\vec{h} \\cdot \\vec{v}}{||\\vec{v}||}$$</div>
                    <p class="technique-use"><strong>Where:</strong> \\(\\vec{h}\\) = hidden state, \\(\\vec{v}\\) = trait vector</p>
                </div>

                <div class="technique-card">
                    <h4>Interpretation</h4>
                    <p class="technique-desc">The projection score tells you how aligned the model's internal state is with a particular trait direction.</p>
                    <ul style="margin: 8px 0; padding-left: 20px; font-size: 12px;">
                        <li><strong>Positive:</strong> Model expressing trait</li>
                        <li><strong>Negative:</strong> Model avoiding trait</li>
                        <li><strong>Near zero:</strong> Neutral/unrelated</li>
                        <li><strong>Magnitude:</strong> Strength of expression</li>
                    </ul>
                </div>

                <div class="technique-card">
                    <h4>Why This Works</h4>
                    <p class="technique-desc">Trait vectors are extracted by finding directions that separate positive/negative examples during training. These directions capture semantic meaning.</p>
                    <p class="technique-use"><strong>Validation:</strong> Vectors achieve 90%+ classification accuracy on held-out examples.</p>
                </div>
            </div>
        </div>

        <!-- Section 2: How to Read the Graph -->
        <div class="extraction-section">
            <h2 class="section-heading">Reading the Graph</h2>
            <div class="techniques-grid" style="grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));">
                <div class="technique-card">
                    <h4>X-Axis: Tokens</h4>
                    <p class="technique-desc">Each point is one token in the sequence. The dashed line separates prompt (left) from response (right).</p>
                </div>

                <div class="technique-card">
                    <h4>Y-Axis: Activation</h4>
                    <p class="technique-desc">Average projection score across all layers. Higher = stronger trait expression for that token.</p>
                </div>

                <div class="technique-card">
                    <h4>Colors: Traits</h4>
                    <p class="technique-desc">Each line is a different trait. Hover over lines to highlight. Click legend to toggle.</p>
                </div>

                <div class="technique-card">
                    <h4>Vertical Line</h4>
                    <p class="technique-desc">Shows the currently selected token (from the global slider). Syncs across all views.</p>
                </div>
            </div>
        </div>

        <!-- Section 3: Patterns to Look For -->
        <div class="extraction-section">
            <h2 class="section-heading">Key Patterns</h2>
            <div class="techniques-grid" style="grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));">
                <div class="technique-card">
                    <h4>Commitment Points</h4>
                    <p class="technique-desc">Sharp rises or falls indicate "decision moments" where the model commits to a behavior. Often occurs early in response.</p>
                    <p class="technique-use"><strong>Example:</strong> Refusal spike at "I cannot" tokens</p>
                </div>

                <div class="technique-card">
                    <h4>Trait Crossings</h4>
                    <p class="technique-desc">When two trait lines cross, the model is transitioning from one mode to another. Watch for correlated inversions.</p>
                    <p class="technique-use"><strong>Example:</strong> Uncertainty dropping as confidence rises</p>
                </div>

                <div class="technique-card">
                    <h4>Persistence</h4>
                    <p class="technique-desc">Traits that stay elevated across many tokens indicate sustained behavioral modes (not just local patterns).</p>
                    <p class="technique-use"><strong>Example:</strong> Sustained sycophancy throughout a response</p>
                </div>

                <div class="technique-card">
                    <h4>Prompt vs Response</h4>
                    <p class="technique-desc">Compare trait levels in prompt (context processing) vs response (generation). Dramatic shifts reveal how context influences behavior.</p>
                    <p class="technique-use"><strong>Example:</strong> Low refusal during prompt, high during harmful response</p>
                </div>
            </div>
        </div>
    `;
}

function renderNoDataMessage(container, traits, promptSet, promptId) {
    const promptLabel = promptSet && promptId ? `${promptSet}/${promptId}` : 'none selected';
    container.innerHTML = `
        <div class="extraction-overview-container">
            ${renderEducationSections()}
            <div class="extraction-section">
                <h2 class="section-heading">Trait Trajectory</h2>
                <div class="card">
                    <div class="info" style="margin-bottom: 10px;">
                        No data available for prompt ${promptLabel} for any selected trait.
                    </div>
                    <div style="background: var(--bg-tertiary); padding: 10px; border-radius: 6px; font-size: 12px;">
                        <p style="color: var(--text-secondary); margin-bottom: 8px;">
                            To capture per-token activation data, run:
                        </p>
                        <pre style="background: var(--bg-primary); color: var(--text-primary); padding: 8px; border-radius: 4px; margin: 8px 0; overflow-x: auto; font-size: 11px;">python inference/capture_raw_activations.py --experiment ${window.paths.getExperiment()} --prompt-set ${promptSet || 'PROMPT_SET'}</pre>
                    </div>
                </div>
            </div>
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
            <div style="color: var(--text-secondary); font-size: 11px; margin-bottom: 8px;">
                No data for: ${failedTraits.map(t => window.getDisplayName(t)).join(', ')}
            </div>
        `;
    }

    container.innerHTML = `
        <div class="extraction-overview-container">
            ${renderEducationSections()}
            <div class="extraction-section">
                <h2 class="section-heading">Trait Trajectory</h2>
                <div class="card" style="padding: 12px;">
                    ${failedHtml}
                    <div id="combined-activation-plot" style="width: 100%;"></div>
                </div>
            </div>
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
}

// Export to global scope
window.renderPerTokenActivation = renderPerTokenActivation;
