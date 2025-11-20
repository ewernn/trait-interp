// Prompt Activation View

async function renderPromptActivation() {
    const contentArea = document.getElementById('content-area');
    const filteredTraits = window.getFilteredTraits();

    if (filteredTraits.length === 0) {
        contentArea.innerHTML = `
            <div class="card">
                <div class="card-title">Per-Token Activation</div>
                <div class="info">Select at least one trait to view per-token activation trajectories</div>
            </div>
        `;
        return;
    }

    // Load data for ALL selected traits
    contentArea.innerHTML = '<div id="all-prompt-traits-container"></div>';
    const container = document.getElementById('all-prompt-traits-container');

    for (const trait of filteredTraits) {
        const tier2Dir = `../experiments/${window.state.experimentData.name}/${trait.name}/inference/residual_stream_activations/`;

        // Create a unique div for this trait
        const traitDiv = document.createElement('div');
        traitDiv.id = `prompt-trait-${trait.name}`;
        traitDiv.style.marginBottom = '0px';
        traitDiv.style.paddingBottom = '12px';
        traitDiv.style.borderBottom = '1px solid var(--border-color)';
        container.appendChild(traitDiv);

        // Try to load the selected prompt
        try {
            const fetchPath = `${tier2Dir}prompt_${window.state.currentPrompt}.json`;
            console.log(`[${trait.name}] Fetching prompt activation data for prompt ${window.state.currentPrompt}`);
            const response = await fetch(fetchPath);

            if (!response.ok) {
                console.log(`[${trait.name}] No data available for prompt ${window.state.currentPrompt} (${response.status})`);
                renderPromptActivationInstructions(traitDiv.id, trait, window.state.currentPrompt);
                continue;
            }

            const data = await response.json();
            console.log(`[${trait.name}] Data loaded successfully for prompt ${window.state.currentPrompt}`);
            renderPromptActivationData(traitDiv.id, trait, data);
        } catch (error) {
            console.log(`[${trait.name}] Load failed for prompt ${window.state.currentPrompt}:`, error.message);
            renderPromptActivationInstructions(traitDiv.id, trait, window.state.currentPrompt);
        }
    }
}

function renderPromptActivationInstructions(containerId, trait, promptNum) {
    const container = document.getElementById(containerId);
    const nLayers = trait.metadata?.n_layers || 26;

    container.innerHTML = `
        <div class="card" style="margin-bottom: 10px;">
            <div class="card-title">${window.getDisplayName(trait.name)}</div>

            <div class="info" style="margin-bottom: 10px; font-size: 12px;">
                ⚠️ No data for prompt ${promptNum}
            </div>

            <div style="background: var(--bg-tertiary); padding: 10px; border-radius: 6px; font-size: 12px;">
                <p style="color: var(--text-secondary); margin-bottom: 8px;">
                    The file <code>prompt_${promptNum}.json</code> does not exist for this trait.
                </p>
                <p style="color: var(--text-secondary); margin-bottom: 8px;">
                    To capture per-token activation trajectory (prompt + response):
                </p>
                <pre style="background: var(--bg-primary); color: var(--text-primary); padding: 8px; border-radius: 4px; margin: 8px 0; overflow-x: auto; font-size: 11px;">python inference/capture_tier2.py --experiment ${window.state.experimentData.name} --trait ${trait.name} --prompts "..." --save-json</pre>
            </div>
        </div>
    `;
}

function renderPromptActivationData(containerId, trait, data) {
    const container = document.getElementById(containerId);

    try {
        const promptProj = data.projections.prompt;  // [n_tokens, n_layers, 3]
        const responseProj = data.projections.response;  // [n_tokens, n_layers, 3]

        // Combine prompt and response
        const allTokens = [...data.prompt.tokens, ...data.response.tokens];
        const allProj = [...promptProj, ...responseProj];
        const nPromptTokens = data.prompt.n_tokens;
        const nTotalTokens = allTokens.length;
        const nLayers = promptProj[0].length;

        console.log(`Combined tokens: ${nTotalTokens} (${nPromptTokens} prompt + ${nTotalTokens - nPromptTokens} response), Layers: ${nLayers}`);

        // Create unique IDs for this trait's elements
        const traitId = trait.name;
        const plotId = `prompt-activation-plot-${traitId}`;

        container.innerHTML = `
            <div class="card" style="margin-bottom: 0px; padding: 8px 12px;">
                <div class="card-title" style="margin-bottom: 4px; font-size: 13px;">${window.getDisplayName(trait.name)}</div>

                <!-- Conversation context -->
                <div style="background: var(--bg-tertiary); padding: 4px 6px; border-radius: 3px; margin-bottom: 6px;">
                    <div style="color: var(--text-primary); font-size: 11px; margin-bottom: 1px;">${data.prompt.text} ${data.response.text}</div>
                    <div style="color: var(--text-secondary); font-size: 9px;">
                        ${nPromptTokens} prompt + ${nTotalTokens - nPromptTokens} response = ${nTotalTokens} tokens
                    </div>
                </div>

                <div id="${plotId}" style="margin-bottom: 0px;"></div>
            </div>
        `;

        // Render the combined activation plot
        renderPromptActivationPlot(plotId, allProj, allTokens, nLayers, nPromptTokens);

    } catch (error) {
        console.error('Error rendering prompt activation data:', error);
        container.innerHTML = `<div class="card"><div class="card-title">Error</div><div class="info">Failed to render prompt activation data: ${error.message}</div></div>`;
    }
}

function renderPromptActivationPlot(divId, projections, tokens, nLayers, nPromptTokens) {
    // projections: [n_tokens, n_layers, 3_sublayers]
    // We want: x-axis = token position, y-axis = activation strength
    // We'll average across all layers and sublayers for each token

    const nTokens = projections.length;
    console.log(`Rendering combined activation plot for ${nTokens} tokens (${nPromptTokens} prompt + ${nTokens - nPromptTokens} response) x ${nLayers} layers`);

    // Skip BOS token (index 0) for better visualization
    const startIdx = 1;

    // Calculate activation strength for each token (average across all layers and sublayers)
    const activations = [];
    const displayTokens = [];

    for (let t = startIdx; t < nTokens; t++) {
        let sum = 0;
        let count = 0;
        for (let l = 0; l < nLayers; l++) {
            for (let s = 0; s < 3; s++) {
                sum += projections[t][l][s];
                count++;
            }
        }
        activations.push(sum / count);
        displayTokens.push(tokens[t]);
    }

    // Create the line plot
    const trace = {
        x: Array.from({length: activations.length}, (_, i) => i),
        y: activations,
        type: 'scatter',
        mode: 'lines+markers',
        line: {
            color: '#4a9eff',
            width: 1.5
        },
        marker: {
            size: 4,
            color: '#4a9eff'
        },
        text: displayTokens,
        hovertemplate: 'Token %{x}: %{text}<br>Activation: %{y:.2f}<extra></extra>'
    };

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
                color: 'rgba(128, 128, 128, 0.15)',
                width: 1,
                dash: 'dot'
            }
        }
    ];

    // Add annotations for prompt/response regions
    const annotations = [
        {
            x: (nPromptTokens - startIdx) / 2 - 0.5,
            y: 1.05,
            yref: 'paper',
            text: 'Prompt',
            showarrow: false,
            font: {
                size: 10,
                color: 'rgba(128, 128, 128, 0.7)'
            }
        },
        {
            x: (nPromptTokens - startIdx) + (activations.length - (nPromptTokens - startIdx)) / 2 - 0.5,
            y: 1.05,
            yref: 'paper',
            text: 'Response',
            showarrow: false,
            font: {
                size: 10,
                color: 'rgba(128, 128, 128, 0.7)'
            }
        }
    ];

    const layout = {
        xaxis: {
            title: 'Token Position',
            tickmode: 'array',
            tickvals: Array.from({length: activations.length}, (_, i) => i),
            ticktext: displayTokens,
            tickangle: -45,
            showgrid: false
        },
        yaxis: {
            title: 'Average Activation (all layers)',
            zeroline: true,
            zerolinecolor: 'rgba(128, 128, 128, 0.3)',
            zerolinewidth: 1,
            showgrid: false
        },
        shapes: shapes,
        annotations: annotations,
        margin: { l: 50, r: 10, t: 25, b: 80 },
        height: 200,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: {
            size: 10,
            color: getComputedStyle(document.documentElement).getPropertyValue('--text-primary')
        },
        hovermode: 'closest'
    };

    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['lasso2d', 'select2d']
    };

    Plotly.newPlot(divId, [trace], layout, config);
}

// Render Layer Deep Dive view
// Export to global scope
window.renderPromptActivation = renderPromptActivation;
