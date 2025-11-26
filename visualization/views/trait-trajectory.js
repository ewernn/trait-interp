// Monitoring View

async function renderTraitTrajectory() {
    const contentArea = document.getElementById('content-area');
    const filteredTraits = window.getFilteredTraits();

    if (filteredTraits.length === 0) {
        contentArea.innerHTML = `
            <div class="card">
                <div class="card-title">All Layers</div>
                <div class="info">Select at least one trait to view trajectories</div>
            </div>
        `;
        return;
    }

    // Load data for ALL selected traits
    contentArea.innerHTML = '<div id="all-traits-container"></div>';
    const container = document.getElementById('all-traits-container');

    // Use global PathBuilder singleton with experiment set
    window.paths.setExperiment(window.state.experimentData.name);

    for (const trait of filteredTraits) {
        // Create a unique div for this trait
        // Sanitize trait name - replace slashes with dashes for valid HTML IDs
        const sanitizedName = trait.name.replace(/\//g, '-');
        const traitDiv = document.createElement('div');
        traitDiv.id = `trait-${sanitizedName}`;
        traitDiv.style.marginBottom = '20px';
        container.appendChild(traitDiv);

        // Try to load the selected prompt using global PathBuilder
        const promptSet = window.state.currentPromptSet;
        const promptId = window.state.currentPromptId;

        if (!promptSet || !promptId) {
            renderAllLayersInstructionsInContainer(traitDiv.id, trait, promptSet, promptId);
            continue;
        }

        try {
            const fetchPath = window.paths.residualStreamData(trait, promptSet, promptId);
            console.log(`[${trait.name}] Fetching trajectory data for ${promptSet}/${promptId}`);
            const response = await fetch(fetchPath);

            if (!response.ok) {
                console.log(`[${trait.name}] No data available for ${promptSet}/${promptId} (${response.status})`);
                renderAllLayersInstructionsInContainer(traitDiv.id, trait, promptSet, promptId);
                continue;
            }

            const data = await response.json();
            console.log(`[${trait.name}] Data loaded successfully for ${promptSet}/${promptId}`);
            renderAllLayersDataInContainer(traitDiv.id, trait, data);
        } catch (error) {
            console.log(`[${trait.name}] Load failed for ${promptSet}/${promptId}:`, error.message);
            renderAllLayersInstructionsInContainer(traitDiv.id, trait, promptSet, promptId);
        }
    }
}

function renderAllLayersInstructionsInContainer(containerId, trait, promptSet, promptId) {
    const container = document.getElementById(containerId);
    if (!container) {
        console.error(`Container ${containerId} not found`);
        return;
    }

    // Get layer count from metadata
    const nLayers = trait.metadata?.n_layers || 26;
    const nCheckpoints = nLayers * 3;
    const promptLabel = promptSet && promptId ? `${promptSet}/${promptId}` : 'none selected';

    container.innerHTML = `
        <div style="margin-bottom: 4px;">
            <span style="color: var(--text-primary); font-size: 14px; font-weight: 600;">${window.getDisplayName(trait.name)}</span>
            <span style="color: var(--text-tertiary); font-size: 11px; margin-left: 8px;">⚠️ No data for prompt ${promptLabel}</span>
        </div>
        <div style="color: var(--text-secondary); font-size: 11px; margin-bottom: 4px;">
            The file <code>${promptId}.json</code> does not exist for this trait in ${promptSet || 'the prompt set'}. You may need to run inference with this prompt.
        </div>
        <div style="color: var(--text-secondary); font-size: 11px; margin-bottom: 4px;">
            To capture per-token projections at all ${nCheckpoints} checkpoints (${nLayers} layers × 3 sublayers):
        </div>
        <pre style="background: var(--bg-secondary); color: var(--text-primary); padding: 8px; border-radius: 4px; margin: 0; overflow-x: auto; font-size: 10px;">python inference/capture_raw_activations.py --experiment ${window.state.experimentData.name} --prompt-set ${promptSet || 'PROMPT_SET'}</pre>
    `;
}

function renderAllLayersDataInContainer(containerId, trait, data) {
    const container = document.getElementById(containerId);
    if (!container) {
        console.error(`Container ${containerId} not found`);
        return;
    }

    try {
        const promptProj = data.projections.prompt;  // [n_tokens, n_layers, 3]
        const responseProj = data.projections.response;
        console.log('Prompt projections shape:', promptProj.length, 'tokens x', promptProj[0].length, 'layers x', promptProj[0][0].length, 'sublayers');
        console.log('Response projections shape:', responseProj.length, 'tokens x', responseProj[0].length, 'layers x', responseProj[0][0].length, 'sublayers');

    // Combine prompt and response projections and tokens
    const allTokens = [...data.prompt.tokens, ...data.response.tokens];
    const allProj = [...promptProj, ...responseProj];
    const nPromptTokens = data.prompt.tokens.length;
    const nLayers = promptProj[0].length;

    // Create unique ID for this trait's heatmap
    // Sanitize trait name - replace slashes with dashes for valid HTML IDs
    const traitId = trait.name.replace(/\//g, '-');
    const trajectoryId = `trajectory-heatmap-${traitId}`;

    container.innerHTML = `
        <div style="margin-bottom: 4px;">
            <span style="color: var(--text-primary); font-size: 14px; font-weight: 600;">${window.getDisplayName(trait.name)}</span>
        </div>
        <div id="${trajectoryId}"></div>
    `;

    // Render combined trajectory heatmap with separator line
    setTimeout(() => {
        try {
            renderCombinedTrajectoryHeatmap(trajectoryId, allProj, allTokens, nPromptTokens, 250);  // Reduced height
        } catch (plotError) {
            console.error(`[${trait.name}] Heatmap rendering failed:`, plotError);
            container.innerHTML += `<div class="info" style="color: var(--danger);">Failed to render heatmap: ${plotError.message}</div>`;
        }
    }, 0);

    // Math rendering removed (renderMath function no longer exists)
    } catch (error) {
        console.error(`[${trait.name}] Error rendering trajectory data:`, error);
        if (container) {
            container.innerHTML = `<div class="card"><div class="card-title">Error: ${window.getDisplayName(trait.name)}</div><div class="info">Failed to render trajectory data: ${error.message}</div></div>`;
        }
    }
}

function renderCombinedTrajectoryHeatmap(divId, projections, tokens, nPromptTokens, height = 400) {
    // projections: [n_tokens, n_layers, 3_sublayers]
    // We'll show layer-averaged (average over 3 sublayers)

    const nTokens = projections.length;
    const nLayers = projections[0].length;  // Dynamically get number of layers
    console.log(`Rendering combined trajectory heatmap for ${nTokens} tokens x ${nLayers} layers (${nPromptTokens} prompt + ${nTokens - nPromptTokens} response)`);

    // Average over sublayers to get [n_tokens, n_layers]
    // Skip BOS token (index 0) for better visualization dynamic range
    const startIdx = 1;  // Skip <bos>

    const layerAvg = [];
    for (let t = startIdx; t < nTokens; t++) {
        layerAvg[t - startIdx] = [];
        for (let l = 0; l < nLayers; l++) {
            const avg = (projections[t][l][0] + projections[t][l][1] + projections[t][l][2]) / 3;
            layerAvg[t - startIdx][l] = avg;
        }
    }

    // Transpose for heatmap: [n_layers, n_tokens-1] (excluding BOS)
    const heatmapData = [];
    const nDisplayTokens = nTokens - startIdx;
    for (let l = 0; l < nLayers; l++) {
        heatmapData[l] = [];
        for (let t = 0; t < nDisplayTokens; t++) {
            heatmapData[l][t] = layerAvg[t][l];
        }
    }

    console.log(`[${divId}] Heatmap data sample:`, heatmapData[0].slice(0, 5));

    // Get colors from CSS variables
    const primaryColor = window.getCssVar('--primary-color', '#a09f6c');
    const separatorColor = `${primaryColor}80`;  // 50% opacity
    const highlightColor = `${primaryColor}33`;  // 20% opacity

    // Get current token index from global state (absolute index across prompt+response)
    // The heatmap skips BOS (startIdx=1), so token at absolute index N = heatmap x position (N - startIdx)
    const currentTokenIdx = window.state.currentTokenIndex || 0;
    const highlightX = currentTokenIdx - startIdx;

    // Create shapes array for separator line and current token highlight
    const shapes = [
        // Vertical line separating prompt and response (adjusted for skipped BOS)
        {
            type: 'line',
            xref: 'x',
            yref: 'paper',
            x0: (nPromptTokens - startIdx) - 0.5,
            x1: (nPromptTokens - startIdx) - 0.5,
            y0: 0,
            y1: 1,
            line: {
                color: separatorColor,
                width: 2,
                dash: 'dash'
            }
        },
        // Highlight for current token from global slider
        {
            type: 'rect',
            xref: 'x',
            yref: 'paper',
            x0: highlightX - 0.5,
            x1: highlightX + 0.5,
            y0: 0,
            y1: 1,
            fillcolor: highlightColor,
            line: { width: 0 }
        }
    ];

    const data = [{
        z: heatmapData,
        x: tokens.slice(startIdx),  // Skip BOS in token labels too
        y: Array.from({length: nLayers}, (_, i) => `L${i}`),
        type: 'heatmap',
        colorscale: window.ASYMB_COLORSCALE,
        zmid: 0,
        hovertemplate: 'Token: %{x}<br>Layer: %{y}<br>Score: %{z:.2f}<extra></extra>'
    }];

    // Adjust margins based on height
    const isCompact = height < 300;
    const layout = window.getPlotlyLayout({
        title: isCompact ? '' : 'Trait Trajectory',
        xaxis: {
            title: isCompact ? '' : 'Tokens',
            tickangle: -45,
            tickfont: { size: isCompact ? 9 : 12 }
        },
        yaxis: {
            title: isCompact ? '' : 'Layer',
            tickfont: { size: isCompact ? 9 : 12 }
        },
        shapes: shapes,
        height: height,
        margin: isCompact ? { t: 5, b: 30, l: 30, r: 5 } : { t: 40, b: 50, l: 40, r: 10 }
    });

    Plotly.newPlot(divId, data, layout, {displayModeBar: false});
}

// Export to global scope
window.renderTraitTrajectory = renderTraitTrajectory;
