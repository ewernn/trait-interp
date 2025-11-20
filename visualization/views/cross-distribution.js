// Cross Distribution View

async function renderCrossDistribution() {
    const contentArea = document.getElementById('content-area');

    contentArea.innerHTML = `
        <div class="info-box">
            <h3>Cross-Distribution Analysis</h3>
            <p>View cross-distribution generalization testing data. This shows which traits have been tested across different data distributions (instruction-based vs natural elicitation).</p>
        </div>

        <div id="cross-dist-loading" style="padding: 20px; text-align: center;">
            Loading cross-distribution data...
        </div>

        <div id="cross-dist-content" style="display: none;"></div>
    `;

    try {
        // Fetch cross-distribution index
        const response = await fetch('/api/cross-distribution/index');
        const data = await response.json();

        document.getElementById('cross-dist-loading').style.display = 'none';
        document.getElementById('cross-dist-content').style.display = 'block';

        const contentDiv = document.getElementById('cross-dist-content');

        if (data.error) {
            contentDiv.innerHTML = `
                <div class="card" style="margin: 20px 0;">
                    <div class="card-title" style="color: var(--danger);">⚠️ Error</div>
                    <p>${data.error}</p>
                    <p style="margin-top: 10px; color: var(--text-secondary);">
                        Run <code>python3 analysis/cross_distribution_scanner.py</code> to generate the index.
                    </p>
                </div>
            `;
            return;
        }

        // Find current experiment
        const currentExp = data.experiments.find(exp => exp.experiment === currentExperiment);
        if (!currentExp) {
            contentDiv.innerHTML = '<p>No data for current experiment.</p>';
            return;
        }

        const stats = currentExp.statistics;
        const traits = currentExp.traits;

        // Summary statistics
        contentDiv.innerHTML = `
            <div class="stats-grid">
                <div class="stat-card">
                    <span class="stat-label">Traits:</span>
                    <span class="stat-value">${stats.total_traits}</span>
                </div>
                <div class="stat-card">
                    <span class="stat-label">Complete:</span>
                    <span class="stat-value">${stats.complete_4x4}</span>
                </div>
                <div class="stat-card">
                    <span class="stat-label">Partial:</span>
                    <span class="stat-value">${stats.partial}</span>
                </div>
            </div>

            <div id="trait-table-container"></div>
        `;

        // Create trait table
        renderTraitTable(traits);

    } catch (error) {
        console.error('Error loading cross-distribution data:', error);
        document.getElementById('cross-dist-loading').innerHTML = `
            <div style="color: var(--danger);">
                Error loading data: ${error.message}
            </div>
        `;
    }
}

function renderTraitTable(traits) {
    const container = document.getElementById('trait-table-container');

    // Helper function to get best score across all methods for a quadrant
    function getBestScore(trait, quadrantKey) {
        if (!trait.best_accuracies || !trait.best_accuracies[quadrantKey]) {
            return null;
        }

        const quadScores = trait.best_accuracies[quadrantKey];
        let bestAcc = 0;
        let bestLayer = 0;
        let bestMethod = '';

        for (const [method, data] of Object.entries(quadScores)) {
            if (data.accuracy > bestAcc) {
                bestAcc = data.accuracy;
                bestLayer = data.layer;
                bestMethod = method;
            }
        }

        return { acc: bestAcc, layer: bestLayer, method: bestMethod };
    }

    // Sort traits by completeness
    const sortedTraits = traits.sort((a, b) => {
        if (a.is_complete_4x4 && !b.is_complete_4x4) return -1;
        if (!a.is_complete_4x4 && b.is_complete_4x4) return 1;
        return b.available_quadrants - a.available_quadrants;
    });

    const tableHTML = `
        <div class="card" style="margin: 20px 0;">
            <div class="card-title">Trait Data Availability</div>
            <table style="width: 100%; border-collapse: collapse; margin-top: 15px; color: var(--text-secondary);">
                <thead>
                    <tr>
                        <th style="text-align: left; padding: 12px 12px 16px 12px; color: var(--text-primary);">Trait</th>
                        <th style="text-align: center; padding: 12px 12px 16px 12px; color: var(--text-primary);">Separability</th>
                        <th style="text-align: center; padding: 12px; color: var(--text-primary);">Inst→Inst</th>
                        <th style="text-align: center; padding: 12px; color: var(--text-primary);">Inst→Nat</th>
                        <th style="text-align: center; padding: 12px; color: var(--text-primary);">Nat→Inst</th>
                        <th style="text-align: center; padding: 12px; color: var(--text-primary);">Nat→Nat</th>
                        <th style="text-align: center; padding: 12px; color: var(--text-primary);">Status</th>
                    </tr>
                </thead>
                <tbody>
                    ${sortedTraits.map(trait => {
                        const cd = trait.cross_distribution;
                        const sepBadge = trait.separability ?
                            `<span style="padding: 3px 8px; border-radius: 3px; font-size: 11px; background: ${
                                trait.separability === 'high' ? 'var(--success)' :
                                trait.separability === 'moderate' ? 'var(--info-border)' :
                                'var(--danger)'
                            }; color: var(--bg-primary);">${trait.separability.toUpperCase()}</span>` :
                            '<span style="color: var(--text-tertiary);">-</span>';

                        const statusBadge = trait.is_complete_4x4 ?
                            '<span style="padding: 3px 8px; border-radius: 3px; font-size: 11px; background: var(--success); color: var(--bg-primary);">✓ COMPLETE</span>' :
                            trait.available_quadrants > 0 ?
                            `<span style="padding: 3px 8px; border-radius: 3px; font-size: 11px; background: var(--info-border); color: var(--bg-primary);">${trait.available_quadrants}/4 PARTIAL</span>` :
                            '<span style="padding: 3px 8px; border-radius: 3px; font-size: 11px; background: var(--text-tertiary); color: var(--bg-primary);">NO DATA</span>';

                        // Get best scores for each quadrant
                        const instToInst = cd.inst_to_inst ? getBestScore(trait, 'inst_inst') : null;
                        const instToNat = cd.inst_to_nat ? getBestScore(trait, 'inst_nat') : null;
                        const natToInst = cd.nat_to_inst ? getBestScore(trait, 'nat_inst') : null;
                        const natToNat = cd.nat_to_nat ? getBestScore(trait, 'nat_nat') : null;

                        const formatScore = (score, quadrant) => {
                            if (!score) return '<span style="color: var(--text-tertiary);">-</span>';
                            return `<span class="quadrant-score" data-trait="${trait.name}" data-quadrant="${quadrant}" style="color: var(--text-secondary); font-size: 12px; cursor: pointer; text-decoration: underline; text-decoration-style: dotted;">${score.acc.toFixed(1)}%@L${score.layer}</span>`;
                        };

                        return `
                            <tr style="margin-bottom: 8px;">
                                <td style="padding: 12px; font-weight: 500; color: var(--text-primary);">${trait.name}</td>
                                <td style="padding: 12px; text-align: center;">${sepBadge}</td>
                                <td style="padding: 12px; text-align: center;">${formatScore(instToInst, 'inst_inst')}</td>
                                <td style="padding: 12px; text-align: center;">${formatScore(instToNat, 'inst_nat')}</td>
                                <td style="padding: 12px; text-align: center;">${formatScore(natToInst, 'nat_inst')}</td>
                                <td style="padding: 12px; text-align: center;">${formatScore(natToNat, 'nat_nat')}</td>
                                <td style="padding: 12px; text-align: center;">${statusBadge}</td>
                            </tr>
                        `;
                    }).join('')}
                </tbody>
            </table>
        </div>
    `;

    container.innerHTML = tableHTML;

    // Add click handlers to quadrant scores
    document.querySelectorAll('.quadrant-score').forEach(scoreSpan => {
        scoreSpan.addEventListener('click', (e) => {
            e.stopPropagation();
            const traitName = scoreSpan.dataset.trait;
            const quadrant = scoreSpan.dataset.quadrant;
            viewQuadrantDetails(traitName, quadrant);
        });
    });
}

async function viewQuadrantDetails(traitName, quadrant) {
    const modal = document.createElement('div');
    modal.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: var(--modal-backdrop);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 1000;
    `;

    // Quadrant display names
    const quadrantNames = {
        'inst_inst': 'Instruction → Instruction',
        'inst_nat': 'Instruction → Natural',
        'nat_inst': 'Natural → Instruction',
        'nat_nat': 'Natural → Natural'
    };

    modal.innerHTML = `
        <div style="background: var(--bg-primary); border-radius: 4px; padding: 20px; max-width: 1000px; max-height: 90vh; overflow-y: auto; position: relative;">
            <button id="close-modal" style="position: absolute; top: 8px; right: 8px; background: none; border: none; font-size: 20px; cursor: pointer; color: var(--text-secondary); padding: 4px 8px;">×</button>
            <h2 style="margin: 0 0 4px 0; font-size: 16px; font-weight: 600; color: var(--text-primary);">${traitName}</h2>
            <h3 style="margin: 0 0 16px 0; font-size: 12px; font-weight: 400; color: var(--text-secondary);">${quadrantNames[quadrant]}</h3>
            <div id="quadrant-detail-content" style="color: var(--text-secondary);">Loading cross-distribution data...</div>
        </div>
    `;

    document.body.appendChild(modal);

    const closeModal = () => document.body.removeChild(modal);
    modal.querySelector('#close-modal').addEventListener('click', closeModal);
    modal.addEventListener('click', (e) => {
        if (e.target === modal) closeModal();
    });

    // Load results if available
    try {
        const response = await fetch(`/api/cross-distribution/results/${traitName}`);
        const data = await response.json();

        const content = document.getElementById('quadrant-detail-content');

        if (data.error) {
            content.innerHTML = `
                <p style="color: var(--text-secondary);">No detailed results available for this trait yet.</p>
                <p style="margin-top: 10px; color: var(--text-secondary); font-size: 14px;">
                    Results will be available after running cross-distribution evaluation.
                </p>
            `;
            return;
        }

        // Extract quadrant data
        const quadrantData = data.quadrants[quadrant];
        if (!quadrantData) {
            content.innerHTML = `<p style="color: var(--danger);">No data for quadrant: ${quadrant}</p>`;
            return;
        }

        const methods = data.methods;
        const nLayers = data.n_layers;

        // Build accuracy heatmap data (4 methods × n_layers)
        const heatmapData = [];
        const methodLabels = [];

        for (const method of methods) {
            if (!quadrantData.methods[method]) continue;

            const methodData = quadrantData.methods[method];
            const allLayers = methodData.all_layers;
            if (!allLayers) continue;

            const accuracies = allLayers.map(layerData => layerData.accuracy * 100); // Convert to percentage
            heatmapData.push(accuracies);
            methodLabels.push(method.replace(/_/g, ' '));
        }

        if (heatmapData.length === 0) {
            content.innerHTML = `<p style="color: var(--text-secondary);">No method data available for this quadrant.</p>`;
            return;
        }

        // Create summary stats - simple compact table
        const availableMethods = methods.filter(method => quadrantData.methods[method]);

        const summaryHTML = availableMethods.length > 0 ? `
            <div style="margin-bottom: 16px;">
                <table style="width: 100%; border-collapse: collapse; font-size: 12px;">
                    <thead>
                        <tr>
                            <th style="text-align: left; padding: 4px 8px 12px 8px; color: var(--text-secondary); font-weight: 400;">Method</th>
                            <th style="text-align: right; padding: 4px 8px 12px 8px; color: var(--text-secondary); font-weight: 400;">Best</th>
                            <th style="text-align: right; padding: 4px 8px 12px 8px; color: var(--text-secondary); font-weight: 400;">Layer</th>
                            <th style="text-align: right; padding: 4px 8px 12px 8px; color: var(--text-secondary); font-weight: 400;">Avg</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${availableMethods.map(method => {
                            const methodData = quadrantData.methods[method];
                            return `
                                <tr>
                                    <td style="padding: 4px 8px; color: var(--text-primary);">${method.replace(/_/g, ' ')}</td>
                                    <td style="padding: 4px 8px; text-align: right; color: var(--text-primary);">${(methodData.best_accuracy * 100).toFixed(1)}%</td>
                                    <td style="padding: 4px 8px; text-align: right; color: var(--text-secondary);">${methodData.best_layer}</td>
                                    <td style="padding: 4px 8px; text-align: right; color: var(--text-tertiary);">${(methodData.avg_accuracy * 100).toFixed(1)}%</td>
                                </tr>
                            `;
                        }).join('')}
                    </tbody>
                </table>
            </div>
        ` : '';

        content.innerHTML = summaryHTML + `
            <div style="margin-top: 16px;">
                <h4 style="margin: 0 0 8px 0; font-size: 12px; font-weight: 600; color: var(--text-primary);">Accuracy Across Layers (${heatmapData.length} Method${heatmapData.length > 1 ? 's' : ''} × ${nLayers} Layers)</h4>
                <div id="quadrant-heatmap" style="width: 100%; height: 500px;"></div>
            </div>
            <div style="margin-top: 12px; font-size: 11px; color: var(--text-tertiary);">
                <div>Vector: ${quadrantData.vector_source} • Test: ${quadrantData.test_source}</div>
            </div>
        `;

        // Render heatmap - transpose so methods are on x-axis, layers on y-axis
        const layers = Array.from({ length: nLayers }, (_, i) => i);

        // Transpose: swap x and y
        const transposedData = heatmapData[0].map((_, colIndex) =>
            heatmapData.map(row => row[colIndex])
        );

        const trace = {
            z: transposedData,
            x: methodLabels,  // Methods on x-axis
            y: layers,        // Layers on y-axis
            type: 'heatmap',
            colorscale: 'Viridis',
            colorbar: {
                title: 'Accuracy (%)',
                titleside: 'right'
            },
            hovertemplate: '%{x}<br>Layer %{y}<br>Accuracy: %{z:.1f}%<extra></extra>',
            zmin: 0,
            zmax: 100
        };

        Plotly.newPlot('quadrant-heatmap', [trace], window.getPlotlyLayout({
            margin: { l: 50, r: 60, t: 20, b: 60 },
            xaxis: {
                title: 'Method',
                side: 'bottom'
            },
            yaxis: {
                title: 'Layer',
                showticklabels: true,
                autorange: 'reversed'  // Layer 0 at top
            },
            height: 600
        }), { displayModeBar: false });

    } catch (error) {
        const content = document.getElementById('quadrant-detail-content');
        content.innerHTML = `<p style="color: var(--danger);">Error loading results: ${error.message}</p>`;
    }
}

// Render monitoring view
// Export to global scope
window.renderCrossDistribution = renderCrossDistribution;
