// Cross Distribution View

async function renderCrossDistribution() {
    const contentArea = document.getElementById('content-area');

    contentArea.innerHTML = `
        <div class="explanation">
            <div class="explanation-summary">We can measure how well trait vectors generalize across different data distributions.</div>
            <div class="explanation-details">
                <p><strong>Cross-distribution testing:</strong> Train vectors on one distribution (instruction-based or natural elicitation), test on another. Measures whether vectors capture genuine behavioral traits vs. surface patterns.</p>
                <p><strong>Four quadrants:</strong> Inst→Inst (baseline), Inst→Nat, Nat→Inst, Nat→Nat (strongest signal)</p>
            </div>
        </div>

        <div id="cross-dist-loading" style="padding: 16px; color: var(--text-secondary);">
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
                <div style="margin: 16px 0; padding: 16px; background: var(--bg-secondary); border-radius: 4px;">
                    <div style="font-size: 12px; font-weight: 600; color: var(--danger); margin-bottom: 8px;">⚠️ Error</div>
                    <p style="margin: 0; font-size: 12px; color: var(--text-secondary);">${data.error}</p>
                    <p style="margin: 8px 0 0 0; font-size: 11px; color: var(--text-tertiary);">
                        Run <code>python3 analysis/cross_distribution_scanner.py</code> to generate the index.
                    </p>
                </div>
            `;
            return;
        }

        // Find current experiment
        const currentExp = data.experiments.find(exp => exp.experiment === window.state.currentExperiment);
        if (!currentExp) {
            contentDiv.innerHTML = '<p style="font-size: 12px; color: var(--text-secondary);">No data for current experiment.</p>';
            return;
        }

        const stats = currentExp.statistics;
        const traits = currentExp.traits;

        // Summary statistics (inline format per design standards)
        contentDiv.innerHTML = `
            <div style="display: flex; gap: 16px; margin: 16px 0; font-size: 12px;">
                <div><span style="color: var(--text-secondary);">Traits:</span> <span style="font-size: 14px; color: var(--text-primary);">${stats.total_traits}</span></div>
                <div><span style="color: var(--text-secondary);">Complete:</span> <span style="font-size: 14px; color: var(--text-primary);">${stats.complete_4x4}</span></div>
                <div><span style="color: var(--text-secondary);">Partial:</span> <span style="font-size: 14px; color: var(--text-primary);">${stats.partial}</span></div>
            </div>

            <div id="trait-table-container"></div>
        `;

        // Create trait table
        renderTraitTable(traits);

    } catch (error) {
        console.error('Error loading cross-distribution data:', error);
        document.getElementById('cross-dist-loading').innerHTML = `
            <div style="color: var(--danger); font-size: 12px;">
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
        <div style="margin: 16px 0;">
            <div style="font-size: 12px; font-weight: 600; color: var(--text-primary); margin-bottom: 8px;">Trait Data Availability</div>
            <table style="width: 100%; border-collapse: collapse; font-size: 12px;">
                <thead>
                    <tr>
                        <th style="text-align: left; padding: 4px 8px 8px 0; color: var(--text-secondary); font-weight: 400;">Trait</th>
                        <th style="text-align: center; padding: 4px 8px 8px 8px; color: var(--text-secondary); font-weight: 400;">Sep</th>
                        <th style="text-align: center; padding: 4px 8px 8px 8px; color: var(--text-secondary); font-weight: 400;">I→I</th>
                        <th style="text-align: center; padding: 4px 8px 8px 8px; color: var(--text-secondary); font-weight: 400;">I→N</th>
                        <th style="text-align: center; padding: 4px 8px 8px 8px; color: var(--text-secondary); font-weight: 400;">N→I</th>
                        <th style="text-align: center; padding: 4px 8px 8px 8px; color: var(--text-secondary); font-weight: 400;">N→N</th>
                        <th style="text-align: center; padding: 4px 8px 8px 8px; color: var(--text-secondary); font-weight: 400;">Status</th>
                    </tr>
                </thead>
                <tbody>
                    ${sortedTraits.map(trait => {
                        const cd = trait.cross_distribution;
                        const sepBadge = trait.separability ?
                            `<span style="padding: 2px 6px; border-radius: 3px; font-size: 10px; background: ${
                                trait.separability === 'high' ? '#4caf50' :
                                trait.separability === 'moderate' ? '#4a9eff' :
                                '#f44336'
                            }; color: var(--bg-primary);">${trait.separability[0].toUpperCase()}</span>` :
                            '<span style="color: var(--text-tertiary);">-</span>';

                        const statusBadge = trait.is_complete_4x4 ?
                            '<span style="padding: 2px 6px; border-radius: 3px; font-size: 10px; background: #4caf50; color: var(--bg-primary);">✓</span>' :
                            trait.available_quadrants > 0 ?
                            `<span style="padding: 2px 6px; border-radius: 3px; font-size: 10px; background: #4a9eff; color: var(--bg-primary);">${trait.available_quadrants}/4</span>` :
                            '<span style="padding: 2px 6px; border-radius: 3px; font-size: 10px; background: var(--text-tertiary); color: var(--bg-primary);">-</span>';

                        // Get best scores for each quadrant
                        const instToInst = cd.inst_to_inst ? getBestScore(trait, 'inst_inst') : null;
                        const instToNat = cd.inst_to_nat ? getBestScore(trait, 'inst_nat') : null;
                        const natToInst = cd.nat_to_inst ? getBestScore(trait, 'nat_inst') : null;
                        const natToNat = cd.nat_to_nat ? getBestScore(trait, 'nat_nat') : null;

                        const formatScore = (score, quadrant) => {
                            if (!score) return '<span style="color: var(--text-tertiary);">-</span>';
                            return `<span class="quadrant-score" data-trait="${trait.name}" data-quadrant="${quadrant}" style="color: var(--primary-color); cursor: pointer; text-decoration: underline; text-decoration-style: dotted;">${score.acc.toFixed(1)}%<span style="color: var(--text-tertiary); font-size: 10px;">@${score.layer}</span></span>`;
                        };

                        return `
                            <tr style="line-height: 1.3;">
                                <td style="padding: 4px 8px 4px 0; color: var(--text-primary);">${trait.name}</td>
                                <td style="padding: 4px 8px; text-align: center;">${sepBadge}</td>
                                <td style="padding: 4px 8px; text-align: center;">${formatScore(instToInst, 'inst_inst')}</td>
                                <td style="padding: 4px 8px; text-align: center;">${formatScore(instToNat, 'inst_nat')}</td>
                                <td style="padding: 4px 8px; text-align: center;">${formatScore(natToInst, 'nat_inst')}</td>
                                <td style="padding: 4px 8px; text-align: center;">${formatScore(natToNat, 'nat_nat')}</td>
                                <td style="padding: 4px 8px; text-align: center;">${statusBadge}</td>
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
        background: rgba(0, 0, 0, 0.7);
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
        <div style="background: var(--bg-primary); border-radius: 4px; padding: 16px; max-width: 1000px; max-height: 90vh; overflow-y: auto; position: relative;">
            <button id="close-modal" style="position: absolute; top: 8px; right: 8px; background: none; border: none; font-size: 18px; cursor: pointer; color: var(--text-secondary); padding: 4px 8px;">×</button>
            <h2 style="margin: 0 0 2px 0; font-size: 14px; font-weight: 600; color: var(--text-primary);">${traitName}</h2>
            <h3 style="margin: 0 0 16px 0; font-size: 12px; font-weight: 400; color: var(--text-secondary);">${quadrantNames[quadrant]}</h3>
            <div id="quadrant-detail-content" style="color: var(--text-secondary); font-size: 12px;">Loading cross-distribution data...</div>
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
                <p style="color: var(--text-secondary); margin: 0;">No detailed results available for this trait yet.</p>
                <p style="margin: 8px 0 0 0; color: var(--text-tertiary); font-size: 11px;">
                    Results will be available after running cross-distribution evaluation.
                </p>
            `;
            return;
        }

        // Extract quadrant data
        const quadrantData = data.quadrants[quadrant];
        if (!quadrantData) {
            content.innerHTML = `<p style="color: var(--danger); margin: 0;">No data for quadrant: ${quadrant}</p>`;
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
            content.innerHTML = `<p style="color: var(--text-secondary); margin: 0;">No method data available for this quadrant.</p>`;
            return;
        }

        // Create summary stats - simple compact table
        const availableMethods = methods.filter(method => quadrantData.methods[method]);

        const summaryHTML = availableMethods.length > 0 ? `
            <div style="margin-bottom: 16px;">
                <table style="width: 100%; border-collapse: collapse; font-size: 11px;">
                    <thead>
                        <tr>
                            <th style="text-align: left; padding: 2px 8px 8px 0; color: var(--text-secondary); font-weight: 400;">Method</th>
                            <th style="text-align: right; padding: 2px 8px 8px 8px; color: var(--text-secondary); font-weight: 400;">Best</th>
                            <th style="text-align: right; padding: 2px 8px 8px 8px; color: var(--text-secondary); font-weight: 400;">Layer</th>
                            <th style="text-align: right; padding: 2px 8px 8px 8px; color: var(--text-secondary); font-weight: 400;">Avg</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${availableMethods.map(method => {
                            const methodData = quadrantData.methods[method];
                            return `
                                <tr>
                                    <td style="padding: 2px 8px 2px 0; color: var(--text-primary);">${method.replace(/_/g, ' ')}</td>
                                    <td style="padding: 2px 8px; text-align: right; color: var(--text-primary);">${(methodData.best_accuracy * 100).toFixed(1)}%</td>
                                    <td style="padding: 2px 8px; text-align: right; color: var(--text-secondary);">${methodData.best_layer}</td>
                                    <td style="padding: 2px 8px; text-align: right; color: var(--text-tertiary);">${(methodData.avg_accuracy * 100).toFixed(1)}%</td>
                                </tr>
                            `;
                        }).join('')}
                    </tbody>
                </table>
            </div>
        ` : '';

        content.innerHTML = summaryHTML + `
            <div style="margin-top: 16px;">
                <h4 style="margin: 0 0 8px 0; font-size: 12px; font-weight: 600; color: var(--text-primary);">Accuracy: ${heatmapData.length} Method${heatmapData.length > 1 ? 's' : ''} × ${nLayers} Layers</h4>
                <div id="quadrant-heatmap" style="width: 100%; height: 400px;"></div>
            </div>
            <div style="margin-top: 8px; font-size: 10px; color: var(--text-tertiary);">
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
                title: { text: 'Acc %', font: { size: 10 } },
                titleside: 'right',
                len: 0.8
            },
            hovertemplate: '%{x}<br>Layer %{y}<br>Accuracy: %{z:.1f}%<extra></extra>',
            zmin: 0,
            zmax: 100
        };

        Plotly.newPlot('quadrant-heatmap', [trace], window.getPlotlyLayout({
            margin: { l: 40, r: 60, t: 10, b: 50 },
            xaxis: {
                title: { text: 'Method', font: { size: 10 } },
                side: 'bottom',
                tickfont: { size: 9 }
            },
            yaxis: {
                title: { text: 'Layer', font: { size: 10 } },
                showticklabels: true,
                autorange: 'reversed',
                tickfont: { size: 9 }
            },
            height: 400
        }), { displayModeBar: false });

    } catch (error) {
        const content = document.getElementById('quadrant-detail-content');
        content.innerHTML = `<p style="color: var(--danger); margin: 0;">Error loading results: ${error.message}</p>`;
    }
}

// Export to global scope
window.renderCrossDistribution = renderCrossDistribution;
