// Layer Deep Dive View - Coming Soon
// Will provide trait-specific analysis of layer components

async function renderLayerDeepDive() {
    const contentArea = document.getElementById('content-area');
    const filteredTraits = window.getFilteredTraits();

    const traitNames = filteredTraits.length > 0
        ? filteredTraits.map(t => window.getDisplayName(t.name)).join(', ')
        : 'none selected';

    contentArea.innerHTML = `
        <div class="card" style="max-width: 800px;">
            <div class="card-title">Layer Deep Dive</div>

            <div style="background: var(--bg-tertiary); padding: 8px 12px; border-radius: 2px; margin-bottom: 16px;">
                <div style="color: var(--text-secondary); font-size: 11px; margin-bottom: 4px;">Selected traits</div>
                <div style="color: var(--text-primary); font-size: 12px; font-weight: 600;">${traitNames}</div>
            </div>

            <div style="color: var(--text-secondary); font-size: 12px; margin-bottom: 16px;">
                <p style="margin-bottom: 8px;">
                    This view will provide <strong>trait-specific</strong> analysis of what happens inside a single layer.
                    Unlike the trajectory views (which show projections across layers), this will decompose
                    <em>how</em> the trait score is computed within a layer.
                </p>
            </div>

            <div style="background: var(--bg-secondary); padding: 12px; border-radius: 2px; margin-bottom: 16px;">
                <h3 style="color: var(--text-primary); margin: 0 0 12px 0; font-size: 13px;">Planned Features</h3>

                <div style="margin-bottom: 12px;">
                    <div style="color: var(--primary-color); font-weight: 600; font-size: 12px; margin-bottom: 2px;">1. Attention vs MLP Breakdown</div>
                    <div style="color: var(--text-secondary); font-size: 11px;">
                        Project attention output and MLP output onto trait vector separately.
                        Shows whether the trait is computed by attention (token mixing) or MLP (feature transformation).
                    </div>
                </div>

                <div style="margin-bottom: 12px;">
                    <div style="color: var(--primary-color); font-weight: 600; font-size: 12px; margin-bottom: 2px;">2. Per-Head Contributions</div>
                    <div style="color: var(--text-secondary); font-size: 11px;">
                        Which of the 8 attention heads push toward or against the trait?
                        Identifies specific heads responsible for trait computation.
                    </div>
                </div>

                <div style="margin-bottom: 12px;">
                    <div style="color: var(--primary-color); font-weight: 600; font-size: 12px; margin-bottom: 2px;">3. SAE Feature Decomposition</div>
                    <div style="color: var(--text-secondary); font-size: 11px;">
                        Map raw MLP neurons (9216) to interpretable SAE features (16k from GemmaScope).
                        Instead of "neuron 4721 contributes +0.8", show "harm_detection feature contributes +0.5".
                    </div>
                </div>

                <div>
                    <div style="color: var(--primary-color); font-weight: 600; font-size: 12px; margin-bottom: 2px;">4. Per-Token Attribution</div>
                    <div style="color: var(--text-secondary); font-size: 11px;">
                        Break down why each token has its trait score: which neurons and heads contributed?
                    </div>
                </div>
            </div>

            <div style="color: var(--text-tertiary); font-size: 11px;">
                See <code>docs/future_ideas.md</code> for full research plan (#6 SAE Feature Decomposition, #9 Component Ablation).
            </div>
        </div>
    `;
}

// Export to global scope
window.renderLayerDeepDive = renderLayerDeepDive;
