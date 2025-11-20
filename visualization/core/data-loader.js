// Centralized Data Loading for Trait Interpretation Visualization

class DataLoader {
    /**
     * Fetch Tier 2 data (residual stream activations for all layers)
     * @param {Object} trait - Trait object with name property (category/trait_name format)
     * @param {number} promptNum - Prompt number
     * @returns {Promise<Object>} - Data with projections and logit lens
     */
    static async fetchTier2(trait, promptNum) {
        const url = `../experiments/${window.state.experimentData.name}/extraction/${trait.name}/inference/residual_stream_activations/prompt_${promptNum}.json`;
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to fetch tier 2 data for ${trait.name} prompt ${promptNum}`);
        }
        return await response.json();
    }

    /**
     * Fetch Tier 3 data (layer internal states for a specific layer)
     * @param {Object} trait - Trait object with name property (category/trait_name format)
     * @param {number} promptNum - Prompt number
     * @param {number} layer - Layer number (default: 16)
     * @returns {Promise<Object>} - Data with attention heads, MLP activations, etc.
     */
    static async fetchTier3(trait, promptNum, layer = 16) {
        const url = `../experiments/${window.state.experimentData.name}/extraction/${trait.name}/inference/layer_internal_states/prompt_${promptNum}_layer${layer}.json`;
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to fetch tier 3 data for ${trait.name} prompt ${promptNum} layer ${layer}`);
        }
        return await response.json();
    }

    /**
     * Fetch vector metadata for a specific method and layer
     * @param {Object} trait - Trait object with name property (category/trait_name format)
     * @param {string} method - Extraction method (mean_diff, probe, ica, gradient)
     * @param {number} layer - Layer number
     * @returns {Promise<Object>} - Vector metadata
     */
    static async fetchVectorMetadata(trait, method, layer) {
        const url = `../experiments/${window.state.experimentData.name}/extraction/${trait.name}/extraction/vectors/${method}_layer${layer}_metadata.json`;
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to fetch vector metadata for ${trait.name} ${method} layer ${layer}`);
        }
        return await response.json();
    }

    /**
     * Fetch all vector metadata for a trait (all methods and layers)
     * @param {Object} trait - Trait object with name property
     * @param {Array<string>} methods - Array of method names
     * @param {Array<number>} layers - Array of layer numbers
     * @returns {Promise<Object>} - Object mapping method_layerN to metadata
     */
    static async fetchAllVectorMetadata(trait, methods, layers) {
        const results = {};
        const promises = [];

        for (const method of methods) {
            for (const layer of layers) {
                const key = `${method}_layer${layer}`;
                const promise = this.fetchVectorMetadata(trait, method, layer)
                    .then(data => {
                        results[key] = data;
                    })
                    .catch(e => {
                        console.warn(`No vector metadata for ${trait.name} ${key}:`, e.message);
                        results[key] = null;
                    });
                promises.push(promise);
            }
        }

        await Promise.all(promises);
        return results;
    }

    /**
     * Fetch cross-distribution analysis data
     * @param {string} traitBaseName - Base name of trait (without _natural suffix)
     * @returns {Promise<Object>} - Cross-distribution results
     */
    static async fetchCrossDistribution(traitBaseName) {
        const url = `../experiments/${window.state.experimentData.name}/validation/${traitBaseName}_full_4x4_results.json`;
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to fetch cross-distribution data for ${traitBaseName}`);
        }
        return await response.json();
    }

    /**
     * Fetch prompt data (shared across all traits)
     * @param {number} promptIdx - Prompt index (0, 1, 2, ...)
     * @returns {Promise<Object>} - Shared prompt data
     */
    static async fetchPrompt(promptIdx) {
        const url = `../experiments/${window.state.experimentData.name}/inference/prompts/prompt_${promptIdx}.json`;
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to fetch prompt ${promptIdx}`);
        }
        return await response.json();
    }

    /**
     * Fetch trait projections for a specific prompt
     * @param {Object} trait - Trait object with name property (category/trait_name format)
     * @param {number} promptIdx - Prompt index
     * @returns {Promise<Object>} - Trait-specific projection data
     */
    static async fetchProjections(trait, promptIdx) {
        const url = `../experiments/${window.state.experimentData.name}/inference/projections/${trait.name}/prompt_${promptIdx}.json`;
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to fetch projections for ${trait.name} prompt ${promptIdx}`);
        }
        return await response.json();
    }

    /**
     * Fetch combined inference data
     * Combines prompt + projections into single object
     * @param {Object} trait - Trait object with name property
     * @param {number} promptIdx - Prompt index
     * @returns {Promise<Object>} - Combined data
     */
    static async fetchInferenceCombined(trait, promptIdx) {
        const prompt = await this.fetchPrompt(promptIdx);
        const projections = await this.fetchProjections(trait, promptIdx);

        return {
            prompt: prompt.prompt,
            response: prompt.response,
            tokens: prompt.tokens,
            trait_scores: { [trait.name]: projections.scores },
            dynamics: { [trait.name]: projections.dynamics }
        };
    }

    /**
     * Discover available prompts
     * @returns {Promise<number[]>} - Array of available prompt indices
     */
    static async discoverPrompts() {
        try {
            const indices = [];
            for (let i = 0; i < 100; i++) {
                try {
                    await this.fetchPrompt(i);
                    indices.push(i);
                } catch (e) {
                    break; // Stop when we hit the first missing prompt
                }
            }
            return indices;
        } catch (e) {
            console.warn('Failed to discover prompts:', e);
            return [];
        }
    }

    /**
     * Fetch JSON file for preview
     * @param {Object} trait - Trait object with name property (category/trait_name format)
     * @param {string} type - Type of JSON (trait_definition, activations_metadata, pos, neg)
     * @returns {Promise<Object>} - JSON data
     */
    static async fetchJSON(trait, type) {
        let url;
        if (type === 'trait_definition') {
            url = `../experiments/${window.state.experimentData.name}/extraction/${trait.name}/extraction/trait_definition.json`;
        } else if (type === 'activations_metadata') {
            url = `../experiments/${window.state.experimentData.name}/extraction/${trait.name}/extraction/activations/metadata.json`;
        } else if (type === 'pos' || type === 'neg') {
            url = `../experiments/${window.state.experimentData.name}/extraction/${trait.name}/extraction/responses/${type}.json`;
        } else {
            throw new Error(`Unknown JSON type: ${type}`);
        }

        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to fetch JSON for ${trait.name} ${type}`);
        }
        return await response.json();
    }

    /**
     * Fetch CSV file for preview
     * @param {Object} trait - Trait object with name property (category/trait_name format)
     * @param {string} category - Category (pos or neg)
     * @param {number} limit - Maximum rows to parse (default: 10)
     * @returns {Promise<Object>} - Parsed CSV data with Papa Parse
     */
    static async fetchCSV(trait, category, limit = 10) {
        const url = `../experiments/${window.state.experimentData.name}/extraction/${trait.name}/extraction/responses/${category}.csv`;
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to fetch CSV for ${trait.name} ${category}`);
        }
        const text = await response.text();
        const parsed = Papa.parse(text, { header: true });
        return {
            data: parsed.data.slice(0, limit),
            total: parsed.data.length,
            headers: parsed.data.length > 0 ? Object.keys(parsed.data[0]) : []
        };
    }

    /**
     * Check if vector extraction exists for a trait
     * @param {Object} trait - Trait object with name property (category/trait_name format)
     * @returns {Promise<boolean>} - True if vectors exist
     */
    static async checkVectorsExist(trait) {
        try {
            const testUrl = `../experiments/${window.state.experimentData.name}/extraction/${trait.name}/extraction/vectors/probe_layer16_metadata.json`;
            const response = await fetch(testUrl);
            return response.ok;
        } catch (e) {
            return false;
        }
    }

    /**
     * Fetch SAE features if available
     * @param {Object} trait - Trait object with name property (category/trait_name format)
     * @param {number} promptNum - Prompt number
     * @param {number} layer - Layer number
     * @returns {Promise<Object>} - SAE feature activations
     */
    static async fetchSAEFeatures(trait, promptNum, layer = 16) {
        const url = `../experiments/${window.state.experimentData.name}/extraction/${trait.name}/inference/sae_features/prompt_${promptNum}_layer${layer}_sae.pt`;
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to fetch SAE features for ${trait.name} prompt ${promptNum} layer ${layer}`);
        }
        // Note: PT files need special handling - this would return blob
        return await response.blob();
    }
}

// Export to global scope
window.DataLoader = DataLoader;
