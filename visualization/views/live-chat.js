// Live Chat View - Chat with the model while watching trait dynamics in real-time
//
// Stream tokens from /api/chat and plot trait scores as they arrive

// Reuse TRAIT_COLORS from trait-dynamics.js if available, otherwise define locally
const CHAT_TRAIT_COLORS = window.TRAIT_COLORS || [
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

// Chat state
let chatHistory = [];  // [{token, trait_scores}, ...]
let isGenerating = false;
let abortController = null;

/**
 * Render the live chat view
 */
async function renderLiveChat() {
    const container = document.getElementById('content-area');
    if (!container) return;

    container.innerHTML = `
        <div class="tool-view live-chat-view">
            <div class="live-chat-container">
                <!-- Top: Trait Chart -->
                <div class="trait-chart-panel">
                    <div class="chart-header">
                        <h3>Trait Dynamics</h3>
                        <div class="chart-legend" id="chart-legend"></div>
                    </div>
                    <div id="trait-chart" class="trait-chart"></div>
                </div>

                <!-- Bottom: Chat Interface -->
                <div class="chat-panel">
                    <div class="chat-messages" id="chat-messages">
                        <div class="chat-placeholder">Send a message to start chatting...</div>
                    </div>
                    <div class="chat-input-area">
                        <textarea
                            id="chat-input"
                            placeholder="Type your message..."
                            rows="2"
                        ></textarea>
                        <div class="chat-controls">
                            <button id="send-btn" class="btn btn-primary">Send</button>
                            <button id="clear-btn" class="btn btn-secondary">Clear</button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;

    // Add styles
    addLiveChatStyles();

    // Setup event handlers
    setupChatHandlers();

    // Initialize empty chart
    initTraitChart();
}

/**
 * Add CSS styles for live chat view (uses primitives from styles.css)
 */
function addLiveChatStyles() {
    if (document.getElementById('live-chat-styles')) return;

    const styles = document.createElement('style');
    styles.id = 'live-chat-styles';
    styles.textContent = `
        .live-chat-view {
            height: calc(100vh - 120px);
            display: flex;
            flex-direction: column;
        }

        .live-chat-container {
            display: flex;
            flex-direction: column;
            gap: 16px;
            flex: 1;
            min-height: 0;
        }

        .chat-panel {
            display: flex;
            flex-direction: column;
            background: var(--bg-secondary);
            border-radius: 2px;
            overflow: hidden;
            flex: 1;
            min-height: 200px;
        }

        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 12px;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .chat-placeholder {
            color: var(--text-tertiary);
            text-align: center;
            padding: 40px;
        }

        .chat-message {
            padding: 8px 12px;
            border-radius: 2px;
            max-width: 85%;
            font-size: var(--text-sm);
            line-height: 1.4;
        }

        .chat-message.user {
            background: var(--primary-color);
            color: var(--text-on-primary);
            align-self: flex-end;
        }

        .chat-message.assistant {
            background: var(--bg-tertiary);
            color: var(--text-primary);
            align-self: flex-start;
        }

        .chat-input-area {
            padding: 12px;
            border-top: 1px solid var(--border-color);
            background: var(--bg-tertiary);
        }

        .chat-input-area textarea {
            width: 100%;
            background: var(--bg-primary);
            border: 1px solid var(--border-color);
            border-radius: 2px;
            padding: 8px;
            color: var(--text-primary);
            font-family: inherit;
            font-size: var(--text-sm);
            resize: none;
        }

        .chat-input-area textarea:focus {
            outline: none;
            border-color: var(--primary-color);
        }

        .chat-controls {
            display: flex;
            gap: 8px;
            margin-top: 8px;
        }

        .chat-controls button {
            padding: 4px 12px;
            border-radius: 2px;
            cursor: pointer;
            font-size: var(--text-sm);
        }

        .chat-controls .btn-primary {
            background: var(--primary-color);
            color: var(--text-on-primary);
            border: none;
        }

        .chat-controls .btn-primary:hover {
            background: var(--primary-hover);
        }

        .chat-controls .btn-secondary {
            background: transparent;
            color: var(--text-secondary);
            border: none;
        }

        .trait-chart-panel {
            display: flex;
            flex-direction: column;
            background: var(--bg-secondary);
            border-radius: 2px;
            overflow: hidden;
            flex: 0 0 auto;
            max-height: 350px;
        }

        .chart-header {
            padding: 12px;
            border-bottom: 1px solid var(--border-color);
        }

        .chart-header h3 {
            margin: 0 0 8px 0;
            font-size: var(--text-base);
            font-weight: 600;
            color: var(--text-primary);
        }

        .chart-legend {
            display: flex;
            flex-wrap: wrap;
            gap: 4px 12px;
            font-size: var(--text-xs);
            color: var(--text-secondary);
        }

        .legend-item {
            display: flex;
            align-items: center;
            gap: 4px;
        }

        .legend-color {
            width: 12px;
            height: 3px;
            border-radius: 1px;
        }

        .trait-chart {
            flex: 1;
            min-height: 200px;
        }

        .generating-indicator {
            display: inline-block;
            width: 6px;
            height: 6px;
            background: var(--primary-color);
            border-radius: 50%;
            animation: chat-pulse 1s infinite;
            margin-left: 4px;
        }

        @keyframes chat-pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.3; }
        }
    `;
    document.head.appendChild(styles);
}

/**
 * Setup chat event handlers
 */
function setupChatHandlers() {
    const input = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    const clearBtn = document.getElementById('clear-btn');

    sendBtn.addEventListener('click', () => sendMessage());
    clearBtn.addEventListener('click', () => clearChat());

    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
}

/**
 * Send message and stream response
 */
async function sendMessage() {
    const input = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    const messagesDiv = document.getElementById('chat-messages');

    const prompt = input.value.trim();
    if (!prompt || isGenerating) return;

    // Clear placeholder
    const placeholder = messagesDiv.querySelector('.chat-placeholder');
    if (placeholder) placeholder.remove();

    // Add user message
    const userMsg = document.createElement('div');
    userMsg.className = 'chat-message user';
    userMsg.textContent = prompt;
    messagesDiv.appendChild(userMsg);

    // Clear input
    input.value = '';

    // Add assistant message container
    const assistantMsg = document.createElement('div');
    assistantMsg.className = 'chat-message assistant';
    assistantMsg.innerHTML = '<span class="generating-indicator"></span>';
    messagesDiv.appendChild(assistantMsg);

    // Scroll to bottom
    messagesDiv.scrollTop = messagesDiv.scrollHeight;

    // Reset chat history for new response
    chatHistory = [];

    // Disable send button
    isGenerating = true;
    sendBtn.disabled = true;
    sendBtn.textContent = 'Generating...';

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt: prompt,
                experiment: window.state.currentExperiment || 'gemma-2-2b-it',
                max_tokens: 100,
                temperature: 0.7
            })
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let responseText = '';

        let streamDone = false;
        while (!streamDone) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            // Process SSE events
            const lines = buffer.split('\n');
            buffer = lines.pop();  // Keep incomplete line in buffer

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const event = JSON.parse(line.slice(6));

                        if (event.error) {
                            assistantMsg.innerHTML = `<span style="color: var(--danger);">Error: ${event.error}</span>`;
                            streamDone = true;
                            break;
                        }

                        // Handle status events (loading, etc.)
                        if (event.status) {
                            assistantMsg.innerHTML = `<span style="color: var(--text-tertiary);">${event.message}</span><span class="generating-indicator"></span>`;
                            continue;
                        }

                        if (event.done) {
                            // Remove generating indicator
                            const indicator = assistantMsg.querySelector('.generating-indicator');
                            if (indicator) indicator.remove();
                            streamDone = true;
                            break;
                        }

                        // Append token
                        responseText += event.token;

                        // Update message display
                        const indicator = assistantMsg.querySelector('.generating-indicator');
                        if (indicator) {
                            assistantMsg.innerHTML = window.escapeHtml(responseText) + '<span class="generating-indicator"></span>';
                        } else {
                            assistantMsg.textContent = responseText;
                        }

                        // Store for chart
                        chatHistory.push(event);

                        // Update chart
                        updateTraitChart();

                        // Scroll to bottom
                        messagesDiv.scrollTop = messagesDiv.scrollHeight;

                    } catch (e) {
                        console.error('Failed to parse SSE event:', e);
                    }
                }
            }
        }

    } catch (error) {
        assistantMsg.innerHTML = `<span style="color: var(--danger);">Error: ${error.message}</span>`;
    } finally {
        isGenerating = false;
        sendBtn.disabled = false;
        sendBtn.textContent = 'Send';

        // Remove any remaining indicator
        const indicator = assistantMsg.querySelector('.generating-indicator');
        if (indicator) indicator.remove();
    }
}

/**
 * Clear chat and reset chart
 */
function clearChat() {
    const messagesDiv = document.getElementById('chat-messages');
    const sendBtn = document.getElementById('send-btn');

    messagesDiv.innerHTML = '<div class="chat-placeholder">Send a message to start chatting...</div>';
    chatHistory = [];
    isGenerating = false;
    sendBtn.disabled = false;
    sendBtn.textContent = 'Send';
    initTraitChart();
}

/**
 * Initialize empty trait chart
 */
function initTraitChart() {
    const chartDiv = document.getElementById('trait-chart');
    if (!chartDiv) return;

    const layout = window.getPlotlyLayout({
        margin: { l: 50, r: 20, t: 20, b: 40 },
        xaxis: {
            title: 'Token',
            showgrid: true,
        },
        yaxis: {
            title: 'Trait Score',
            showgrid: true,
            zeroline: true,
        },
        showlegend: false,
        hovermode: 'x unified',
    });

    Plotly.newPlot(chartDiv, [], layout, { responsive: true });
}

/**
 * Update trait chart with new data
 */
function updateTraitChart() {
    const chartDiv = document.getElementById('trait-chart');
    const legendDiv = document.getElementById('chart-legend');
    if (!chartDiv || chatHistory.length === 0) return;

    // Get all trait names from first event
    const firstEvent = chatHistory[0];
    const traitNames = Object.keys(firstEvent.trait_scores || {});

    if (traitNames.length === 0) return;

    // Build traces
    const traces = traitNames.map((trait, idx) => {
        const y = chatHistory.map(e => e.trait_scores[trait] || 0);
        const x = chatHistory.map((_, i) => i + 1);

        return {
            name: trait,
            x: x,
            y: y,
            type: 'scatter',
            mode: 'lines',
            line: {
                color: CHAT_TRAIT_COLORS[idx % CHAT_TRAIT_COLORS.length],
                width: 2,
            },
            hovertemplate: `${trait}: %{y:.3f}<extra></extra>`,
        };
    });

    // Update legend
    legendDiv.innerHTML = traitNames.map((trait, idx) => `
        <span class="legend-item">
            <span class="legend-color" style="background: ${CHAT_TRAIT_COLORS[idx % CHAT_TRAIT_COLORS.length]}"></span>
            ${trait}
        </span>
    `).join('');

    // Update chart
    const layout = window.getPlotlyLayout({
        margin: { l: 50, r: 20, t: 20, b: 40 },
        xaxis: {
            title: 'Token',
            showgrid: true,
        },
        yaxis: {
            title: 'Trait Score',
            showgrid: true,
            zeroline: true,
        },
        showlegend: false,
        hovermode: 'x unified',
    });

    Plotly.react(chartDiv, traces, layout, { responsive: true });
}

// Use global escapeHtml from state.js (no local wrapper to avoid recursion)

// Export for view system
window.renderLiveChat = renderLiveChat;
