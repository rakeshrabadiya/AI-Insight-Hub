/**
 * AI Insight Hub — Frontend Application Logic
 * Pure Vanilla JavaScript ES6+ implementation
 * Handles API health telemetry, dynamic status updates, and interactive diagnostics.
 */

// Configuration Constants
const CONFIG = {
    API_BASE_URL: 'http://127.0.0.1:5000',
    HEALTH_ENDPOINT: '/api/health',
    POLL_INTERVAL_MS: 10000,
    REQUEST_TIMEOUT_MS: 5000
};

// Application State
const state = {
    isOnline: false,
    latencyMs: 0,
    lastChecked: null,
    lastResponseData: null,
    lastHttpStatus: null,
    pollTimer: null,
    isPollingActive: true
};

// DOM Elements Cache
const elements = {
    // Header
    headerStatusDot: document.getElementById('headerStatusDot'),
    headerApiStatusText: document.getElementById('headerApiStatusText'),
    btnRefreshHealth: document.getElementById('btnRefreshHealth'),
    refreshIcon: document.getElementById('refreshIcon'),

    // Statistics Cards
    statApiStatusValue: document.getElementById('statApiStatusValue'),
    statApiLatency: document.getElementById('statApiLatency'),
    statApiIconBox: document.getElementById('statApiIconBox'),

    // System Diagnostics Panel
    flaskBadge: document.getElementById('flaskBadge'),
    flaskStatusDot: document.getElementById('flaskStatusDot'),
    flaskBadgeText: document.getElementById('flaskBadgeText'),
    apiEndpointDisplay: document.getElementById('apiEndpointDisplay'),
    apiLatencyDisplay: document.getElementById('apiLatencyDisplay'),
    apiTimestampDisplay: document.getElementById('apiTimestampDisplay'),
    apiAlertBanner: document.getElementById('apiAlertBanner'),
    btnRetryConnection: document.getElementById('btnRetryConnection'),

    // Console Diagnostic Viewer
    consoleTargetUrl: document.getElementById('consoleTargetUrl'),
    consoleHttpStatus: document.getElementById('consoleHttpStatus'),
    consoleLatency: document.getElementById('consoleLatency'),
    rawJsonResponse: document.getElementById('rawJsonResponse'),
    chkAutoPoll: document.getElementById('chkAutoPoll'),
    btnCopyJson: document.getElementById('btnCopyJson')
};

/**
 * Perform Health Check Ping against the Flask Backend API
 */
async function checkApiHealth() {
    const fullUrl = `${CONFIG.API_BASE_URL}${CONFIG.HEALTH_ENDPOINT}`;
    const startTime = performance.now();

    // Start UI loading animation
    setLoadingState(true);

    // Abort controller for network timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), CONFIG.REQUEST_TIMEOUT_MS);

    try {
        const response = await fetch(fullUrl, {
            method: 'GET',
            headers: {
                'Accept': 'application/json'
            },
            signal: controller.signal
        });

        clearTimeout(timeoutId);
        const endTime = performance.now();
        const latency = Math.round(endTime - startTime);

        if (!response.ok) {
            throw new Error(`HTTP Error ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();

        // Update State
        state.isOnline = true;
        state.latencyMs = latency;
        state.lastChecked = new Date();
        state.lastResponseData = data;
        state.lastHttpStatus = `${response.status} OK`;

        // Update UI for Success
        renderOnlineState(data);

    } catch (error) {
        clearTimeout(timeoutId);
        const endTime = performance.now();
        const latency = Math.round(endTime - startTime);

        const isAbort = error.name === 'AbortError';
        const errorMessage = isAbort
            ? 'Connection timed out after 5 seconds'
            : (error.message || 'Network connection failed');

        // Update State
        state.isOnline = false;
        state.latencyMs = latency;
        state.lastChecked = new Date();
        state.lastResponseData = {
            status: "offline",
            error: errorMessage,
            hint: "Ensure Flask is running via: python backend/app.py",
            target: fullUrl
        };
        state.lastHttpStatus = isAbort ? '408 Timeout' : '0 Offline / Refused';

        // Update UI for Failure
        renderOfflineState(errorMessage);

    } finally {
        setLoadingState(false);
    }
}

/**
 * Render UI when Flask API is ONLINE
 */
function renderOnlineState(data) {
    const timestampStr = new Date().toLocaleTimeString();

    // 1. Header Updates
    if (elements.headerStatusDot) {
        elements.headerStatusDot.className = 'status-pulse-dot online';
    }
    if (elements.headerApiStatusText) {
        elements.headerApiStatusText.textContent = 'ONLINE';
        elements.headerApiStatusText.className = 'status-pill-value text-emerald';
    }

    // 2. Statistics Card
    if (elements.statApiStatusValue) {
        elements.statApiStatusValue.textContent = 'ONLINE';
        elements.statApiStatusValue.className = 'stat-value text-emerald';
    }
    if (elements.statApiLatency) {
        elements.statApiLatency.textContent = `Latency: ${state.latencyMs} ms • 200 OK`;
    }
    if (elements.statApiIconBox) {
        elements.statApiIconBox.className = 'stat-icon-box stat-icon-emerald';
    }

    // 3. System Status Panel
    if (elements.flaskBadge) {
        elements.flaskBadge.className = 'status-badge badge-online';
    }
    if (elements.flaskBadgeText) {
        elements.flaskBadgeText.textContent = 'ONLINE';
    }
    if (elements.apiLatencyDisplay) {
        elements.apiLatencyDisplay.textContent = `${state.latencyMs} ms`;
        elements.apiLatencyDisplay.className = 'metric-value text-emerald';
    }
    if (elements.apiTimestampDisplay) {
        elements.apiTimestampDisplay.textContent = `${timestampStr} (UTC: ${data.timestamp || 'N/A'})`;
    }
    if (elements.apiAlertBanner) {
        elements.apiAlertBanner.style.display = 'none';
    }

    // 4. Live Console Viewer
    updateConsoleViewer(data, true);
}

/**
 * Render UI when Flask API is OFFLINE
 */
function renderOfflineState(errorMessage) {
    const timestampStr = new Date().toLocaleTimeString();

    // 1. Header Updates
    if (elements.headerStatusDot) {
        elements.headerStatusDot.className = 'status-pulse-dot offline';
    }
    if (elements.headerApiStatusText) {
        elements.headerApiStatusText.textContent = 'OFFLINE';
        elements.headerApiStatusText.className = 'status-pill-value text-rose';
    }

    // 2. Statistics Card
    if (elements.statApiStatusValue) {
        elements.statApiStatusValue.textContent = 'OFFLINE';
        elements.statApiStatusValue.className = 'stat-value text-rose';
    }
    if (elements.statApiLatency) {
        elements.statApiLatency.textContent = 'Connection Refused';
    }
    if (elements.statApiIconBox) {
        elements.statApiIconBox.className = 'stat-icon-box stat-icon-rose';
    }

    // 3. System Status Panel
    if (elements.flaskBadge) {
        elements.flaskBadge.className = 'status-badge badge-offline';
    }
    if (elements.flaskBadgeText) {
        elements.flaskBadgeText.textContent = 'OFFLINE';
    }
    if (elements.apiLatencyDisplay) {
        elements.apiLatencyDisplay.textContent = 'Unreachable';
        elements.apiLatencyDisplay.className = 'metric-value text-rose';
    }
    if (elements.apiTimestampDisplay) {
        elements.apiTimestampDisplay.textContent = `${timestampStr} (Failed to connect)`;
    }
    if (elements.apiAlertBanner) {
        elements.apiAlertBanner.style.display = 'flex';
    }

    // 4. Live Console Viewer
    updateConsoleViewer(state.lastResponseData, false);
}

/**
 * Update the Live JSON Diagnostics Console
 */
function updateConsoleViewer(data, isSuccess) {
    if (elements.consoleTargetUrl) {
        elements.consoleTargetUrl.textContent = `${CONFIG.API_BASE_URL}${CONFIG.HEALTH_ENDPOINT}`;
    }
    if (elements.consoleHttpStatus) {
        elements.consoleHttpStatus.textContent = state.lastHttpStatus;
        elements.consoleHttpStatus.className = isSuccess
            ? 'meta-tag-value font-mono text-emerald'
            : 'meta-tag-value font-mono text-rose';
    }
    if (elements.consoleLatency) {
        elements.consoleLatency.textContent = isSuccess ? `${state.latencyMs} ms` : 'N/A';
    }
    if (elements.rawJsonResponse) {
        elements.rawJsonResponse.innerHTML = syntaxHighlightJson(JSON.stringify(data, null, 2));
    }
}

/**
 * Formats and syntax highlights JSON for visual terminal aesthetic
 */
function syntaxHighlightJson(jsonString) {
    if (!jsonString) return '';
    const escaped = jsonString
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');

    return escaped.replace(
        /("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g,
        function (match) {
            let cls = 'json-number';
            if (/^"/.test(match)) {
                if (/:$/.test(match)) {
                    cls = 'json-key';
                } else {
                    cls = 'json-string';
                }
            } else if (/true|false/.test(match)) {
                cls = 'json-boolean';
            } else if (/null/.test(match)) {
                cls = 'json-null';
            }
            return '<span class="' + cls + '">' + match + '</span>';
        }
    );
}

/**
 * Toggle UI loading spinner during ping
 */
function setLoadingState(isLoading) {
    if (elements.refreshIcon) {
        if (isLoading) {
            elements.refreshIcon.classList.add('spinning');
        } else {
            elements.refreshIcon.classList.remove('spinning');
        }
    }
    if (elements.btnRefreshHealth) {
        elements.btnRefreshHealth.disabled = isLoading;
    }
}

/**
 * Copy formatted JSON payload to clipboard
 */
async function copyJsonPayload() {
    if (!state.lastResponseData) return;
    try {
        const text = JSON.stringify(state.lastResponseData, null, 2);
        await navigator.clipboard.writeText(text);
        if (elements.btnCopyJson) {
            const originalText = elements.btnCopyJson.textContent;
            elements.btnCopyJson.textContent = 'Copied!';
            elements.btnCopyJson.classList.add('text-emerald');
            setTimeout(() => {
                elements.btnCopyJson.textContent = originalText;
                elements.btnCopyJson.classList.remove('text-emerald');
            }, 1800);
        }
    } catch (err) {
        console.error('Failed to copy JSON:', err);
    }
}

/**
 * Initialize Auto-polling timer
 */
function startPolling() {
    if (state.pollTimer) clearInterval(state.pollTimer);
    state.pollTimer = setInterval(() => {
        if (state.isPollingActive) {
            checkApiHealth();
        }
    }, CONFIG.POLL_INTERVAL_MS);
}

function stopPolling() {
    if (state.pollTimer) {
        clearInterval(state.pollTimer);
        state.pollTimer = null;
    }
}

/**
 * Bind DOM Event Listeners
 */
function bindEventListeners() {
    // Manual Refresh Button
    if (elements.btnRefreshHealth) {
        elements.btnRefreshHealth.addEventListener('click', () => {
            checkApiHealth();
        });
    }

    // Retry Button in Banner
    if (elements.btnRetryConnection) {
        elements.btnRetryConnection.addEventListener('click', () => {
            checkApiHealth();
        });
    }

    // Copy JSON Button
    if (elements.btnCopyJson) {
        elements.btnCopyJson.addEventListener('click', copyJsonPayload);
    }

    // Auto-Poll Checkbox
    if (elements.chkAutoPoll) {
        elements.chkAutoPoll.addEventListener('change', (e) => {
            state.isPollingActive = e.target.checked;
            if (state.isPollingActive) {
                startPolling();
                checkApiHealth(); // trigger instant check on toggle enable
            } else {
                stopPolling();
            }
        });
    }
}

/**
 * Application Initialization
 */
document.addEventListener('DOMContentLoaded', () => {
    bindEventListeners();
    checkApiHealth(); // Initial Ping on load
    startPolling();   // Start recurring health checks
});
