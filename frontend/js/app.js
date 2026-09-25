/**
 * AI Insight Hub — Frontend Application Logic
 * Pure Vanilla JavaScript ES6+ implementation
 * Handles API health telemetry, dynamic status updates, R engine execution,
 * and interactive dataset/console diagnostics.
 */

// Configuration Constants
const CONFIG = {
    API_BASE_URL: 'http://127.0.0.1:5000',
    HEALTH_ENDPOINT: '/api/health',
    R_TEST_ENDPOINT: '/api/r-engine/test',
    DATASETS_ENDPOINT: '/api/datasets/validate',
    POLL_INTERVAL_MS: 10000,
    REQUEST_TIMEOUT_MS: 8000
};

// Application State
const state = {
    isOnline: false,
    latencyMs: 0,
    lastChecked: null,
    lastResponseData: null,
    lastHttpStatus: null,
    currentTab: 'health',
    pollTimer: null,
    isPollingActive: true,
    rEngineState: {
        tested: false,
        status: 'NOT TESTED', // 'ONLINE' | 'ERROR' | 'NOT CONNECTED' | 'NOT TESTED'
        version: null,
        data: null
    }
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
    statREngineValue: document.getElementById('statREngineValue'),

    // System Diagnostics Panel — Flask
    flaskBadge: document.getElementById('flaskBadge'),
    flaskStatusDot: document.getElementById('flaskStatusDot'),
    flaskBadgeText: document.getElementById('flaskBadgeText'),
    apiEndpointDisplay: document.getElementById('apiEndpointDisplay'),
    apiLatencyDisplay: document.getElementById('apiLatencyDisplay'),
    apiTimestampDisplay: document.getElementById('apiTimestampDisplay'),
    apiAlertBanner: document.getElementById('apiAlertBanner'),
    btnRetryConnection: document.getElementById('btnRetryConnection'),

    // System Diagnostics Panel — R Engine
    rEngineBadge: document.getElementById('rEngineBadge'),
    rEngineStatusDot: document.getElementById('rEngineStatusDot'),
    rEngineBadgeText: document.getElementById('rEngineBadgeText'),
    rEngineStateDisplay: document.getElementById('rEngineStateDisplay'),
    rEngineVersionDisplay: document.getElementById('rEngineVersionDisplay'),
    btnTestREngine: document.getElementById('btnTestREngine'),
    rTestIcon: document.getElementById('rTestIcon'),

    // Console Diagnostic Viewer
    consoleTargetUrl: document.getElementById('consoleTargetUrl'),
    consoleHttpStatus: document.getElementById('consoleHttpStatus'),
    consoleLatency: document.getElementById('consoleLatency'),
    rawJsonResponse: document.getElementById('rawJsonResponse'),
    chkAutoPoll: document.getElementById('chkAutoPoll'),
    btnCopyJson: document.getElementById('btnCopyJson'),

    // Console Tabs
    tabHealth: document.getElementById('tabHealth'),
    tabREngine: document.getElementById('tabREngine'),
    tabDatasets: document.getElementById('tabDatasets')
};

/**
 * Perform Health Check Ping against the Flask Backend API
 */
async function checkApiHealth() {
    const fullUrl = `${CONFIG.API_BASE_URL}${CONFIG.HEALTH_ENDPOINT}`;
    const startTime = performance.now();

    setLoadingState(true);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), CONFIG.REQUEST_TIMEOUT_MS);

    try {
        const response = await fetch(fullUrl, {
            method: 'GET',
            headers: { 'Accept': 'application/json' },
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

        // Update R Engine status from health if present
        if (data.r_engine === 'online' && !state.rEngineState.tested) {
            updateREngineUI('ONLINE', 'Ready for Subprocess Invocation');
        } else if (data.r_engine === 'not_connected' && !state.rEngineState.tested) {
            updateREngineUI('NOT CONNECTED', 'R Engine Not Detected');
        }

        renderOnlineState(data);

    } catch (error) {
        clearTimeout(timeoutId);
        const endTime = performance.now();
        const latency = Math.round(endTime - startTime);

        const isAbort = error.name === 'AbortError';
        const errorMessage = isAbort
            ? 'Connection timed out after 5 seconds'
            : (error.message || 'Network connection failed');

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

        renderOfflineState(errorMessage);

    } finally {
        setLoadingState(false);
    }
}

/**
 * Test R Engine Execution via POST /api/r-engine/test
 */
async function testREngine() {
    const fullUrl = `${CONFIG.API_BASE_URL}${CONFIG.R_TEST_ENDPOINT}`;
    const startTime = performance.now();

    setRTestLoadingState(true);
    switchConsoleTab('r-engine');

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 15000);

    try {
        const response = await fetch(fullUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify({
                source: "AI Insight Hub Frontend",
                timestamp: new Date().toISOString(),
                action: "verify_r_subsystem"
            }),
            signal: controller.signal
        });

        clearTimeout(timeoutId);
        const endTime = performance.now();
        const latency = Math.round(endTime - startTime);
        const data = await response.json();

        state.rEngineState.tested = true;
        state.rEngineState.data = data;
        state.lastResponseData = data;
        state.lastHttpStatus = `${response.status} ${response.statusText || (response.ok ? 'OK' : 'Error')}`;
        state.latencyMs = latency;

        if (response.ok && data.success) {
            const versionStr = data.data && data.data.r_version ? data.data.r_version : "R Engine Active";
            updateREngineUI('ONLINE', versionStr);
            updateConsoleViewer(data, true, fullUrl, 'POST');
        } else {
            const errorMsg = data.error || data.message || "R Execution Failed";
            updateREngineUI('ERROR', errorMsg);
            updateConsoleViewer(data, false, fullUrl, 'POST');
        }

    } catch (error) {
        clearTimeout(timeoutId);
        const endTime = performance.now();
        const latency = Math.round(endTime - startTime);
        const errorData = {
            success: false,
            engine: "R",
            message: "Unable to communicate with R test endpoint",
            error: error.message || "Network Error / Flask Offline",
            hint: "Make sure Flask backend is running on http://127.0.0.1:5000"
        };

        state.rEngineState.tested = true;
        state.lastResponseData = errorData;
        state.lastHttpStatus = '503 Connection Error';
        state.latencyMs = latency;

        updateREngineUI('ERROR', error.message || 'Connection Error');
        updateConsoleViewer(errorData, false, fullUrl, 'POST');

    } finally {
        setRTestLoadingState(false);
    }
}

/**
 * Validate All Datasets via GET /api/datasets/validate
 */
async function testDatasetsValidation() {
    const fullUrl = `${CONFIG.API_BASE_URL}${CONFIG.DATASETS_ENDPOINT}`;
    const startTime = performance.now();

    switchConsoleTab('datasets');

    try {
        const response = await fetch(fullUrl, {
            method: 'GET',
            headers: { 'Accept': 'application/json' }
        });

        const endTime = performance.now();
        const latency = Math.round(endTime - startTime);
        const data = await response.json();

        state.lastResponseData = data;
        state.lastHttpStatus = `${response.status} ${response.ok ? 'OK' : 'Error'}`;
        state.latencyMs = latency;

        updateConsoleViewer(data, response.ok && data.all_valid, fullUrl, 'GET');

    } catch (error) {
        const errorData = {
            success: false,
            error: error.message || "Failed to validate datasets",
            target: fullUrl
        };
        updateConsoleViewer(errorData, false, fullUrl, 'GET');
    }
}

/**
 * Update R Engine UI Elements Across Dashboard
 */
function updateREngineUI(status, detail) {
    state.rEngineState.status = status;

    if (elements.rEngineBadgeText) {
        elements.rEngineBadgeText.textContent = status;
    }

    if (status === 'ONLINE') {
        if (elements.rEngineBadge) elements.rEngineBadge.className = 'status-badge badge-online';
        if (elements.rEngineStatusDot) elements.rEngineStatusDot.className = 'status-dot dot-emerald';
        if (elements.statREngineValue) {
            elements.statREngineValue.textContent = 'ONLINE';
            elements.statREngineValue.className = 'stat-value text-emerald';
        }
        if (elements.rEngineVersionDisplay) {
            elements.rEngineVersionDisplay.textContent = detail;
            elements.rEngineVersionDisplay.className = 'metric-value text-emerald font-mono';
        }
        if (elements.rEngineStateDisplay) {
            elements.rEngineStateDisplay.textContent = 'Bridge Verified & Operational';
        }
    } else if (status === 'ERROR') {
        if (elements.rEngineBadge) elements.rEngineBadge.className = 'status-badge badge-offline';
        if (elements.rEngineStatusDot) elements.rEngineStatusDot.className = 'status-dot dot-rose';
        if (elements.statREngineValue) {
            elements.statREngineValue.textContent = 'ERROR';
            elements.statREngineValue.className = 'stat-value text-rose';
        }
        if (elements.rEngineVersionDisplay) {
            elements.rEngineVersionDisplay.textContent = 'Execution Failure';
            elements.rEngineVersionDisplay.className = 'metric-value text-rose font-mono';
        }
        if (elements.rEngineStateDisplay) {
            elements.rEngineStateDisplay.textContent = 'R Engine Error Detected';
        }
    } else {
        if (elements.rEngineBadge) elements.rEngineBadge.className = 'status-badge badge-neutral';
        if (elements.rEngineStatusDot) elements.rEngineStatusDot.className = 'status-dot dot-amber';
        if (elements.statREngineValue) {
            elements.statREngineValue.textContent = 'NOT CONNECTED';
            elements.statREngineValue.className = 'stat-value text-amber';
        }
        if (elements.rEngineVersionDisplay) {
            elements.rEngineVersionDisplay.textContent = detail || 'Pending Connection';
            elements.rEngineVersionDisplay.className = 'metric-value text-amber font-mono';
        }
    }
}

/**
 * Render UI when Flask API is ONLINE
 */
function renderOnlineState(data) {
    const timestampStr = new Date().toLocaleTimeString();

    if (elements.headerStatusDot) elements.headerStatusDot.className = 'status-pulse-dot online';
    if (elements.headerApiStatusText) {
        elements.headerApiStatusText.textContent = 'ONLINE';
        elements.headerApiStatusText.className = 'status-pill-value text-emerald';
    }

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

    if (elements.flaskBadge) elements.flaskBadge.className = 'status-badge badge-online';
    if (elements.flaskBadgeText) elements.flaskBadgeText.textContent = 'ONLINE';
    if (elements.apiLatencyDisplay) {
        elements.apiLatencyDisplay.textContent = `${state.latencyMs} ms`;
        elements.apiLatencyDisplay.className = 'metric-value text-emerald';
    }
    if (elements.apiTimestampDisplay) {
        elements.apiTimestampDisplay.textContent = `${timestampStr} (UTC: ${data.timestamp || 'N/A'})`;
    }
    if (elements.apiAlertBanner) elements.apiAlertBanner.style.display = 'none';

    if (state.currentTab === 'health') {
        updateConsoleViewer(data, true, `${CONFIG.API_BASE_URL}${CONFIG.HEALTH_ENDPOINT}`, 'GET');
    }
}

/**
 * Render UI when Flask API is OFFLINE
 */
function renderOfflineState(errorMessage) {
    const timestampStr = new Date().toLocaleTimeString();

    if (elements.headerStatusDot) elements.headerStatusDot.className = 'status-pulse-dot offline';
    if (elements.headerApiStatusText) {
        elements.headerApiStatusText.textContent = 'OFFLINE';
        elements.headerApiStatusText.className = 'status-pill-value text-rose';
    }

    if (elements.statApiStatusValue) {
        elements.statApiStatusValue.textContent = 'OFFLINE';
        elements.statApiStatusValue.className = 'stat-value text-rose';
    }
    if (elements.statApiLatency) elements.statApiLatency.textContent = 'Connection Refused';
    if (elements.statApiIconBox) elements.statApiIconBox.className = 'stat-icon-box stat-icon-rose';

    if (elements.flaskBadge) elements.flaskBadge.className = 'status-badge badge-offline';
    if (elements.flaskBadgeText) elements.flaskBadgeText.textContent = 'OFFLINE';
    if (elements.apiLatencyDisplay) {
        elements.apiLatencyDisplay.textContent = 'Unreachable';
        elements.apiLatencyDisplay.className = 'metric-value text-rose';
    }
    if (elements.apiTimestampDisplay) {
        elements.apiTimestampDisplay.textContent = `${timestampStr} (Failed to connect)`;
    }
    if (elements.apiAlertBanner) elements.apiAlertBanner.style.display = 'flex';

    if (state.currentTab === 'health') {
        updateConsoleViewer(state.lastResponseData, false, `${CONFIG.API_BASE_URL}${CONFIG.HEALTH_ENDPOINT}`, 'GET');
    }
}

/**
 * Update the Live JSON Diagnostics Console
 */
function updateConsoleViewer(data, isSuccess, targetUrl, method) {
    if (elements.consoleTargetUrl) {
        elements.consoleTargetUrl.textContent = targetUrl || `${CONFIG.API_BASE_URL}${CONFIG.HEALTH_ENDPOINT}`;
    }
    if (elements.consoleHttpStatus) {
        elements.consoleHttpStatus.textContent = state.lastHttpStatus || '--';
        elements.consoleHttpStatus.className = isSuccess
            ? 'meta-tag-value font-mono text-emerald'
            : 'meta-tag-value font-mono text-rose';
    }
    if (elements.consoleLatency) {
        elements.consoleLatency.textContent = `${state.latencyMs} ms`;
    }
    if (elements.rawJsonResponse) {
        elements.rawJsonResponse.innerHTML = syntaxHighlightJson(JSON.stringify(data, null, 2));
    }
}

/**
 * Tab Switching Helper
 */
function switchConsoleTab(tabKey) {
    state.currentTab = tabKey;
    [elements.tabHealth, elements.tabREngine, elements.tabDatasets].forEach(tab => {
        if (tab) tab.classList.remove('active');
    });

    if (tabKey === 'health' && elements.tabHealth) elements.tabHealth.classList.add('active');
    if (tabKey === 'r-engine' && elements.tabREngine) elements.tabREngine.classList.add('active');
    if (tabKey === 'datasets' && elements.tabDatasets) elements.tabDatasets.classList.add('active');
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
 * Toggle UI loading spinner during health ping
 */
function setLoadingState(isLoading) {
    if (elements.refreshIcon) {
        if (isLoading) elements.refreshIcon.classList.add('spinning');
        else elements.refreshIcon.classList.remove('spinning');
    }
    if (elements.btnRefreshHealth) elements.btnRefreshHealth.disabled = isLoading;
}

/**
 * Toggle R Engine test button loading spinner
 */
function setRTestLoadingState(isLoading) {
    if (elements.rTestIcon) {
        if (isLoading) elements.rTestIcon.classList.add('spinning');
        else elements.rTestIcon.classList.remove('spinning');
    }
    if (elements.btnTestREngine) elements.btnTestREngine.disabled = isLoading;
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
 * Auto-polling timer
 */
function startPolling() {
    if (state.pollTimer) clearInterval(state.pollTimer);
    state.pollTimer = setInterval(() => {
        if (state.isPollingActive && state.currentTab === 'health') {
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
    // Manual Health Ping
    if (elements.btnRefreshHealth) {
        elements.btnRefreshHealth.addEventListener('click', () => {
            switchConsoleTab('health');
            checkApiHealth();
        });
    }

    // Test R Engine Button
    if (elements.btnTestREngine) {
        elements.btnTestREngine.addEventListener('click', () => {
            testREngine();
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

    // Tab Buttons
    if (elements.tabHealth) {
        elements.tabHealth.addEventListener('click', () => {
            switchConsoleTab('health');
            checkApiHealth();
        });
    }
    if (elements.tabREngine) {
        elements.tabREngine.addEventListener('click', () => {
            testREngine();
        });
    }
    if (elements.tabDatasets) {
        elements.tabDatasets.addEventListener('click', () => {
            testDatasetsValidation();
        });
    }

    // Auto-Poll Checkbox
    if (elements.chkAutoPoll) {
        elements.chkAutoPoll.addEventListener('change', (e) => {
            state.isPollingActive = e.target.checked;
            if (state.isPollingActive) {
                startPolling();
                checkApiHealth();
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
    checkApiHealth();
    startPolling();
});

