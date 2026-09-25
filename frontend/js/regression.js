/**
 * AI Insight Hub — Phase 3 Linear Regression Page Logic
 * Pure Vanilla JavaScript ES6+ (no framework, no chart library).
 *
 * Data flow:
 *   Form input -> client validation -> POST /api/regression/predict
 *              -> Flask -> RRunner -> r_models/regression/predict.R
 *              -> saved model.rds -> JSON prediction -> result card
 *
 * The scatter chart is drawn as hand-built inline SVG, matching the
 * zero-dependency approach already used by the dashboard.
 */

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
const CONFIG = {
    API_BASE_URL: 'http://127.0.0.1:5000',
    PREDICT_ENDPOINT: '/api/regression/predict',
    METRICS_ENDPOINT: '/api/regression/metrics',
    EVALUATION_ENDPOINT: '/api/regression/evaluation',
    SCHEMA_ENDPOINT: '/api/regression/schema',
    REQUEST_TIMEOUT_MS: 20000
};

// Feature keys, matching the trained model's formula and the API contract
const FIELDS = ['area', 'bedrooms', 'bathrooms', 'location_score', 'property_age'];

// Labels per field, used in client-side validation messages
const FIELD_LABELS = {
    area: 'Area (sq.ft)',
    bedrooms: 'Bedrooms',
    bathrooms: 'Bathrooms',
    location_score: 'Location Score',
    property_age: 'Property Age'
};

// Client-side rules mirroring the API's domain rules
const FIELD_RULES = {
    area: { exclusiveMin: 0 },
    bedrooms: { exclusiveMin: 0 },
    bathrooms: { exclusiveMin: 0 },
    property_age: { min: 0 }
};

// A mid-sized property from within the training distribution
const SAMPLE_PROPERTY = {
    area: 1500,
    bedrooms: 3,
    bathrooms: 2,
    location_score: 8,
    property_age: 5
};

// Ranges reported by the API, filled in on load (min/max may be null)
const featureRanges = {};

// ---------------------------------------------------------------------------
// DOM Cache
// ---------------------------------------------------------------------------
const el = {
    form: document.getElementById('predictionForm'),
    btnPredict: document.getElementById('btnPredict'),
    btnPredictLabel: document.getElementById('btnPredictLabel'),
    predictIcon: document.getElementById('predictIcon'),
    btnSample: document.getElementById('btnSample'),
    btnRetryModel: document.getElementById('btnRetryModel'),

    modelStateDisplay: document.getElementById('modelStateDisplay'),
    datasetRangeNote: document.getElementById('datasetRangeNote'),
    alertBanner: document.getElementById('modelAlertBanner'),
    alertTitle: document.getElementById('modelAlertTitle'),
    alertMessage: document.getElementById('modelAlertMessage'),

    resultPanel: document.getElementById('resultPanel'),
    resultEmpty: document.getElementById('resultEmpty'),
    resultBody: document.getElementById('resultBody'),
    resultError: document.getElementById('resultError'),
    resultErrorTitle: document.getElementById('resultErrorTitle'),
    resultErrorList: document.getElementById('resultErrorList'),
    resultPriceLakhs: document.getElementById('resultPriceLakhs'),
    resultRawInr: document.getElementById('resultRawInr'),
    resultModelName: document.getElementById('resultModelName'),
    resultAlgorithm: document.getElementById('resultAlgorithm'),
    resultTrainedAt: document.getElementById('resultTrainedAt'),
    resultExecTime: document.getElementById('resultExecTime'),

    metricR2: document.getElementById('metricR2'),
    metricRmse: document.getElementById('metricRmse'),
    metricMae: document.getElementById('metricMae'),
    metricAdjR2: document.getElementById('metricAdjR2'),
    metricTestRows: document.getElementById('metricTestRows'),
    metricTrainRows: document.getElementById('metricTrainRows'),

    scatterChart: document.getElementById('scatterChart'),
    chartPlaceholder: document.getElementById('chartPlaceholder'),
    chartPointCount: document.getElementById('chartPointCount'),
    chartWrapper: document.getElementById('chartWrapper'),
    chartTooltip: document.getElementById('chartTooltip'),

    coefficientsGrid: document.getElementById('coefficientsGrid')
};

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

/** Formats a rupee amount using the Indian numbering system. */
function formatInr(value, decimals = 2) {
    if (typeof value !== 'number' || !isFinite(value)) return '—';
    return '₹' + value.toLocaleString('en-IN', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
    });
}

/** Converts a rupee amount to lakhs (1 lakh = 100,000) for the headline figure. */
function toLakhs(value) {
    return (value / 100000).toFixed(2);
}

/** Formats a compact rupee amount for metric tiles (e.g. ₹5,646). */
function formatCompactInr(value) {
    if (typeof value !== 'number' || !isFinite(value)) return '—';
    return '₹' + Math.round(value).toLocaleString('en-IN');
}

/** Formats a number to a fixed number of decimals. */
function formatNumber(value, decimals) {
    if (typeof value !== 'number' || !isFinite(value)) return '—';
    return value.toFixed(decimals);
}

/** Returns the DOM id suffix used for a field (area -> Area). */
function fieldSuffix(key) {
    return key.charAt(0).toUpperCase() + key.slice(1);
}

// ---------------------------------------------------------------------------
// Network helper
// ---------------------------------------------------------------------------

/**
 * Performs a fetch with a timeout.
 * Network failures resolve with ok=false rather than throwing.
 * @returns {Promise<{ok: boolean, status: number, data: object|null}>}
 */
async function apiFetch(path, options = {}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), CONFIG.REQUEST_TIMEOUT_MS);
    try {
        const response = await fetch(CONFIG.API_BASE_URL + path, {
            ...options,
            signal: controller.signal
        });
        let data = null;
        try {
            data = await response.json();
        } catch (e) {
            data = null;
        }
        return { ok: response.ok, status: response.status, data };
    } catch (error) {
        const message = error.name === 'AbortError'
            ? 'The request timed out. The R engine may be busy or the backend may be offline.'
            : 'Unable to reach the Flask backend. Make sure it is running on port 5000.';
        return { ok: false, status: 0, data: { error: message } };
    } finally {
        clearTimeout(timer);
    }
}

// ---------------------------------------------------------------------------
// Client-side validation
// ---------------------------------------------------------------------------

/**
 * Validates the form values against the domain rules and the dataset ranges.
 * @returns {Object} field -> error message (empty string when valid)
 */
function validateForm() {
    const errors = {};

    for (const key of FIELDS) {
        const input = document.getElementById('input' + fieldSuffix(key));
        const label = FIELD_LABELS[key];
        const raw = input.value.trim();

        if (raw === '') {
            errors[key] = `${label} is required.`;
            continue;
        }

        const value = Number(raw);
        if (!isFinite(value)) {
            errors[key] = `${label} must be a number.`;
            continue;
        }

        const rule = FIELD_RULES[key];
        if (rule && rule.exclusiveMin !== undefined && value <= rule.exclusiveMin) {
            errors[key] = `${label} must be greater than ${rule.exclusiveMin}.`;
            continue;
        }
        if (rule && rule.min !== undefined && value < rule.min) {
            errors[key] = `${label} must be ${rule.min} or greater.`;
            continue;
        }

        const range = featureRanges[key];
        if (range && typeof range.min === 'number' && typeof range.max === 'number') {
            if (value < range.min || value > range.max) {
                errors[key] =
                    `${label} must be between ${range.min} and ${range.max} ` +
                    `(the range the model was trained on).`;
                continue;
            }
        }

        errors[key] = '';
    }

    return errors;
}

/** Paints per-field validation messages. */
function renderFieldErrors(errors) {
    for (const key of FIELDS) {
        const suffix = fieldSuffix(key);
        const input = document.getElementById('input' + suffix);
        const errorSpan = document.getElementById('error' + suffix);
        const message = errors[key] || '';
        errorSpan.textContent = message;
        input.classList.toggle('has-error', Boolean(message));
    }
}

/** Populates the range hints under each field from the training data. */
function renderRangeHints() {
    for (const key of FIELDS) {
        const hint = document.getElementById('hint' + fieldSuffix(key));
        const range = featureRanges[key];
        if (range && typeof range.min === 'number' && typeof range.max === 'number') {
            hint.textContent = `Model trained on ${range.min} – ${range.max}`;
        }
    }
}

/** Reads the current numeric values from the form. */
function readFormValues() {
    const payload = {};
    for (const key of FIELDS) {
        const input = document.getElementById('input' + fieldSuffix(key));
        payload[key] = Number(input.value);
    }
    return payload;
}

// ---------------------------------------------------------------------------
// Result rendering
// ---------------------------------------------------------------------------

/** Switches the result card between the empty, loading, success and error states. */
function setResultState(state) {
    el.resultPanel.dataset.state = state;

    const isEmptyOrLoading = state === 'empty' || state === 'loading';
    el.resultEmpty.style.display = isEmptyOrLoading ? 'block' : 'none';

    if (state === 'loading') {
        el.resultEmpty.querySelector('.result-empty-title').textContent = 'Running the model...';
        el.resultEmpty.querySelector('.result-empty-text').textContent =
            'Flask is executing predict.R against the saved model.rds.';
    } else {
        el.resultEmpty.querySelector('.result-empty-title').textContent = 'No prediction yet';
        el.resultEmpty.querySelector('.result-empty-text').textContent =
            'Enter the property attributes and run the model to see an estimated price.';
    }

    el.resultBody.style.display = state === 'success' ? 'block' : 'none';
    el.resultError.style.display = state === 'error' ? 'block' : 'none';
}

/** Renders a successful prediction into the premium result card. */
function renderPrediction(data) {
    const price = data.prediction;
    el.resultPriceLakhs.textContent = '₹' + toLakhs(price);
    el.resultRawInr.textContent = formatInr(price, 2);
    el.resultModelName.textContent = data.model || 'Linear Regression';
    el.resultAlgorithm.textContent = data.algorithm || '—';
    el.resultTrainedAt.textContent = data.trained_at || '—';
    el.resultExecTime.textContent = typeof data.execution_time_ms === 'number'
        ? data.execution_time_ms.toFixed(0) + ' ms'
        : '—';
    setResultState('success');
}

/** Renders a validation or server error as a clean list of messages. */
function renderError(title, messages) {
    el.resultErrorTitle.textContent = title;
    el.resultErrorList.innerHTML = '';
    (Array.isArray(messages) ? messages : [messages])
        .filter(Boolean)
        .forEach(message => {
            const li = document.createElement('li');
            li.textContent = message;
            el.resultErrorList.appendChild(li);
        });
    setResultState('error');
}

/**
 * Best-effort mapping of server error strings back onto specific form fields,
 * so the user sees inline errors in the same style as client-side validation.
 */
function mapServerFieldErrors(serverErrors) {
    serverErrors.forEach(message => {
        for (const key of FIELDS) {
            if (message.includes(FIELD_LABELS[key])) {
                const suffix = fieldSuffix(key);
                document.getElementById('error' + suffix).textContent = message;
                document.getElementById('input' + suffix).classList.add('has-error');
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Prediction flow
// ---------------------------------------------------------------------------

async function submitPrediction(event) {
    if (event) event.preventDefault();

    const errors = validateForm();
    renderFieldErrors(errors);

    if (Object.values(errors).some(Boolean)) {
        renderError('Please correct the highlighted fields.',
            ['One or more property attributes are outside the range the model was trained on.']);
        return;
    }

    setResultState('loading');
    setPredictLoading(true);

    const result = await apiFetch(CONFIG.PREDICT_ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
        body: JSON.stringify(readFormValues())
    });

    setPredictLoading(false);

    if (result.ok && result.data && result.data.success) {
        renderPrediction(result.data);
    } else if (result.data && Array.isArray(result.data.errors) && result.data.errors.length) {
        mapServerFieldErrors(result.data.errors);
        renderError('Invalid property details', result.data.errors);
    } else {
        const message = (result.data && result.data.error) || 'The prediction could not be completed.';
        renderError('Prediction failed', [message]);
    }
}

function setPredictLoading(isLoading) {
    el.btnPredict.disabled = isLoading;
    el.btnPredictLabel.textContent = isLoading ? 'PREDICTING...' : 'PREDICT PROPERTY PRICE';
    el.predictIcon.classList.toggle('spinning', isLoading);
}

function fillSampleProperty() {
    for (const key of FIELDS) {
        const suffix = fieldSuffix(key);
        document.getElementById('input' + suffix).value = SAMPLE_PROPERTY[key];
        document.getElementById('input' + suffix).classList.remove('has-error');
        document.getElementById('error' + suffix).textContent = '';
    }
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

async function loadMetrics() {
    const result = await apiFetch(CONFIG.METRICS_ENDPOINT);

    if (!result.ok || !result.data || !result.data.success) {
        el.metricR2.textContent = '—';
        el.metricRmse.textContent = '—';
        el.metricMae.textContent = '—';
        el.modelStateDisplay.textContent = 'Metrics unavailable';
        el.modelStateDisplay.className = 'meta-value text-rose';
        return;
    }

    const metrics = result.data.metrics;
    const dataset = result.data.dataset || {};

    el.metricR2.textContent = formatNumber(metrics.r2, 4);
    el.metricRmse.textContent = formatCompactInr(metrics.rmse);
    el.metricMae.textContent = formatCompactInr(metrics.mae);
    el.metricAdjR2.textContent = formatNumber(metrics.adjusted_r2, 4);
    el.metricTestRows.textContent = dataset.rows_test !== undefined ? dataset.rows_test : '—';
    el.metricTrainRows.textContent = dataset.rows_train !== undefined ? dataset.rows_train : '—';

    el.modelStateDisplay.textContent = 'Model trained & ready';
    el.modelStateDisplay.className = 'meta-value text-emerald';
}

// ---------------------------------------------------------------------------
// Coefficients
// ---------------------------------------------------------------------------

function renderCoefficients(coefficients) {
    el.coefficientsGrid.innerHTML = '';

    if (!Array.isArray(coefficients) || coefficients.length === 0) {
        el.coefficientsGrid.innerHTML =
            '<p class="chart-placeholder">No coefficient data available.</p>';
        return;
    }

    coefficients.forEach(coef => {
        const card = document.createElement('div');
        card.className = 'coefficient-card';

        const term = document.createElement('div');
        term.className = 'coefficient-term';
        term.textContent = coef.term;
        card.appendChild(term);

        const stats = document.createElement('div');
        stats.className = 'coefficient-stats';

        const rows = [
            ['Estimate', formatNumber(coef.estimate, 2)],
            ['Std. Error', formatNumber(coef.std_error, 2)],
            ['t-value', formatNumber(coef.t_value, 2)],
            ['p-value', coef.p_value != null ? formatNumber(coef.p_value, 4) : '—']
        ];

        rows.forEach(row => {
            const rowEl = document.createElement('div');
            rowEl.className = 'coefficient-stat';
            const labelSpan = document.createElement('span');
            labelSpan.textContent = row[0];
            const valueSpan = document.createElement('span');
            valueSpan.className = 'coefficient-stat-value';
            valueSpan.textContent = row[1];
            rowEl.appendChild(labelSpan);
            rowEl.appendChild(valueSpan);
            stats.appendChild(rowEl);
        });
        card.appendChild(stats);

        const isSignificant = coef.p_value != null && coef.p_value < 0.05;
        const badge = document.createElement('span');
        badge.className = 'coefficient-significance ' + (isSignificant ? 'sig' : 'nonsig');
        badge.textContent = isSignificant ? 'Significant (p < 0.05)' : 'Not significant';
        card.appendChild(badge);

        el.coefficientsGrid.appendChild(card);
    });
}

async function loadCoefficients() {
    const result = await apiFetch(CONFIG.METRICS_ENDPOINT);
    if (result.ok && result.data && result.data.success) {
        renderCoefficients(result.data.coefficients);
    } else {
        el.coefficientsGrid.innerHTML =
            '<p class="chart-placeholder">Coefficient data unavailable.</p>';
    }
}

// ---------------------------------------------------------------------------
// Actual vs Predicted scatter chart (hand-built SVG, no chart library)
// ---------------------------------------------------------------------------

const SVG_NS = 'http://www.w3.org/2000/svg';

function svgEl(tag, attrs = {}) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [key, value] of Object.entries(attrs)) {
        node.setAttribute(key, value);
    }
    return node;
}

/**
 * Renders the actual-vs-predicted scatter plot.
 * Both axes are in INR and share a domain, so the dashed diagonal is a true
 * 45-degree perfect-prediction line.
 */
function renderScatterChart(points) {
    el.scatterChart.innerHTML = '';
    if (!Array.isArray(points) || points.length === 0) return;

    const width = 900;
    const height = 520;
    const margin = { top: 20, right: 30, bottom: 70, left: 95 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;

    // Shared domain across both axes, starting at zero for readability.
    const allValues = points.reduce((acc, p) => acc.concat([p.actual, p.predicted]), []);
    const domainMax = Math.max(...allValues) * 1.05;
    const domainMin = 0;

    const xScale = v => margin.left + ((v - domainMin) / (domainMax - domainMin)) * plotWidth;
    const yScale = v => margin.top + plotHeight - ((v - domainMin) / (domainMax - domainMin)) * plotHeight;

    // Grid lines and axis ticks
    const tickCount = 5;
    for (let i = 0; i <= tickCount; i++) {
        const value = domainMin + ((domainMax - domainMin) * i) / tickCount;
        const x = xScale(value);
        const y = yScale(value);

        el.scatterChart.appendChild(svgEl('line', {
            x1: x, y1: margin.top, x2: x, y2: margin.top + plotHeight, class: 'chart-grid-line'
        }));
        el.scatterChart.appendChild(svgEl('line', {
            x1: margin.left, y1: y, x2: margin.left + plotWidth, y2: y, class: 'chart-grid-line'
        }));

        const xLabel = svgEl('text', {
            x: x, y: margin.top + plotHeight + 22, 'text-anchor': 'middle', class: 'chart-tick-text'
        });
        xLabel.textContent = formatCompactInr(value);
        el.scatterChart.appendChild(xLabel);

        const yLabel = svgEl('text', {
            x: margin.left - 12, y: y + 4, 'text-anchor': 'end', class: 'chart-tick-text'
        });
        yLabel.textContent = formatCompactInr(value);
        el.scatterChart.appendChild(yLabel);
    }

    // Axis lines
    el.scatterChart.appendChild(svgEl('line', {
        x1: margin.left, y1: margin.top + plotHeight,
        x2: margin.left + plotWidth, y2: margin.top + plotHeight, class: 'chart-axis-line'
    }));
    el.scatterChart.appendChild(svgEl('line', {
        x1: margin.left, y1: margin.top, x2: margin.left, y2: margin.top + plotHeight, class: 'chart-axis-line'
    }));

    // Axis titles
    const xTitle = svgEl('text', {
        x: margin.left + plotWidth / 2, y: height - 16, 'text-anchor': 'middle', class: 'chart-axis-title'
    });
    xTitle.textContent = 'Actual Price (INR)';
    el.scatterChart.appendChild(xTitle);

    const yMid = margin.top + plotHeight / 2;
    const yTitle = svgEl('text', {
        x: 26, y: yMid, 'text-anchor': 'middle', class: 'chart-axis-title',
        transform: `rotate(-90 26 ${yMid})`
    });
    yTitle.textContent = 'Predicted Price (INR)';
    el.scatterChart.appendChild(yTitle);

    // Perfect-prediction diagonal (y = x)
    el.scatterChart.appendChild(svgEl('line', {
        x1: xScale(domainMin), y1: yScale(domainMin),
        x2: xScale(domainMax), y2: yScale(domainMax), class: 'chart-diagonal'
    }));

    // Data points
    points.forEach(point => {
        const circle = svgEl('circle', {
            cx: xScale(point.actual),
            cy: yScale(point.predicted),
            r: 6,
            class: 'chart-point'
        });
        circle.dataset.actual = point.actual;
        circle.dataset.predicted = point.predicted;
        circle.dataset.residual = point.residual;
        el.scatterChart.appendChild(circle);
    });
}

/** Shows a tooltip for the hovered data point. */
function setupChartTooltip() {
    el.scatterChart.addEventListener('mousemove', event => {
        const target = event.target.closest('.chart-point');
        if (!target) {
            el.chartTooltip.style.display = 'none';
            return;
        }

        const rows = [
            ['Actual', formatInr(Number(target.dataset.actual), 0)],
            ['Predicted', formatInr(Number(target.dataset.predicted), 0)],
            ['Residual', formatInr(Number(target.dataset.residual), 0)]
        ];

        el.chartTooltip.innerHTML = rows.map(row =>
            `<div class="chart-tooltip-row"><span>${row[0]}</span><span>${row[1]}</span></div>`
        ).join('');

        const wrapperRect = el.chartWrapper.getBoundingClientRect();
        el.chartTooltip.style.display = 'block';
        el.chartTooltip.style.left = (event.clientX - wrapperRect.left + 14) + 'px';
        el.chartTooltip.style.top = (event.clientY - wrapperRect.top - 12) + 'px';
    });

    el.scatterChart.addEventListener('mouseleave', () => {
        el.chartTooltip.style.display = 'none';
    });
}

async function loadEvaluation() {
    const result = await apiFetch(CONFIG.EVALUATION_ENDPOINT);

    if (!result.ok || !result.data || !result.data.success) {
        el.chartPlaceholder.textContent = 'Evaluation data unavailable.';
        el.chartPointCount.textContent = '—';
        return;
    }

    renderScatterChart(result.data.points);
    el.chartPlaceholder.style.display = 'none';
    el.chartPointCount.textContent = result.data.count + ' test records';
}

// ---------------------------------------------------------------------------
// Input schema (dataset ranges)
// ---------------------------------------------------------------------------

async function loadSchema() {
    const result = await apiFetch(CONFIG.SCHEMA_ENDPOINT);

    if (!result.ok || !result.data || !Array.isArray(result.data.features)) return;

    result.data.features.forEach(feature => {
        featureRanges[feature.name] = { min: feature.min, max: feature.max };
    });

    renderRangeHints();
    el.datasetRangeNote.textContent = 'Ranges from training data';
}

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

async function initialize() {
    setResultState('empty');
    await loadSchema();
    await Promise.all([loadMetrics(), loadEvaluation(), loadCoefficients()]);

    // If the metrics failed to load, surface the banner so the user knows why.
    if (el.modelStateDisplay.textContent === 'Metrics unavailable') {
        el.alertTitle.textContent = 'Model Metrics Unavailable';
        el.alertMessage.innerHTML =
            'Could not load model metrics. Train the model with ' +
            '<code class="code-pill">Rscript r_models/regression/train.R</code> ' +
            'and ensure the Flask backend is running.';
        el.alertBanner.style.display = 'flex';
    }
}

document.addEventListener('DOMContentLoaded', () => {
    el.form.addEventListener('submit', submitPrediction);
    el.btnSample.addEventListener('click', fillSampleProperty);
    el.btnRetryModel.addEventListener('click', () => {
        el.alertBanner.style.display = 'none';
        initialize();
    });

    setupChartTooltip();
    initialize();
});
