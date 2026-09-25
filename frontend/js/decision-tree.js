/**
 * AI Insight Hub — Phase 4 Decision Tree Page Logic
 * Pure Vanilla JavaScript ES6+ (no framework, no chart library).
 *
 * Data flow:
 *   Form input -> client validation -> POST /api/decision-tree/predict
 *              -> Flask -> RRunner -> r_models/decision_tree/predict.R
 *              -> saved model.rds -> JSON prediction -> result card
 *
 * The tree is drawn as hand-built absolutely-positioned nodes joined by an SVG
 * connector layer, matching the zero-dependency approach used elsewhere in the
 * project. Every value rendered comes from the API; nothing is invented here.
 */

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
const CONFIG = {
    API_BASE_URL: 'http://127.0.0.1:5000',
    PREDICT_ENDPOINT: '/api/decision-tree/predict',
    METRICS_ENDPOINT: '/api/decision-tree/metrics',
    TREE_ENDPOINT: '/api/decision-tree/tree',
    SCHEMA_ENDPOINT: '/api/decision-tree/schema',
    REQUEST_TIMEOUT_MS: 20000
};

// Feature keys, matching the trained model's formula and the API contract
const FIELDS = ['age', 'income', 'credit_score', 'existing_loans', 'employment_years'];

// Labels per field, used in client-side validation messages
const FIELD_LABELS = {
    age: 'Age',
    income: 'Annual Income',
    credit_score: 'Credit Score',
    existing_loans: 'Existing Loans',
    employment_years: 'Employment Years'
};

// Client-side rules mirroring the API's domain rules
const FIELD_RULES = {
    age: { exclusiveMin: 0 },
    income: { exclusiveMin: 0 },
    credit_score: { min: 300, max: 900 },
    existing_loans: { min: 0 },
    employment_years: { min: 0 }
};

// A mid-range applicant from within the training distribution
const SAMPLE_APPLICANT = {
    age: 32,
    income: 95000,
    credit_score: 735,
    existing_loans: 1,
    employment_years: 5
};

// The three risk tiers the classifier can return
const RISK_CLASSES = ['LOW', 'MEDIUM', 'HIGH'];

// Ranges reported by the API, filled in on load (min/max may be null)
const featureRanges = {};

// Human labels for the tree's feature names, used when the API omits them
const FALLBACK_FEATURE_LABELS = {
    age: 'Age',
    income: 'Annual Income (INR)',
    credit_score: 'Credit Score',
    existing_loans: 'Existing Loans',
    employment_years: 'Employment Years'
};

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
    resultRiskBadge: document.getElementById('resultRiskBadge'),
    confidenceBlock: document.getElementById('confidenceBlock'),
    resultConfidence: document.getElementById('resultConfidence'),
    confidenceFill: document.getElementById('confidenceFill'),
    probList: document.getElementById('probList'),
    inputSummary: document.getElementById('inputSummary'),
    resultWarnings: document.getElementById('resultWarnings'),
    resultModelName: document.getElementById('resultModelName'),
    resultAlgorithm: document.getElementById('resultAlgorithm'),
    resultTrainedAt: document.getElementById('resultTrainedAt'),
    resultExecTime: document.getElementById('resultExecTime'),

    metricAccuracy: document.getElementById('metricAccuracy'),
    metricPrecision: document.getElementById('metricPrecision'),
    metricRecall: document.getElementById('metricRecall'),
    metricF1: document.getElementById('metricF1'),
    metricBalanced: document.getElementById('metricBalanced'),
    metricTestRows: document.getElementById('metricTestRows'),
    metricTrainRows: document.getElementById('metricTrainRows'),
    performanceNote: document.getElementById('performanceNote'),

    pathPanel: document.getElementById('pathPanel'),
    pathEmpty: document.getElementById('pathEmpty'),
    pathBody: document.getElementById('pathBody'),
    pathList: document.getElementById('pathList'),

    treeCanvas: document.getElementById('treeCanvas'),
    treePlaceholder: document.getElementById('treePlaceholder'),
    treeNodeCount: document.getElementById('treeNodeCount'),
    importanceGrid: document.getElementById('importanceGrid'),

    matrixWrap: document.getElementById('matrixWrap'),
    matrixNote: document.getElementById('matrixNote')
};

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

/** Formats a rupee amount using the Indian numbering system. */
function formatInr(value, decimals = 0) {
    if (typeof value !== 'number' || !isFinite(value)) return '—';
    return '₹' + value.toLocaleString('en-IN', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
    });
}

/** Formats a number to a fixed number of decimals. */
function formatNumber(value, decimals) {
    if (typeof value !== 'number' || !isFinite(value)) return '—';
    return value.toFixed(decimals);
}

/** Formats a probability as a percentage. */
function formatPercent(value) {
    if (typeof value !== 'number' || !isFinite(value)) return '—';
    return (value * 100).toFixed(1) + '%';
}

/** Formats a numeric feature value for display. */
function formatFeatureValue(key, value) {
    if (typeof value !== 'number' || !isFinite(value)) return '—';
    if (key === 'income') return formatInr(value, 0);
    if (key === 'credit_score' || key === 'existing_loans' || key === 'age') {
        return String(Math.round(value));
    }
    return formatNumber(value, 2);
}

/** Returns the DOM id suffix used for a field (credit_score -> Credit_score). */
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
        if (rule && rule.max !== undefined && value > rule.max) {
            errors[key] = `${label} must be at most ${rule.max}.`;
            continue;
        }

        // A value outside the trained range is still classifiable, so it is not
        // a validation error — the request goes through and the API reports it
        // as an extrapolation warning on the result.
        errors[key] = '';
    }

    return errors;
}

/**
 * Lists features whose value falls outside the range the model was trained on.
 * Mirrors the API's `warnings` so the user is told before they submit.
 * @returns {Array} one entry per out-of-range feature
 */
function findExtrapolationWarnings(values) {
    const warnings = [];

    for (const key of FIELDS) {
        const range = featureRanges[key];
        const value = Number(values[key]);
        if (!range || typeof range.min !== 'number' || typeof range.max !== 'number') {
            continue;
        }
        if (isFinite(value) && (value < range.min || value > range.max)) {
            warnings.push({
                field: key,
                label: FIELD_LABELS[key],
                value,
                trainedMin: range.min,
                trainedMax: range.max
            });
        }
    }

    return warnings;
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
        el.resultEmpty.querySelector('.result-empty-title').textContent = 'No assessment yet';
        el.resultEmpty.querySelector('.result-empty-text').textContent =
            'Enter the applicant profile and run the model to see a risk classification.';
    }

    el.resultBody.style.display = state === 'success' ? 'block' : 'none';
    el.resultError.style.display = state === 'error' ? 'block' : 'none';
}

/** Renders the per-class probability bars returned by the model. */
function renderProbabilities(probabilities, predicted) {
    el.probList.innerHTML = '';

    if (!probabilities || typeof probabilities !== 'object') return;

    RISK_CLASSES.forEach(className => {
        const value = probabilities[className];
        if (typeof value !== 'number') return;

        const row = document.createElement('div');
        row.className = 'prob-row';

        const label = document.createElement('span');
        label.className = 'prob-class';
        label.dataset.risk = className;
        label.textContent = className + (className === predicted ? ' ✓' : '');

        const track = document.createElement('div');
        track.className = 'prob-track';

        const fill = document.createElement('div');
        fill.className = 'prob-fill';
        fill.dataset.risk = className;
        track.appendChild(fill);

        const valueSpan = document.createElement('span');
        valueSpan.className = 'prob-value';
        valueSpan.textContent = formatPercent(value);

        row.appendChild(label);
        row.appendChild(track);
        row.appendChild(valueSpan);
        el.probList.appendChild(row);

        // Width is set after insertion so the CSS transition actually runs.
        setTimeout(() => { fill.style.width = (value * 100).toFixed(2) + '%'; }, 0);
    });
}

/** Renders the applicant values the model actually classified. */
function renderInputSummary(inputs) {
    el.inputSummary.innerHTML = '';
    if (!inputs || typeof inputs !== 'object') return;

    FIELDS.forEach(key => {
        if (!(key in inputs)) return;

        const keySpan = document.createElement('span');
        keySpan.className = 'input-summary-key';
        keySpan.textContent = FIELD_LABELS[key];

        const valueSpan = document.createElement('span');
        valueSpan.className = 'input-summary-value';
        valueSpan.textContent = formatFeatureValue(key, inputs[key]);

        el.inputSummary.appendChild(keySpan);
        el.inputSummary.appendChild(valueSpan);
    });
}

/**
 * Renders the API's extrapolation warnings under the input summary.
 * These are advisory, not errors: the model did return a real prediction, it
 * just did so for an applicant outside the range it was trained on.
 * @param {Array} warnings entries from POST /api/decision-tree/predict
 */
function renderWarnings(warnings) {
    const host = el.resultWarnings;
    if (!host) return;

    host.innerHTML = '';

    const list = Array.isArray(warnings) ? warnings : [];
    if (list.length === 0) {
        host.style.display = 'none';
        return;
    }

    const block = document.createElement('div');
    block.className = 'warning-block';

    const title = document.createElement('div');
    title.className = 'warning-title';
    title.textContent = 'Outside the training range';
    block.appendChild(title);

    list.forEach(function (entry) {
        const item = document.createElement('div');
        item.className = 'warning-item';
        item.textContent = typeof entry === 'string' ? entry : (entry.message || '');
        block.appendChild(item);
    });

    host.appendChild(block);
    host.style.display = 'block';
}

/** Renders a successful classification into the premium result card. */
function renderPrediction(data) {
    const risk = data.prediction;

    el.resultRiskBadge.textContent = risk + ' RISK';
    el.resultRiskBadge.dataset.risk = risk;

    // Confidence is shown only when the model actually reported one.
    if (typeof data.confidence === 'number' && isFinite(data.confidence)) {
        el.confidenceBlock.style.display = 'block';
        el.resultConfidence.textContent = formatPercent(data.confidence);
        setTimeout(() => {
            el.confidenceFill.style.width = (data.confidence * 100).toFixed(2) + '%';
        }, 0);
    } else {
        el.confidenceBlock.style.display = 'none';
        el.resultConfidence.textContent = '—';
        el.confidenceFill.style.width = '0%';
    }

    renderProbabilities(data.class_probabilities, risk);
    renderInputSummary(data.inputs);
    renderWarnings(data.warnings);

    el.resultModelName.textContent = data.model || 'Decision Tree';
    el.resultAlgorithm.textContent = data.algorithm || '—';
    el.resultTrainedAt.textContent = data.trained_at || '—';
    el.resultExecTime.textContent = typeof data.execution_time_ms === 'number'
        ? data.execution_time_ms.toFixed(0) + ' ms'
        : '—';

    setResultState('success');
    renderDecisionPath(data.decision_path, risk);
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
// Decision path trace
// ---------------------------------------------------------------------------

/**
 * Renders the real decision path the R model reported for this request.
 * Each step is a node the record actually passed through.
 */
function renderDecisionPath(path, predicted) {
    if (!Array.isArray(path) || path.length === 0) {
        el.pathPanel.dataset.state = 'empty';
        el.pathEmpty.style.display = 'block';
        el.pathBody.style.display = 'none';
        el.pathList.innerHTML = '';
        return;
    }

    el.pathList.innerHTML = '';

    path.forEach(step => {
        const isLeaf = step.branch === 'leaf';

        const li = document.createElement('li');
        li.className = 'path-step';
        li.dataset.kind = isLeaf ? 'leaf' : 'decision';
        if (isLeaf) li.dataset.risk = predicted;

        const node = document.createElement('div');
        node.className = 'path-node';
        node.textContent = String(step.node);
        li.appendChild(node);

        const detail = document.createElement('div');
        detail.className = 'path-detail';

        const condition = document.createElement('div');
        condition.className = 'path-condition';
        condition.textContent = step.condition || '';
        detail.appendChild(condition);

        // For a decision node, show the real value that took this branch.
        if (!isLeaf && typeof step.applicant_value === 'number') {
            const values = document.createElement('div');
            values.className = 'path-values';

            const variableChip = document.createElement('span');
            variableChip.className = 'path-chip';
            variableChip.textContent = 'Node ' + step.node;

            const valueChip = document.createElement('span');
            valueChip.className = 'path-chip';
            const strong = document.createElement('strong');
            strong.textContent = formatFeatureValue(step.variable, step.applicant_value);
            valueChip.appendChild(document.createTextNode(step.variable + ' = '));
            valueChip.appendChild(strong);

            const samplesChip = document.createElement('span');
            samplesChip.className = 'path-chip';
            samplesChip.textContent = 'n = ' + step.samples_at_node;

            values.appendChild(variableChip);
            values.appendChild(valueChip);
            values.appendChild(samplesChip);
            detail.appendChild(values);
        }

        li.appendChild(detail);
        el.pathList.appendChild(li);
    });

    el.pathPanel.dataset.state = 'success';
    el.pathEmpty.style.display = 'none';
    el.pathBody.style.display = 'block';
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
            ['One or more applicant details are outside the range the model was trained on.']);
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
        renderError('Invalid applicant details', result.data.errors);
    } else {
        const message = (result.data && result.data.error) || 'The assessment could not be completed.';
        renderError('Assessment failed', [message]);
    }
}

function setPredictLoading(isLoading) {
    el.btnPredict.disabled = isLoading;
    el.btnPredictLabel.textContent = isLoading ? 'ANALYZING...' : 'ANALYZE FINANCIAL RISK';
    el.predictIcon.classList.toggle('spinning', isLoading);
}

function fillSampleApplicant() {
    for (const key of FIELDS) {
        const suffix = fieldSuffix(key);
        document.getElementById('input' + suffix).value = SAMPLE_APPLICANT[key];
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
        el.metricAccuracy.textContent = '—';
        el.metricPrecision.textContent = '—';
        el.metricRecall.textContent = '—';
        el.metricF1.textContent = '—';
        el.modelStateDisplay.textContent = 'Metrics unavailable';
        el.modelStateDisplay.className = 'meta-value text-rose';
        return;
    }

    const metrics = result.data.metrics;
    const dataset = result.data.dataset || {};

    el.metricAccuracy.textContent = formatNumber(metrics.accuracy, 4);
    el.metricPrecision.textContent = formatNumber(metrics.precision, 4);
    el.metricRecall.textContent = formatNumber(metrics.recall, 4);
    el.metricF1.textContent = formatNumber(metrics.f1_score, 4);
    el.metricBalanced.textContent = formatNumber(metrics.balanced_accuracy, 4);
    el.metricTestRows.textContent = dataset.rows_test !== undefined ? dataset.rows_test : '—';
    el.metricTrainRows.textContent = dataset.rows_train !== undefined ? dataset.rows_train : '—';

    // Explain the accuracy against the majority-class baseline, because on an
    // imbalanced dataset a raw accuracy number is easy to misread.
    const baseline = metrics.majority_class_baseline;
    if (baseline && typeof baseline.accuracy === 'number') {
        el.performanceNote.textContent =
            `Precision, recall and F1 are macro averages across the ${RISK_CLASSES.length} risk tiers. ` +
            `Always predicting the most common class ("${baseline.class}") would score ` +
            `${formatNumber(baseline.accuracy, 4)} accuracy, so the F1 score is the fairer ` +
            `comparison on this imbalanced dataset.`;
    } else {
        el.performanceNote.textContent =
            `Precision, recall and F1 are macro averages across the ${RISK_CLASSES.length} risk tiers.`;
    }

    renderConfusionMatrix(result.data.confusion_matrix, dataset.rows_test);

    el.modelStateDisplay.textContent = 'Model trained & ready';
    el.modelStateDisplay.className = 'meta-value text-emerald';
}

// ---------------------------------------------------------------------------
// Confusion matrix
// ---------------------------------------------------------------------------
function renderConfusionMatrix(matrix, testRows) {
    if (!matrix || !Array.isArray(matrix.labels) || !Array.isArray(matrix.matrix)) {
        el.matrixWrap.innerHTML = '<p class="chart-placeholder">Confusion matrix unavailable.</p>';
        return;
    }

    const labels = matrix.labels;
    const size = labels.length;

    el.matrixWrap.innerHTML = '';

    const grid = document.createElement('div');
    grid.className = 'matrix-grid';
    // One corner cell plus one header per class, then one row per class.
    grid.style.gridTemplateColumns = `70px repeat(${size}, 1fr)`;

    const corner = document.createElement('div');
    corner.className = 'matrix-corner';
    corner.textContent = 'actual \\ predicted';
    grid.appendChild(corner);

    labels.forEach(label => {
        const head = document.createElement('div');
        head.className = 'matrix-head';
        head.dataset.risk = label;
        head.textContent = label;
        grid.appendChild(head);
    });

    let correct = 0;
    let total = 0;

    labels.forEach((rowLabel, r) => {
        const rowHead = document.createElement('div');
        rowHead.className = 'matrix-head';
        rowHead.dataset.risk = rowLabel;
        rowHead.textContent = rowLabel;
        grid.appendChild(rowHead);

        labels.forEach((colLabel, c) => {
            const value = Number(matrix.matrix[r][c]) || 0;
            total += value;
            if (r === c) correct += value;

            const cell = document.createElement('div');
            cell.className = 'matrix-cell';
            cell.dataset.diagonal = String(r === c);
            cell.dataset.zero = String(value === 0);
            cell.textContent = value;

            const caption = document.createElement('span');
            caption.className = 'matrix-cell-label';
            caption.textContent = r === c ? 'correct' : (value === 0 ? 'none' : 'wrong');
            cell.appendChild(caption);

            grid.appendChild(cell);
        });
    });

    el.matrixWrap.appendChild(grid);

    el.matrixNote.textContent = total > 0
        ? `${correct} of ${total} held-out test records were classified correctly. ` +
          `Rows are the actual risk tier; columns are what the model predicted.`
        : 'No test records were available for this matrix.';
}

// ---------------------------------------------------------------------------
// Decision tree visualisation
// ---------------------------------------------------------------------------

/** Measures how deep the tree actually goes, so the canvas can be sized. */
function treeDepth(node) {
    if (!node || node.is_leaf || (!node.left && !node.right)) return 1;
    return 1 + Math.max(treeDepth(node.left), treeDepth(node.right));
}

/** Counts the nodes at each level of the real tree. */
function countPerLevel(node, level, accumulator) {
    if (!node) return accumulator;
    accumulator[level] = (accumulator[level] || 0) + 1;
    if (!node.is_leaf) {
        countPerLevel(node.left, level + 1, accumulator);
        countPerLevel(node.right, level + 1, accumulator);
    }
    return accumulator;
}

/**
 * Renders the real rpart tree returned by the API.
 * The layout is a classic top-down tree: every node the model grew is placed
 * level by level and joined to its children with SVG connectors, so whatever
 * shape rpart produced is exactly what gets drawn.
 */
function renderTree(treeData) {
    const root = treeData && treeData.root;
    if (!root || typeof root !== 'object') {
        el.treeCanvas.innerHTML = '';
        el.treePlaceholder.style.display = 'block';
        el.treePlaceholder.textContent = 'Tree structure unavailable.';
        el.treeNodeCount.textContent = '—';
        return;
    }

    const labels = Object.assign({}, FALLBACK_FEATURE_LABELS, treeData.feature_labels || {});

    const depth = treeDepth(root);
    const perLevel = countPerLevel(root, 0, {});

    const NODE_W = 190;
    const H_GAP = 26;
    const V_GAP = 104;
    const PAD = 20;

    const widest = Math.max.apply(null, Object.values(perLevel));
    const width = Math.max(widest * (NODE_W + H_GAP) + PAD * 2, 640);
    const height = depth * V_GAP + PAD * 2;

    el.treeCanvas.innerHTML = '';
    el.treeCanvas.style.position = 'relative';
    el.treeCanvas.style.width = width + 'px';
    el.treeCanvas.style.height = height + 'px';

    // The SVG connector layer sits behind the node cards.
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('class', 'tree-links');
    svg.setAttribute('width', width);
    svg.setAttribute('height', height);
    el.treeCanvas.appendChild(svg);

    const positions = {};
    let slot = 0;

    (function assign(node, level) {
        if (!node) return;
        const slotsAtLevel = perLevel[level] || 1;
        const span = slotsAtLevel * (NODE_W + H_GAP) - H_GAP;
        const x = PAD + (span - NODE_W) / 2 + slot * (NODE_W + H_GAP);
        const y = PAD + level * V_GAP;
        slot += 1;
        positions[node.id] = { x, y, node };
        if (!node.is_leaf) {
            assign(node.left, level + 1);
            assign(node.right, level + 1);
        }
    })(root, 0);

    // Connectors from each internal node to both of its children.
    Object.values(positions).forEach(pos => {
        const node = pos.node;
        if (node.is_leaf) return;

        [[node.left, node.left_condition], [node.right, node.right_condition]].forEach(pair => {
            const child = pair[0];
            const condition = pair[1];
            if (!child) return;
            const childPos = positions[child.id];
            if (!childPos) return;

            const x1 = pos.x + NODE_W / 2;
            const y1 = pos.y + 78;
            const x2 = childPos.x + NODE_W / 2;
            const y2 = childPos.y;
            const midY = (y1 + y2) / 2;

            const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            path.setAttribute('class', 'tree-link');
            path.setAttribute('d', 'M ' + x1 + ' ' + y1 + ' C ' + x1 + ' ' + midY +
                              ', ' + x2 + ' ' + midY + ', ' + x2 + ' ' + y2);
            svg.appendChild(path);

            const edgeLabel = document.createElement('div');
            edgeLabel.className = 'tree-edge';
            edgeLabel.textContent = condition || '';
            edgeLabel.style.left = ((x1 + x2) / 2) + 'px';
            edgeLabel.style.top = midY + 'px';
            el.treeCanvas.appendChild(edgeLabel);
        });
    });

    // Node cards
    Object.values(positions).forEach(pos => {
        const node = pos.node;
        const card = document.createElement('div');
        card.className = 'tree-node';
        card.dataset.leaf = String(Boolean(node.is_leaf));
        card.dataset.risk = node.predicted_class;
        card.style.left = pos.x + 'px';
        card.style.top = pos.y + 'px';

        const rule = document.createElement('div');
        rule.className = 'tree-node-rule';
        if (node.is_leaf) {
            rule.textContent = 'Node ' + node.id + ' — final class';
        } else {
            const variable = labels[node.variable] || node.variable;
            rule.textContent = variable + ' ' + (node.left_is_below ? '<' : '>=') +
                               ' ' + formatNumber(node.cut, 2);
        }
        card.appendChild(rule);

        const meta = document.createElement('div');
        meta.className = 'tree-node-meta';

        const classSpan = document.createElement('span');
        classSpan.className = 'tree-node-class';
        classSpan.dataset.risk = node.predicted_class;
        classSpan.textContent = node.predicted_class;
        meta.appendChild(classSpan);

        const countSpan = document.createElement('span');
        countSpan.textContent = 'n=' + node.samples;
        meta.appendChild(countSpan);

        card.appendChild(meta);

        // Tooltip carries the node's real class distribution.
        if (node.class_probabilities) {
            const probs = RISK_CLASSES
                .map(c => c + ' ' + formatPercent(node.class_probabilities[c]))
                .join('  ');
            card.title = 'Node ' + node.id + ' — n=' + node.samples + '\n' + probs;
        }

        el.treeCanvas.appendChild(card);
    });

    el.treePlaceholder.style.display = 'none';
    el.treeNodeCount.textContent =
        treeData.total_nodes + ' nodes · ' + treeData.leaf_nodes +
        ' leaves · depth ' + treeDepth(root);
}

/** Renders the model's Gini-gain variable importance. */
function renderImportance(items) {
    el.importanceGrid.innerHTML = '';

    if (!Array.isArray(items) || items.length === 0) {
        el.importanceGrid.innerHTML =
            '<p class="chart-placeholder">Variable importance unavailable.</p>';
        return;
    }

    const maxValue = Math.max.apply(null, items.map(i => Number(i.importance) || 0));

    items.forEach(item => {
        const value = Number(item.importance) || 0;
        const row = document.createElement('div');
        row.className = 'importance-row';

        const name = document.createElement('span');
        name.className = 'importance-name';
        name.textContent = item.feature;

        const track = document.createElement('div');
        track.className = 'importance-track';

        const fill = document.createElement('div');
        fill.className = 'importance-fill';
        track.appendChild(fill);

        const valueSpan = document.createElement('span');
        valueSpan.className = 'importance-value';
        valueSpan.textContent = formatNumber(value, 2);

        row.appendChild(name);
        row.appendChild(track);
        row.appendChild(valueSpan);
        el.importanceGrid.appendChild(row);

        const pct = maxValue > 0 ? (value / maxValue) * 100 : 0;
        setTimeout(() => { fill.style.width = pct.toFixed(2) + '%'; }, 0);
    });
}

async function loadTree() {
    const result = await apiFetch(CONFIG.TREE_ENDPOINT);

    if (!result.ok || !result.data || !result.data.success) {
        el.treeCanvas.innerHTML = '';
        el.treePlaceholder.style.display = 'block';
        el.treePlaceholder.textContent = 'Tree structure unavailable.';
        el.treeNodeCount.textContent = '—';
        el.importanceGrid.innerHTML =
            '<p class="chart-placeholder">Variable importance unavailable.</p>';
        return;
    }

    renderTree(result.data);
    renderImportance(result.data.variable_importance);
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
    el.pathPanel.dataset.state = 'empty';
    el.pathEmpty.style.display = 'block';
    el.pathBody.style.display = 'none';
    el.pathList.innerHTML = '';

    await loadSchema();
    await Promise.all([loadMetrics(), loadTree()]);

    // If the metrics failed to load, surface the banner so the user knows why.
    if (el.modelStateDisplay.textContent === 'Metrics unavailable') {
        el.alertTitle.textContent = 'Model Metrics Unavailable';
        el.alertMessage.innerHTML =
            'Could not load model metrics. Train the model with ' +
            '<code class="code-pill">Rscript r_models/decision_tree/train.R</code> ' +
            'and ensure the Flask backend is running.';
        el.alertBanner.style.display = 'flex';
    }
}

document.addEventListener('DOMContentLoaded', () => {
    el.form.addEventListener('submit', submitPrediction);
    el.btnSample.addEventListener('click', fillSampleApplicant);
    el.btnRetryModel.addEventListener('click', () => {
        el.alertBanner.style.display = 'none';
        initialize();
    });

    initialize();
});
