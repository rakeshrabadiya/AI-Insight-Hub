/**
 * AI Insight Hub — Phase 5 KNN Page Logic
 * Pure Vanilla JavaScript ES6+ (no framework, no chart library).
 *
 * Data flow:
 *   Form input -> client validation -> POST /api/knn/predict
 *              -> Flask -> RRunner -> r_models/knn/predict.R
 *              -> saved model.rds -> JSON prediction -> result card
 *
 * Every value rendered comes from the API. Nothing on this page invents a
 * prediction, a metric, a confidence or a neighbour distance.
 */

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
const CONFIG = {
    API_BASE_URL: 'http://127.0.0.1:5000',
    PREDICT_ENDPOINT: '/api/knn/predict',
    METRICS_ENDPOINT: '/api/knn/metrics',
    CONFIG_ENDPOINT: '/api/knn/config',
    SCHEMA_ENDPOINT: '/api/knn/schema',
    REQUEST_TIMEOUT_MS: 20000
};

// Feature keys, matching the trained model's feature set and the API contract
const FIELDS = ['study_hours', 'attendance', 'previous_score',
                'assignments_completed', 'practical_score'];

// Labels per field, used in client-side validation messages
const FIELD_LABELS = {
    study_hours: 'Study Hours',
    attendance: 'Attendance',
    previous_score: 'Previous Score',
    assignments_completed: 'Assignments Completed',
    practical_score: 'Practical Score'
};

// Client-side rules mirroring the API's domain rules
const FIELD_RULES = {
    study_hours: { min: 0 },
    attendance: { min: 0, max: 100 },
    previous_score: { min: 0, max: 100 },
    assignments_completed: { min: 0 },
    practical_score: { min: 0, max: 100 }
};

// A mid-range student from within the training distribution
const SAMPLE_STUDENT = {
    study_hours: 6.5,
    attendance: 88,
    previous_score: 74,
    assignments_completed: 9,
    practical_score: 81
};

// The three performance tiers the classifier is defined over
const PERFORMANCE_CLASSES = ['LOW', 'MEDIUM', 'HIGH'];

// Ranges and the selected K reported by the API, filled in on load
const featureRanges = {};
let modelConfig = null;

// Human labels for the model's feature names, used when the API omits them
const FALLBACK_FEATURE_LABELS = {
    study_hours: 'Study Hours',
    attendance: 'Attendance (%)',
    previous_score: 'Previous Score (%)',
    assignments_completed: 'Assignments Completed',
    practical_score: 'Practical Score (%)'
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
    selectedKDisplay: document.getElementById('selectedKDisplay'),
    alertBanner: document.getElementById('modelAlertBanner'),
    alertTitle: document.getElementById('modelAlertTitle'),
    alertMessage: document.getElementById('modelAlertMessage'),

    resultPanel: document.getElementById('resultPanel'),
    resultEmpty: document.getElementById('resultEmpty'),
    resultBody: document.getElementById('resultBody'),
    resultError: document.getElementById('resultError'),
    resultErrorTitle: document.getElementById('resultErrorTitle'),
    resultErrorList: document.getElementById('resultErrorList'),
    resultBadge: document.getElementById('resultPerformanceBadge'),
    confidenceBlock: document.getElementById('confidenceBlock'),
    resultConfidence: document.getElementById('resultConfidence'),
    confidenceFill: document.getElementById('confidenceFill'),
    confidenceNote: document.getElementById('confidenceNote'),
    voteList: document.getElementById('voteList'),
    inputSummary: document.getElementById('inputSummary'),
    resultWarnings: document.getElementById('resultWarnings'),
    resultModelName: document.getElementById('resultModelName'),
    resultAlgorithm: document.getElementById('resultAlgorithm'),
    resultSelectedK: document.getElementById('resultSelectedK'),
    resultTrainedAt: document.getElementById('resultTrainedAt'),
    resultExecTime: document.getElementById('resultExecTime'),

    metricAccuracy: document.getElementById('metricAccuracy'),
    metricPrecision: document.getElementById('metricPrecision'),
    metricRecall: document.getElementById('metricRecall'),
    metricF1: document.getElementById('metricF1'),
    metricSupportedF1: document.getElementById('metricSupportedF1'),
    metricTestRows: document.getElementById('metricTestRows'),
    metricTrainRows: document.getElementById('metricTrainRows'),
    performanceNote: document.getElementById('performanceNote'),
    balanceNote: document.getElementById('balanceNote'),

    neighborPanel: document.getElementById('neighborPanel'),
    neighborEmpty: document.getElementById('neighborEmpty'),
    neighborBody: document.getElementById('neighborBody'),
    neighborList: document.getElementById('neighborList'),
    neighborSummary: document.getElementById('neighborSummary'),

    kTableBody: document.getElementById('kTableBody'),
    kTableWrap: document.getElementById('kTableWrap'),

    matrixWrap: document.getElementById('matrixWrap'),
    matrixNote: document.getElementById('matrixNote')
};

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

/** Formats a number to a fixed number of decimals. */
function formatNumber(value, decimals) {
    if (typeof value !== 'number' || !isFinite(value)) return '—';
    return value.toFixed(decimals);
}

/** Formats a probability / proportion as a percentage. */
function formatPercent(value) {
    if (typeof value !== 'number' || !isFinite(value)) return '—';
    return (value * 100).toFixed(1) + '%';
}

/** Formats a numeric feature value for display. */
function formatFeatureValue(key, value) {
    if (typeof value !== 'number' || !isFinite(value)) return '—';
    if (key === 'assignments_completed') return String(Math.round(value));
    return formatNumber(value, 2);
}

/** Returns the DOM id suffix used for a field (study_hours -> Study_hours). */
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
 * Validates the form values against the domain rules.
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
        el.resultEmpty.querySelector('.result-empty-title').textContent = 'No prediction yet';
        el.resultEmpty.querySelector('.result-empty-text').textContent =
            'Enter the student profile and run the model to predict the performance tier.';
    }

    el.resultBody.style.display = state === 'success' ? 'block' : 'none';
    el.resultError.style.display = state === 'error' ? 'block' : 'none';
}

/**
 * Renders the real class counts across the K neighbours.
 * These are integer counts of actual neighbours, not probabilities.
 */
function renderNeighborVotes(distribution) {
    el.voteList.innerHTML = '';
    if (!distribution || typeof distribution !== 'object') return;

    const total = Object.values(distribution)
        .reduce((sum, value) => sum + (Number(value) || 0), 0);

    PERFORMANCE_CLASSES.forEach(className => {
        const count = Number(distribution[className]) || 0;
        const share = total > 0 ? count / total : 0;

        const row = document.createElement('div');
        row.className = 'vote-row';
        row.dataset.performance = className;

        const label = document.createElement('span');
        label.className = 'vote-class';
        label.dataset.performance = className;
        label.textContent = className;

        const track = document.createElement('div');
        track.className = 'vote-track';

        const fill = document.createElement('div');
        fill.className = 'vote-fill';
        fill.dataset.performance = className;
        track.appendChild(fill);

        const value = document.createElement('span');
        value.className = 'vote-value';
        value.textContent = count + (total > 0 ? ' / ' + total : '');

        row.appendChild(label);
        row.appendChild(track);
        row.appendChild(value);
        el.voteList.appendChild(row);

        setTimeout(() => { fill.style.width = (share * 100).toFixed(2) + '%'; }, 0);
    });
}

/** Renders the student values the model actually classified. */
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
 * just did so for a student outside the range it was trained on.
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

/** Renders a successful prediction into the premium result card. */
function renderPrediction(data) {
    const performance = data.prediction;

    el.resultBadge.textContent = performance;
    el.resultBadge.dataset.performance = performance;

    // Confidence is shown only when the model actually reported one.
    if (typeof data.confidence === 'number' && isFinite(data.confidence)) {
        el.confidenceBlock.style.display = 'block';
        el.resultConfidence.textContent = formatPercent(data.confidence);
        if (data.confidence_basis) {
            el.confidenceNote.textContent = data.confidence_basis;
        }
        setTimeout(() => {
            el.confidenceFill.style.width = (data.confidence * 100).toFixed(2) + '%';
        }, 0);
    } else {
        el.confidenceBlock.style.display = 'none';
        el.resultConfidence.textContent = '—';
        el.confidenceFill.style.width = '0%';
    }

    renderNeighborVotes(data.neighbor_class_distribution);
    renderInputSummary(data.inputs);
    renderWarnings(data.warnings);

    el.resultModelName.textContent = data.model || 'K-Nearest Neighbors';
    el.resultAlgorithm.textContent = data.algorithm || '—';
    el.resultSelectedK.textContent = typeof data.k === 'number' ? 'K = ' + data.k : '—';
    el.resultTrainedAt.textContent = data.trained_at || '—';
    el.resultExecTime.textContent = typeof data.execution_time_ms === 'number'
        ? data.execution_time_ms.toFixed(0) + ' ms'
        : '—';

    setResultState('success');
    renderNeighbors(data.neighbors, data.k);
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
// Nearest-neighbour trace
// ---------------------------------------------------------------------------

/**
 * Renders the real nearest training observations the R model reported.
 * Each card shows the neighbour's own feature values and the Euclidean distance
 * the model actually computed — no distances are generated here.
 */
function renderNeighbors(neighbors, k) {
    if (!Array.isArray(neighbors) || neighbors.length === 0) {
        el.neighborPanel.dataset.state = 'empty';
        el.neighborEmpty.style.display = 'block';
        el.neighborBody.style.display = 'none';
        el.neighborList.innerHTML = '';
        el.neighborSummary.textContent = '';
        return;
    }

    el.neighborList.innerHTML = '';

    neighbors.forEach((entry, position) => {
        const card = document.createElement('li');
        card.className = 'neighbor-card';
        card.dataset.performance = entry.neighbor_class;

        const head = document.createElement('div');
        head.className = 'neighbor-head';

        const rank = document.createElement('span');
        rank.className = 'neighbor-rank';
        rank.textContent = 'Neighbor ' + (entry.neighbor !== undefined ? entry.neighbor : position + 1);

        const cls = document.createElement('span');
        cls.className = 'neighbor-class';
        cls.dataset.performance = entry.neighbor_class;
        cls.textContent = entry.neighbor_class;

        const dist = document.createElement('span');
        dist.className = 'neighbor-distance';
        dist.textContent = 'Distance ' + formatNumber(entry.distance, 4);

        head.appendChild(rank);
        head.appendChild(cls);
        head.appendChild(dist);

        const values = document.createElement('div');
        values.className = 'neighbor-values';

        const features = entry.scaled_features || {};
        FIELDS.forEach(key => {
            if (!(key in features)) return;

            const chip = document.createElement('span');
            chip.className = 'neighbor-chip';

            const name = document.createElement('span');
            name.className = 'neighbor-chip-key';
            name.textContent = (FALLBACK_FEATURE_LABELS[key] || FIELD_LABELS[key]) + ':';

            const value = document.createElement('strong');
            value.textContent = formatFeatureValue(key, features[key]);

            chip.appendChild(name);
            chip.appendChild(value);
            values.appendChild(chip);
        });

        card.appendChild(head);
        card.appendChild(values);
        el.neighborList.appendChild(card);
    });

    const kText = typeof k === 'number' ? k : neighbors.length;
    el.neighborSummary.textContent =
        `${kText} nearest training observations, ordered by Euclidean distance. ` +
        `Distances are measured in standardised feature space (mean 0, standard deviation 1).`;

    el.neighborPanel.dataset.state = 'success';
    el.neighborEmpty.style.display = 'none';
    el.neighborBody.style.display = 'block';
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
            ['One or more student details are outside the accepted range.']);
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
        renderError('Invalid student details', result.data.errors);
    } else {
        const message = (result.data && result.data.error) || 'The prediction could not be completed.';
        renderError('Prediction failed', [message]);
    }
}

function setPredictLoading(isLoading) {
    el.btnPredict.disabled = isLoading;
    el.btnPredictLabel.textContent = isLoading ? 'PREDICTING...' : 'PREDICT PERFORMANCE';
    el.predictIcon.classList.toggle('spinning', isLoading);
}

function fillSampleStudent() {
    for (const key of FIELDS) {
        const suffix = fieldSuffix(key);
        document.getElementById('input' + suffix).value = SAMPLE_STUDENT[key];
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
        el.metricSupportedF1.textContent = '—';
        el.modelStateDisplay.textContent = 'Metrics unavailable';
        el.modelStateDisplay.className = 'meta-value text-rose';
        return;
    }

    const metrics = result.data.metrics;
    const dataset = result.data.dataset || {};
    const supported = metrics.macro_supported_only || {};

    el.metricAccuracy.textContent = formatNumber(metrics.accuracy, 4);
    el.metricPrecision.textContent = formatNumber(metrics.precision, 4);
    el.metricRecall.textContent = formatNumber(metrics.recall, 4);
    el.metricF1.textContent = formatNumber(metrics.f1_score, 4);
    el.metricSupportedF1.textContent = formatNumber(supported.f1_score, 4);
    el.metricTestRows.textContent = dataset.rows_test !== undefined ? dataset.rows_test : '—';
    el.metricTrainRows.textContent = dataset.rows_train !== undefined ? dataset.rows_train : '—';

    // Explain the accuracy against the majority-class baseline, because on an
    // imbalanced dataset a raw accuracy number is easy to misread.
    const baseline = metrics.majority_class_baseline;
    if (baseline && typeof baseline.accuracy === 'number') {
        el.performanceNote.textContent =
            `Precision, recall and F1 are macro averages across the ${PERFORMANCE_CLASSES.length} ` +
            `declared tiers. Always predicting the most common class ("${baseline.class}") ` +
            `would score ${formatNumber(baseline.accuracy, 4)} accuracy, so the F1 score is the ` +
            `fairer comparison on this imbalanced dataset.`;
    } else {
        el.performanceNote.textContent =
            `Precision, recall and F1 are macro averages across the ${PERFORMANCE_CLASSES.length} ` +
            `declared tiers.`;
    }

    renderBalanceNote(result.data.class_distribution_note);
    renderKComparison(result.data.k_comparison, result.data.selected_k);
    renderConfusionMatrix(result.data.confusion_matrix, dataset.rows_test);

    el.modelStateDisplay.textContent = 'Model trained & ready';
    el.modelStateDisplay.className = 'meta-value text-emerald';
}

/**
 * Renders the dataset's class-balance caveat. This dataset has a single LOW
 * row, so LOW has no test support and cannot win a K=5 majority vote — the user
 * is told this rather than discovering it as a silent failure to predict LOW.
 */
function renderBalanceNote(note) {
    if (!el.balanceNote) return;

    if (!note || !note.message) {
        el.balanceNote.style.display = 'none';
        return;
    }

    el.balanceNote.textContent = note.message;
    el.balanceNote.style.display = 'block';
}

/** Renders the real K-comparison table produced during training. */
function renderKComparison(comparison, selectedK) {
    el.kTableBody.innerHTML = '';

    if (!Array.isArray(comparison) || comparison.length === 0) {
        el.kTableWrap.innerHTML = '<p class="chart-placeholder">K comparison unavailable.</p>';
        return;
    }

    // Present in ascending K order so the table reads naturally.
    const rows = comparison.slice().sort((a, b) => a.k - b.k);
    const bestF1 = Math.max.apply(null, rows.map(r => Number(r.cv_f1_mean) || 0));

    rows.forEach(entry => {
        const tr = document.createElement('tr');
        if (entry.k === selectedK) tr.dataset.selected = 'true';

        const kCell = document.createElement('td');
        kCell.className = 'k-cell';
        kCell.textContent = entry.k;

        const f1Cell = document.createElement('td');
        f1Cell.className = 'k-value';
        f1Cell.textContent = formatNumber(entry.cv_f1_mean, 4);

        const seCell = document.createElement('td');
        seCell.className = 'k-value';
        seCell.textContent = '± ' + formatNumber(entry.cv_f1_se, 4);

        const barCell = document.createElement('td');
        barCell.className = 'k-bar-cell';

        const track = document.createElement('div');
        track.className = 'k-track';

        const fill = document.createElement('div');
        fill.className = 'k-fill';
        track.appendChild(fill);

        barCell.appendChild(track);

        tr.appendChild(kCell);
        tr.appendChild(f1Cell);
        tr.appendChild(seCell);
        tr.appendChild(barCell);
        el.kTableBody.appendChild(tr);

        const pct = bestF1 > 0 ? ((Number(entry.cv_f1_mean) || 0) / bestF1) * 100 : 0;
        setTimeout(() => { fill.style.width = pct.toFixed(2) + '%'; }, 0);
    });
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
        head.dataset.performance = label;
        head.textContent = label;
        grid.appendChild(head);
    });

    let correct = 0;
    let total = 0;

    labels.forEach((rowLabel, r) => {
        const rowHead = document.createElement('div');
        rowHead.className = 'matrix-head';
        rowHead.dataset.performance = rowLabel;
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
          `Rows are the actual tier; columns are what the model predicted.`
        : 'No test records were available for this matrix.';
}

// ---------------------------------------------------------------------------
// Config + schema loading
// ---------------------------------------------------------------------------

/** Loads the model's selected K and class-balance note. */
async function loadConfig() {
    const result = await apiFetch(CONFIG.CONFIG_ENDPOINT);
    if (!result.ok || !result.data || !result.data.success) return;

    modelConfig = result.data;

    if (el.selectedKDisplay) {
        el.selectedKDisplay.textContent = 'K = ' + result.data.selected_k;
    }
}

/** Loads the per-feature input schema and the dataset ranges. */
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
    el.neighborPanel.dataset.state = 'empty';
    el.neighborEmpty.style.display = 'block';
    el.neighborBody.style.display = 'none';
    el.neighborList.innerHTML = '';

    await loadConfig();
    await loadSchema();
    await loadMetrics();

    // If the metrics failed to load, surface the banner so the user knows why.
    if (el.modelStateDisplay.textContent === 'Metrics unavailable') {
        el.alertTitle.textContent = 'Model Metrics Unavailable';
        el.alertMessage.innerHTML =
            'Could not load model metrics. Train the model with ' +
            '<code class="code-pill">Rscript r_models/knn/train.R</code> ' +
            'and ensure the Flask backend is running.';
        el.alertBanner.style.display = 'flex';
    }
}

document.addEventListener('DOMContentLoaded', () => {
    el.form.addEventListener('submit', submitPrediction);
    el.btnSample.addEventListener('click', fillSampleStudent);
    el.btnRetryModel.addEventListener('click', () => {
        el.alertBanner.style.display = 'none';
        initialize();
    });

    initialize();
});
