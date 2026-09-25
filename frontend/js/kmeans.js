/**
 * AI Insight Hub — Phase 6 K-Means Page Logic
 * Pure Vanilla JavaScript ES6+ (no framework, no chart library). The two charts
 * (cluster scatter, elbow/silhouette) are drawn on a <canvas> by hand, so the
 * project keeps its zero-dependency frontend.
 *
 * Data flow:
 *   Form input -> client validation -> POST /api/kmeans/predict
 *              -> Flask -> RRunner -> r_models/kmeans/predict.R
 *              -> saved model.rds -> nearest cluster centre -> result card
 *
 *   On load: GET /api/kmeans/metrics   -> stat tiles, K evaluation, profiles
 *           GET /api/kmeans/clusters   -> cluster sizes / labels / centres
 *           GET /api/kmeans/profiles   -> detailed profile tables
 *           GET /api/kmeans/schema     -> per-field ranges for validation
 *
 * Every value rendered comes from the API. Nothing on this page invents a
 * cluster, a metric, a distance or a cluster label.
 */

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
const CONFIG = {
    API_BASE_URL: 'http://127.0.0.1:5000',
    PREDICT_ENDPOINT: '/api/kmeans/predict',
    METRICS_ENDPOINT: '/api/kmeans/metrics',
    CLUSTERS_ENDPOINT: '/api/kmeans/clusters',
    PROFILES_ENDPOINT: '/api/kmeans/profiles',
    SCHEMA_ENDPOINT: '/api/kmeans/schema',
    REQUEST_TIMEOUT_MS: 20000
};

// Clustering features, matching the trained model's feature set and the API
// contract. customer_id is deliberately absent: it is the identifier, not a
// feature, and clustering on it would recover registration order.
const FIELDS = ['age', 'annual_income', 'spending_score', 'purchase_frequency'];

// Labels per field, used in client-side validation messages
const FIELD_LABELS = {
    age: 'Age',
    annual_income: 'Annual Income',
    spending_score: 'Spending Score',
    purchase_frequency: 'Purchase Frequency'
};

// Client-side rules mirroring the API's domain rules
const FIELD_RULES = {
    age: { min: 0, max: 120 },
    annual_income: { min: 0 },
    spending_score: { min: 0, max: 100 },
    purchase_frequency: { min: 0 }
};

// A mid-range customer from within the training distribution
const SAMPLE_CUSTOMER = {
    age: 42,
    annual_income: 85,
    spending_score: 55,
    purchase_frequency: 28
};

// Human labels for the model's feature names, used when the API omits them
const FALLBACK_FEATURE_LABELS = {
    age: 'Age',
    annual_income: 'Annual Income',
    spending_score: 'Spending Score',
    purchase_frequency: 'Purchase Frequency'
};

// Categorical cluster palette. A K-Means cluster has no severity ordering, so
// these are evenly-spaced hues rather than a red->green scale.
const CLUSTER_COLORS = {
    1: { color: '#00f0ff', glow: 'rgba(0, 240, 255, 0.30)' },
    2: { color: '#8b5cf6', glow: 'rgba(139, 92, 246, 0.30)' },
    3: { color: '#f59e0b', glow: 'rgba(245, 158, 11, 0.30)' },
    4: { color: '#10b981', glow: 'rgba(16, 185, 129, 0.30)' },
    5: { color: '#f43f5e', glow: 'rgba(244, 63, 94, 0.30)' },
    6: { color: '#ec4899', glow: 'rgba(236, 72, 153, 0.30)' }
};

const FALLBACK_CLUSTER_COLOR = { color: '#38bdf8', glow: 'rgba(56, 189, 248, 0.30)' };

// Ranges reported by the API, filled in on load
const featureRanges = {};
let lastScatterData = null;

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

    metricRecords: document.getElementById('metricRecords'),
    metricSelectedK: document.getElementById('metricSelectedK'),
    metricClusterCount: document.getElementById('metricClusterCount'),
    metricSilhouette: document.getElementById('metricSilhouette'),
    metricWss: document.getElementById('metricWss'),

    clusterCardGrid: document.getElementById('clusterCardGrid'),

    resultPanel: document.getElementById('resultPanel'),
    resultEmpty: document.getElementById('resultEmpty'),
    resultBody: document.getElementById('resultBody'),
    resultError: document.getElementById('resultError'),
    resultErrorTitle: document.getElementById('resultErrorTitle'),
    resultErrorList: document.getElementById('resultErrorList'),
    resultBadge: document.getElementById('resultClusterBadge'),
    resultClusterLabel: document.getElementById('resultClusterLabel'),
    resultDistance: document.getElementById('resultDistance'),
    separationFill: document.getElementById('separationFill'),
    distanceNote: document.getElementById('distanceNote'),
    distanceList: document.getElementById('distanceList'),
    inputSummary: document.getElementById('inputSummary'),
    resultWarnings: document.getElementById('resultWarnings'),
    resultModelName: document.getElementById('resultModelName'),
    resultClusterSize: document.getElementById('resultClusterSize'),
    resultTrainedAt: document.getElementById('resultTrainedAt'),
    resultExecTime: document.getElementById('resultExecTime'),

    vizSubtitle: document.getElementById('vizSubtitle'),
    vizLegend: document.getElementById('vizLegend'),
    scatterWrap: document.getElementById('scatterWrap'),
    vizNote: document.getElementById('vizNote'),

    kSelectionBasis: document.getElementById('kSelectionBasis'),
    elbowWrap: document.getElementById('elbowWrap'),
    kSelectionNote: document.getElementById('kSelectionNote'),
    kTableWrap: document.getElementById('kTableWrap'),
    kTableBody: document.getElementById('kTableBody'),

    profileGrid: document.getElementById('profileGrid')
};

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

/** Formats a number to a fixed number of decimals. */
function formatNumber(value, decimals) {
    if (typeof value !== 'number' || !isFinite(value)) return '—';
    return value.toFixed(decimals);
}

/** Formats a proportion as a percentage. */
function formatPercent(value) {
    if (typeof value !== 'number' || !isFinite(value)) return '—';
    return (value * 100).toFixed(1) + '%';
}

/** Formats a numeric feature value for display. */
function formatFeatureValue(key, value) {
    if (typeof value !== 'number' || !isFinite(value)) return '—';
    if (key === 'age' || key === 'purchase_frequency') return formatNumber(value, 0);
    return formatNumber(value, 2);
}

/** Returns the DOM id suffix used for a field (annual_income -> Annual_income). */
function fieldSuffix(key) {
    return key.charAt(0).toUpperCase() + key.slice(1);
}

/** Returns the palette entry for a cluster id, falling back safely. */
function clusterPalette(clusterId) {
    return CLUSTER_COLORS[clusterId] || FALLBACK_CLUSTER_COLOR;
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

        // A value outside the trained range is still assignable, so it is not a
        // validation error — the request goes through and the API reports it as
        // an extrapolation warning on the result.
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
        if (hint && range && typeof range.min === 'number' && typeof range.max === 'number') {
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
            'Flask is measuring this customer against the saved cluster centres.';
    } else {
        el.resultEmpty.querySelector('.result-empty-title').textContent = 'No assignment yet';
        el.resultEmpty.querySelector('.result-empty-text').textContent =
            'Enter the customer profile and run the model to find the nearest cluster centre.';
    }

    el.resultBody.style.display = state === 'success' ? 'block' : 'none';
    el.resultError.style.display = state === 'error' ? 'block' : 'none';
}

/**
 * Renders the real measured distance to every cluster centre.
 * The nearest centre is the winner, so each bar is drawn relative to the
 * furthest centre: the winner keeps the longest bar, and the relative widths
 * still show how far apart the centres really are.
 */
function renderDistanceRanking(distances, assignedCluster) {
    el.distanceList.innerHTML = '';
    if (!distances || typeof distances !== 'object') return;

    const entries = Object.keys(distances)
        .map(key => ({
            cluster: Number(String(key).replace(/\D/g, '')) || 0,
            name: key,
            distance: Number(distances[key])
        }))
        .filter(entry => isFinite(entry.distance));

    if (entries.length === 0) return;

    const min = Math.min.apply(null, entries.map(e => e.distance));
    const max = Math.max.apply(null, entries.map(e => e.distance));
    const span = max - min || 1;

    entries.forEach(entry => {
        const row = document.createElement('div');
        row.className = 'vote-row';
        row.dataset.cluster = String(entry.cluster);
        row.dataset.winner = String(entry.cluster === assignedCluster);

        const label = document.createElement('span');
        label.className = 'vote-class';
        label.dataset.cluster = String(entry.cluster);
        label.textContent = entry.name;

        const track = document.createElement('div');
        track.className = 'vote-track';

        const fill = document.createElement('div');
        fill.className = 'vote-fill';
        fill.dataset.cluster = String(entry.cluster);
        track.appendChild(fill);

        const value = document.createElement('span');
        value.className = 'vote-value';
        value.textContent = formatNumber(entry.distance, 4);

        row.appendChild(label);
        row.appendChild(track);
        row.appendChild(value);
        el.distanceList.appendChild(row);

        // A nearer centre gets a longer bar, so the winner reads at a glance.
        const share = 25 + 75 * (1 - (entry.distance - min) / span);
        setTimeout(() => { fill.style.width = share.toFixed(2) + '%'; }, 0);
    });
}

/** Renders the customer values the model actually measured. */
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
 * These are advisory, not errors: the model did return a real assignment, it
 * just did so for a customer outside the range it was trained on.
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

/** Renders a successful cluster assignment into the result card. */
function renderAssignment(data) {
    const cluster = data.cluster;

    el.resultBadge.textContent = 'Cluster ' + cluster;
    el.resultBadge.dataset.cluster = String(cluster);
    el.resultClusterLabel.textContent = data.cluster_label
        ? data.cluster_label
        : 'No descriptive label — this centre sits within 0.5 sd of the population on most features.';

    el.resultDistance.textContent = formatNumber(data.distance, 4);

    // The separation ratio is the distance to the winning centre over the
    // distance to the next-closest one. Lower means a clearer assignment.
    const ratio = data.separation_ratio;
    if (typeof ratio === 'number' && isFinite(ratio)) {
        const share = Math.max(0, Math.min(100, (1 - ratio) * 100));
        el.distanceNote.textContent =
            `Nearest-next-closest ratio ${formatNumber(ratio, 4)}. ` +
            (data.distance_basis || 'Distance measured in standardised feature space.') +
            ' A ratio well below 1 means this centre is clearly the closest.';
        setTimeout(() => { el.separationFill.style.width = share.toFixed(2) + '%'; }, 0);
    } else {
        el.distanceNote.textContent = data.distance_basis || '';
        el.separationFill.style.width = '0%';
    }

    renderDistanceRanking(data.distances, cluster);
    renderInputSummary(data.inputs);
    renderWarnings(data.warnings);

    el.resultModelName.textContent = data.model || 'K-Means';
    el.resultClusterSize.textContent = typeof data.cluster_size === 'number'
        ? data.cluster_size + ' customers'
        : '—';
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
// Assignment flow
// ---------------------------------------------------------------------------
async function submitAssignment(event) {
    if (event) event.preventDefault();

    const errors = validateForm();
    renderFieldErrors(errors);

    if (Object.values(errors).some(Boolean)) {
        renderError('Please correct the highlighted fields.',
            ['One or more customer details are outside the accepted range.']);
        return;
    }

    setResultState('loading');
    setAssignLoading(true);

    const result = await apiFetch(CONFIG.PREDICT_ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
        body: JSON.stringify(readFormValues())
    });

    setAssignLoading(false);

    if (result.ok && result.data && result.data.success) {
        renderAssignment(result.data);
    } else if (result.data && Array.isArray(result.data.errors) && result.data.errors.length) {
        mapServerFieldErrors(result.data.errors);
        renderError('Invalid customer details', result.data.errors);
    } else {
        const message = (result.data && result.data.error) || 'The assignment could not be completed.';
        renderError('Assignment failed', [message]);
    }
}

function setAssignLoading(isLoading) {
    el.btnPredict.disabled = isLoading;
    el.btnPredictLabel.textContent = isLoading ? 'ASSIGNING...' : 'ASSIGN CUSTOMER TO CLUSTER';
    el.predictIcon.classList.toggle('spinning', isLoading);
}

function fillSampleCustomer() {
    for (const key of FIELDS) {
        const suffix = fieldSuffix(key);
        document.getElementById('input' + suffix).value = SAMPLE_CUSTOMER[key];
        document.getElementById('input' + suffix).classList.remove('has-error');
        document.getElementById('error' + suffix).textContent = '';
    }
}

// ---------------------------------------------------------------------------
// Cluster overview cards
// ---------------------------------------------------------------------------

/** Renders one summary card per cluster from the real profile statistics. */
function renderClusterCards(profiles) {
    el.clusterCardGrid.innerHTML = '';
    if (!Array.isArray(profiles) || profiles.length === 0) return;

    profiles.forEach(profile => {
        const clusterId = Number(profile.cluster);
        const palette = clusterPalette(clusterId);

        const card = document.createElement('div');
        card.className = 'cluster-card';
        card.dataset.cluster = String(clusterId);
        card.style.setProperty('--cluster-color', palette.color);
        card.style.setProperty('--cluster-glow', palette.glow);

        const head = document.createElement('div');
        head.className = 'cluster-card-head';

        const name = document.createElement('span');
        name.className = 'cluster-card-name';
        name.textContent = 'CLUSTER ' + clusterId;

        const share = document.createElement('span');
        share.className = 'cluster-card-share';
        share.textContent = formatPercent(profile.size_share);

        head.appendChild(name);
        head.appendChild(share);

        const count = document.createElement('div');
        count.className = 'cluster-card-count';
        count.textContent = typeof profile.size === 'number' ? String(profile.size) : '—';

        const countLabel = document.createElement('div');
        countLabel.className = 'cluster-card-count-label';
        countLabel.textContent = 'Customers';

        // The generated label, plus the individual descriptors behind it.
        const label = document.createElement('div');
        label.className = 'cluster-card-count-label';
        label.textContent = profile.label || ('Cluster ' + clusterId);

        const chips = document.createElement('div');
        chips.className = 'cluster-card-descriptors';
        const descriptors = Array.isArray(profile.descriptors) ? profile.descriptors : [];
        const chipValues = descriptors.length > 0
            ? descriptors
            : [profile.label || ('Cluster ' + clusterId)];
        chipValues.forEach(text => {
            const chip = document.createElement('span');
            chip.className = 'descriptor-chip';
            if (profile.label_is_generic) chip.classList.add('descriptor-chip-generic');
            chip.textContent = text;
            chips.appendChild(chip);
        });

        const bar = document.createElement('div');
        bar.className = 'cluster-card-bar';
        const barFill = document.createElement('div');
        barFill.className = 'cluster-card-bar-fill';
        bar.appendChild(barFill);

        card.appendChild(head);
        card.appendChild(count);
        card.appendChild(countLabel);
        card.appendChild(label);
        card.appendChild(chips);
        card.appendChild(bar);
        el.clusterCardGrid.appendChild(card);

        const pct = (Number(profile.size_share) || 0) * 100;
        setTimeout(() => { barFill.style.width = pct.toFixed(2) + '%'; }, 0);
    });
}

// ---------------------------------------------------------------------------
// Cluster scatter plot (hand-drawn canvas — no chart library)
// ---------------------------------------------------------------------------

/**
 * Draws every customer at its PCA position, coloured by its real cluster, with
 * the cluster centres overlaid as larger ringed markers.
 *
 * The 2-D position comes from the PCA projection the training script computed
 * FOR DISPLAY. The clustering itself ran on the four scaled features; this is
 * only a way to look at a four-dimensional result on a two-dimensional screen.
 */
function drawClusterScatter(canvas, points, centers) {
    if (!canvas || !points || !Array.isArray(points.PC1) || points.PC1.length === 0) return;

    const width = canvas.clientWidth || 900;
    const height = 460;
    const padding = { top: 24, right: 24, bottom: 46, left: 56 };

    canvas.width = width * (window.devicePixelRatio || 1);
    canvas.height = height * (window.devicePixelRatio || 1);
    canvas.style.height = height + 'px';

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.scale(window.devicePixelRatio || 1, window.devicePixelRatio || 1);
    ctx.clearRect(0, 0, width, height);

    const xs = points.PC1;
    const ys = points.PC2;
    const clusters = points.cluster;

    let minX = Math.min.apply(null, xs);
    let maxX = Math.max.apply(null, xs);
    let minY = Math.min.apply(null, ys);
    let maxY = Math.max.apply(null, ys);

    // Pull the axis range out to include the projected centres too.
    if (Array.isArray(centers)) {
        centers.forEach(function (entry) {
            if (typeof entry.PC1 === 'number') {
                minX = Math.min(minX, entry.PC1);
                maxX = Math.max(maxX, entry.PC1);
            }
            if (typeof entry.PC2 === 'number') {
                minY = Math.min(minY, entry.PC2);
                maxY = Math.max(maxY, entry.PC2);
            }
        });
    }

    // A little padding stops edge points from touching the frame.
    const padX = (maxX - minX) * 0.06 || 1;
    const padY = (maxY - minY) * 0.06 || 1;
    minX -= padX; maxX += padX; minY -= padY; maxY += padY;

    const plotW = width - padding.left - padding.right;
    const plotH = height - padding.top - padding.bottom;

    const scaleX = value => padding.left + ((value - minX) / (maxX - minX)) * plotW;
    const scaleY = value => padding.top + plotH - ((value - minY) / (maxY - minY)) * plotH;

    // --- Grid + axes -------------------------------------------------------
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.10)';
    ctx.lineWidth = 1;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.fillStyle = 'rgba(148, 163, 184, 0.75)';

    for (let step = 0; step <= 4; step++) {
        const gy = padding.top + (plotH / 4) * step;
        ctx.beginPath();
        ctx.moveTo(padding.left, gy);
        ctx.lineTo(padding.left + plotW, gy);
        ctx.stroke();

        const value = maxY - ((maxY - minY) / 4) * step;
        ctx.textAlign = 'right';
        ctx.fillText(formatNumber(value, 1), padding.left - 8, gy + 3);
    }

    for (let step = 0; step <= 5; step++) {
        const gx = padding.left + (plotW / 5) * step;
        ctx.beginPath();
        ctx.moveTo(gx, padding.top);
        ctx.lineTo(gx, padding.top + plotH);
        ctx.stroke();

        const value = minX + ((maxX - minX) / 5) * step;
        ctx.textAlign = 'center';
        ctx.fillText(formatNumber(value, 1), gx, padding.top + plotH + 18);
    }

    ctx.textAlign = 'center';
    ctx.fillStyle = 'rgba(148, 163, 184, 0.9)';
    ctx.fillText('PC1', padding.left + plotW / 2, height - 8);
    ctx.save();
    ctx.translate(14, padding.top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('PC2', 0, 0);
    ctx.restore();

    // --- Customer points ---------------------------------------------------
    ctx.globalAlpha = 0.62;
    for (let i = 0; i < xs.length; i++) {
        const palette = clusterPalette(Number(clusters[i]));
        ctx.beginPath();
        ctx.arc(scaleX(xs[i]), scaleY(ys[i]), 3.6, 0, Math.PI * 2);
        ctx.fillStyle = palette.color;
        ctx.fill();
    }
    ctx.globalAlpha = 1;

    // --- Cluster centres ---------------------------------------------------
    if (Array.isArray(centers)) {
        centers.forEach(function (entry) {
            const clusterId = Number(entry.cluster);
            const palette = clusterPalette(clusterId);
            const cx = scaleX(entry.PC1);
            const cy = scaleY(entry.PC2);

            // Halo, so the centre reads as an average rather than a point.
            ctx.beginPath();
            ctx.arc(cx, cy, 15, 0, Math.PI * 2);
            ctx.fillStyle = palette.glow;
            ctx.fill();

            ctx.beginPath();
            ctx.arc(cx, cy, 6.5, 0, Math.PI * 2);
            ctx.fillStyle = palette.color;
            ctx.fill();
            ctx.lineWidth = 2;
            ctx.strokeStyle = '#060913';
            ctx.stroke();

            ctx.font = 'bold 11px "JetBrains Mono", monospace';
            ctx.fillStyle = palette.color;
            ctx.textAlign = 'left';
            ctx.fillText('C' + clusterId, cx + 11, cy + 4);
        });
    }
}

/** Builds the scatter plot and its legend from the real cluster artifacts. */
function renderClusterScatter(data) {
    const viz = data.visualization || {};
    const points = viz.points;
    const centers = data.cluster_centers_projected || [];

    el.scatterWrap.innerHTML = '';
    el.vizLegend.innerHTML = '';

    if (!points || !Array.isArray(points.PC1) || points.PC1.length === 0) {
        el.scatterWrap.innerHTML = '<p class="chart-placeholder">Cluster projection unavailable.</p>';
        return;
    }

    const canvas = document.createElement('canvas');
    canvas.className = 'scatter-canvas';
    canvas.setAttribute('role', 'img');
    canvas.setAttribute('aria-label',
        'Scatter plot of customers projected onto two principal components, coloured by cluster');
    el.scatterWrap.appendChild(canvas);

    drawClusterScatter(canvas, points, centers);

    // --- Legend, built from the real per-cluster sizes and labels ----------
    const sizes = data.cluster_sizes || {};
    const labels = data.cluster_labels || {};
    Object.keys(sizes)
        .map(key => ({ cluster: Number(key), size: Number(sizes[key]) }))
        .sort((a, b) => a.cluster - b.cluster)
        .forEach(function (entry) {
            const palette = clusterPalette(entry.cluster);
            const item = document.createElement('div');
            item.className = 'viz-legend-item';
            item.dataset.cluster = String(entry.cluster);

            const swatch = document.createElement('span');
            swatch.className = 'viz-legend-swatch';
            swatch.style.setProperty('--swatch-color', palette.color);
            swatch.style.setProperty('--swatch-glow', palette.glow);

            const name = document.createElement('span');
            name.textContent = 'Cluster ' + entry.cluster;

            const count = document.createElement('span');
            count.className = 'viz-legend-count';
            count.textContent = '(' + entry.size + ')';

            const label = document.createElement('span');
            label.className = 'viz-legend-count';
            label.textContent = labels[String(entry.cluster)] || '';

            item.appendChild(swatch);
            item.appendChild(name);
            item.appendChild(count);
            item.appendChild(label);
            el.vizLegend.appendChild(item);
        });

    // --- Explanation -------------------------------------------------------
    const explained = viz.explained_variance_ratio || {};
    const pct = key => {
        const value = Number(explained[key]);
        return isFinite(value) ? formatPercent(value) : '—';
    };

    el.vizSubtitle.textContent =
        'PCA projection of the clustered customers — display only, the clustering used all four features';

    el.vizNote.textContent =
        `Each dot is one real customer from customers.csv, plotted at its position on the two leading ` +
        `principal components (PC1 explains ${pct('PC1')}, PC2 ${pct('PC2')}, ` +
        `${formatPercent(Number(viz.total_explained_variance) || 0)} together). ` +
        `The ringed markers are the cluster centres. ` +
        `This projection is for display only — the clustering itself ran on all four scaled features, ` +
        `never on these two components, and the distances quoted under Customer Segmentation are the ` +
        `four-feature distances the model actually minimises.`;
}

/** Redraws the scatter plot on window resize so it stays crisp. */
function attachScatterResize() {
    if (typeof window === 'undefined' || !window.addEventListener) return;
    window.addEventListener('resize', function () {
        const canvas = el.scatterWrap ? el.scatterWrap.querySelector('canvas') : null;
        if (!canvas || !lastScatterData) return;
        drawClusterScatter(canvas, lastScatterData.points, lastScatterData.centers);
    });
}

// ---------------------------------------------------------------------------
// K evaluation chart (elbow + silhouette)
// ---------------------------------------------------------------------------

/**
 * Draws two curves against K: the WSS elbow curve and the average silhouette.
 * The elbow is the largest proportional WSS fall; the silhouette peak is the K
 * with the best separation. Both are computed in R during training.
 */
function drawElbowChart(canvas, comparison, selectedK, elbowK) {
    if (!canvas || !Array.isArray(comparison) || comparison.length === 0) return;

    const width = canvas.clientWidth || 620;
    const height = 320;
    const padding = { top: 20, right: 20, bottom: 42, left: 56 };

    canvas.width = width * (window.devicePixelRatio || 1);
    canvas.height = height * (window.devicePixelRatio || 1);
    canvas.style.height = height + 'px';

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.scale(window.devicePixelRatio || 1, window.devicePixelRatio || 1);
    ctx.clearRect(0, 0, width, height);

    const rows = comparison.slice().sort((a, b) => a.k - b.k);
    const ks = rows.map(r => Number(r.k));
    const wssValues = rows.map(r => Number(r.wss));
    const silhouetteValues = rows.map(r => Number(r.silhouette_mean));

    const plotW = width - padding.left - padding.right;
    const plotH = height - padding.top - padding.bottom;

    const xFor = k => padding.left + ((k - ks[0]) / (ks[ks.length - 1] - ks[0] || 1)) * plotW;

    // --- Left axis: WSS, which always falls as K rises ----------------------
    const wssMax = Math.max.apply(null, wssValues);
    const wssMin = Math.min.apply(null, wssValues);
    const yWss = value => padding.top + ((value - wssMin) / (wssMax - wssMin || 1)) * plotH;

    // --- Right axis: silhouette, on its -1..1 scale -------------------------
    const ySil = value => padding.top + ((1 - value) / 2) * plotH;

    // --- Grid --------------------------------------------------------------
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.10)';
    ctx.lineWidth = 1;
    ctx.font = '10px "JetBrains Mono", monospace';

    for (let step = 0; step <= 4; step++) {
        const gy = padding.top + (plotH / 4) * step;
        ctx.beginPath();
        ctx.moveTo(padding.left, gy);
        ctx.lineTo(padding.left + plotW, gy);
        ctx.stroke();

        ctx.fillStyle = 'rgba(148, 163, 184, 0.75)';
        ctx.textAlign = 'right';
        ctx.fillText(formatNumber(wssMax - ((wssMax - wssMin) / 4) * step, 0), padding.left - 8, gy + 3);

        ctx.fillStyle = 'rgba(139, 92, 246, 0.8)';
        ctx.textAlign = 'left';
        ctx.fillText(formatNumber(1 - step / 2, 1), padding.left + plotW + 6, gy + 3);
    }

    // --- K tick labels -----------------------------------------------------
    ctx.fillStyle = 'rgba(148, 163, 184, 0.85)';
    ctx.textAlign = 'center';
    ks.forEach(k => ctx.fillText(String(k), xFor(k), padding.top + plotH + 18));

    ctx.fillStyle = 'rgba(148, 163, 184, 0.9)';
    ctx.fillText('Number of Clusters (K)', padding.left + plotW / 2, height - 6);

    // --- Elbow marker: a vertical guide at the elbow K ---------------------
    if (typeof elbowK === 'number' && ks.indexOf(elbowK) !== -1) {
        ctx.save();
        ctx.setLineDash([4, 4]);
        ctx.strokeStyle = 'rgba(245, 158, 11, 0.65)';
        ctx.beginPath();
        ctx.moveTo(xFor(elbowK), padding.top);
        ctx.lineTo(xFor(elbowK), padding.top + plotH);
        ctx.stroke();
        ctx.restore();

        ctx.fillStyle = 'rgba(245, 158, 11, 0.95)';
        ctx.font = 'bold 10px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        ctx.fillText('elbow', xFor(elbowK), padding.top - 6);
    }

    // --- WSS curve (the elbow method) --------------------------------------
    ctx.strokeStyle = '#00f0ff';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    wssValues.forEach(function (value, index) {
        const x = xFor(ks[index]);
        const y = yWss(value);
        if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();

    wssValues.forEach(function (value, index) {
        ctx.beginPath();
        ctx.arc(xFor(ks[index]), yWss(value), 4, 0, Math.PI * 2);
        ctx.fillStyle = ks[index] === elbowK ? '#f59e0b' : '#00f0ff';
        ctx.fill();
    });

    // --- Silhouette curve (the optimum) ------------------------------------
    ctx.strokeStyle = '#8b5cf6';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    let started = false;
    silhouetteValues.forEach(function (value, index) {
        if (!isFinite(value)) return;
        const x = xFor(ks[index]);
        const y = ySil(value);
        if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
    });
    ctx.stroke();

    silhouetteValues.forEach(function (value, index) {
        if (!isFinite(value)) return;
        ctx.beginPath();
        ctx.arc(xFor(ks[index]), ySil(value), 4, 0, Math.PI * 2);
        ctx.fillStyle = ks[index] === selectedK ? '#10b981' : '#8b5cf6';
        ctx.fill();
    });

    // --- Selected K guide --------------------------------------------------
    if (typeof selectedK === 'number' && ks.indexOf(selectedK) !== -1) {
        ctx.strokeStyle = 'rgba(16, 185, 129, 0.75)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(xFor(selectedK), padding.top);
        ctx.lineTo(xFor(selectedK), padding.top + plotH);
        ctx.stroke();

        ctx.fillStyle = 'rgba(16, 185, 129, 0.95)';
        ctx.font = 'bold 10px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        ctx.fillText('K = ' + selectedK, xFor(selectedK), height - 24);
    }

    // --- Series legend -----------------------------------------------------
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillStyle = '#00f0ff';
    ctx.fillText('— WSS (left axis)', padding.left + 4, padding.top + 10);
    ctx.fillStyle = '#8b5cf6';
    ctx.fillText('— Silhouette (right axis)', padding.left + 4, padding.top + 24);
}

/** Renders the K-comparison table from the real training results. */
function renderKTable(comparison, selectedK, elbowK) {
    el.kTableBody.innerHTML = '';

    if (!Array.isArray(comparison) || comparison.length === 0) {
        el.kTableWrap.innerHTML = '<p class="chart-placeholder">K comparison unavailable.</p>';
        return;
    }

    comparison.slice().sort((a, b) => a.k - b.k).forEach(entry => {
        const tr = document.createElement('tr');
        if (entry.k === selectedK) tr.dataset.selected = 'true';

        const kCell = document.createElement('td');
        kCell.className = 'k-cell';
        kCell.textContent = entry.k;
        if (entry.k === elbowK) {
            const mark = document.createElement('div');
            mark.className = 'k-row-mark';
            mark.textContent = 'elbow';
            kCell.appendChild(mark);
        }
        if (entry.k === selectedK) {
            const mark = document.createElement('div');
            mark.className = 'k-row-mark';
            mark.textContent = 'selected';
            kCell.appendChild(mark);
        }

        const wssCell = document.createElement('td');
        wssCell.className = 'k-value';
        wssCell.textContent = formatNumber(entry.wss, 2);

        const silCell = document.createElement('td');
        silCell.className = 'k-value';
        silCell.textContent = formatNumber(entry.silhouette_mean, 4);

        const betweenCell = document.createElement('td');
        betweenCell.className = 'k-value';
        betweenCell.textContent = formatNumber(entry.between_ratio, 4);

        tr.appendChild(kCell);
        tr.appendChild(wssCell);
        tr.appendChild(silCell);
        tr.appendChild(betweenCell);
        el.kTableBody.appendChild(tr);
    });
}

// ---------------------------------------------------------------------------
// Cluster profile tables
// ---------------------------------------------------------------------------

/** Renders one profile card per cluster from the real profile statistics. */
function renderProfiles(data) {
    el.profileGrid.innerHTML = '';

    const profiles = data.profiles || [];
    if (!Array.isArray(profiles) || profiles.length === 0) {
        el.profileGrid.innerHTML = '<p class="chart-placeholder">Cluster profiles unavailable.</p>';
        return;
    }

    const features = Array.isArray(data.features) && data.features.length
        ? data.features
        : FIELDS;

    profiles.forEach(profile => {
        const clusterId = Number(profile.cluster);
        const palette = clusterPalette(clusterId);

        const card = document.createElement('div');
        card.className = 'profile-card';
        card.dataset.cluster = String(clusterId);
        card.style.setProperty('--cluster-color', palette.color);
        card.style.setProperty('--cluster-glow', palette.glow);

        // --- Head ------------------------------------------------------------
        const head = document.createElement('div');
        head.className = 'profile-card-head';

        const dot = document.createElement('span');
        dot.className = 'profile-card-dot';

        const name = document.createElement('span');
        name.className = 'profile-card-name';
        name.textContent = 'Cluster ' + clusterId;

        const segment = document.createElement('span');
        segment.className = 'profile-card-segment';
        segment.textContent = profile.segment || 'Segment';

        head.appendChild(dot);
        head.appendChild(name);
        head.appendChild(segment);

        const label = document.createElement('div');
        label.className = 'profile-card-label';
        label.textContent = profile.label || ('Cluster ' + clusterId);

        const meta = document.createElement('div');
        meta.className = 'profile-card-meta';
        meta.textContent =
            `${typeof profile.size === 'number' ? profile.size : '—'} customers · ` +
            `${formatPercent(profile.size_share)} of the dataset`;

        // --- Feature table: member mean, centre, and the offset in sd units --
        const table = document.createElement('table');
        table.className = 'profile-table';

        const thead = document.createElement('thead');
        const headRow = document.createElement('tr');
        ['Feature', 'Cluster mean', 'Centre (original)', 'vs population'].forEach(function (title) {
            const th = document.createElement('th');
            th.textContent = title;
            headRow.appendChild(th);
        });
        thead.appendChild(headRow);
        table.appendChild(thead);

        const tbody = document.createElement('tbody');
        features.forEach(function (key) {
            const mean = Number((profile.original_mean_features || {})[key]);
            const centre = Number((profile.center_original || {})[key]);
            const offset = Number((profile.center_vs_population_sd || {})[key]);

            const tr = document.createElement('tr');
            // The sign of the sd offset colours the last column: above or below
            // the population, or close enough to call neither.
            tr.dataset.sign = !isFinite(offset) ? 'neutral'
                : (offset >= 0.5 ? 'high' : (offset <= -0.5 ? 'low' : 'neutral'));

            const nameCell = document.createElement('td');
            nameCell.textContent = FIELD_LABELS[key] || FALLBACK_FEATURE_LABELS[key] || key;

            const meanCell = document.createElement('td');
            meanCell.textContent = formatFeatureValue(key, mean);

            const centreCell = document.createElement('td');
            centreCell.textContent = formatFeatureValue(key, centre);

            const offsetCell = document.createElement('td');
            offsetCell.textContent = isFinite(offset)
                ? (offset >= 0 ? '+' : '') + formatNumber(offset, 2) + ' sd'
                : '—';

            tr.appendChild(nameCell);
            tr.appendChild(meanCell);
            tr.appendChild(centreCell);
            tr.appendChild(offsetCell);
            tbody.appendChild(tr);
        });
        table.appendChild(tbody);

        // --- Footer ----------------------------------------------------------
        const footer = document.createElement('div');
        footer.className = 'profile-footer';

        const wssLabel = document.createElement('span');
        wssLabel.textContent = 'Share of WSS';
        const wssValue = document.createElement('span');
        wssValue.className = 'profile-footer-value';
        wssValue.textContent = formatPercent(profile.wss_share);

        const distLabel = document.createElement('span');
        distLabel.textContent = 'Mean distance to centre';
        const distValue = document.createElement('span');
        distValue.className = 'profile-footer-value';
        distValue.textContent = formatNumber(profile.mean_distance_to_center, 4);

        footer.appendChild(wssLabel);
        footer.appendChild(wssValue);
        footer.appendChild(distLabel);
        footer.appendChild(distValue);

        card.appendChild(head);
        card.appendChild(label);
        card.appendChild(meta);
        card.appendChild(table);
        card.appendChild(footer);
        el.profileGrid.appendChild(card);
    });
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

/** Loads the metrics and paints the stat tiles, K evaluation and cluster cards. */
async function loadMetrics() {
    const result = await apiFetch(CONFIG.METRICS_ENDPOINT);

    if (!result.ok || !result.data || !result.data.success) {
        el.metricRecords.textContent = '—';
        el.metricSelectedK.textContent = '—';
        el.metricClusterCount.textContent = '—';
        el.metricSilhouette.textContent = '—';
        el.metricWss.textContent = '—';
        el.modelStateDisplay.textContent = 'Metrics unavailable';
        el.modelStateDisplay.className = 'meta-value text-rose';
        return;
    }

    const data = result.data;
    const metrics = data.metrics || {};
    const dataset = data.dataset || {};
    const selection = data.k_selection || {};
    const comparison = Array.isArray(data.k_comparison) ? data.k_comparison : [];

    // --- Stat tiles --------------------------------------------------------
    el.metricRecords.textContent = dataset.rows_clean !== undefined ? String(dataset.rows_clean) : '—';
    el.metricSelectedK.textContent = data.selected_k !== undefined ? String(data.selected_k) : '—';
    el.metricClusterCount.textContent = metrics.cluster_count !== undefined
        ? String(metrics.cluster_count) : '—';
    el.metricSilhouette.textContent = formatNumber(metrics.silhouette_score, 4);
    el.metricWss.textContent = formatNumber(metrics.wss, 4);
    el.selectedKDisplay.textContent = 'K = ' + data.selected_k;

    // --- K evaluation ------------------------------------------------------
    el.kSelectionBasis.textContent = selection.basis || 'elbow + silhouette';
    el.kSelectionNote.textContent =
        (selection.reason || '') +
        ' ' + ((selection.explain && selection.explain.elbow) || '') +
        ' ' + ((selection.explain && selection.explain.silhouette) || '');

    el.elbowWrap.innerHTML = '';
    if (comparison.length > 0) {
        const canvas = document.createElement('canvas');
        canvas.className = 'elbow-canvas';
        canvas.setAttribute('role', 'img');
        canvas.setAttribute('aria-label',
            'K versus within-cluster sum of squares and average silhouette width');
        el.elbowWrap.appendChild(canvas);
        drawElbowChart(canvas, comparison, data.selected_k, selection.elbow_k);
    } else {
        el.elbowWrap.innerHTML = '<p class="chart-placeholder">K evaluation unavailable.</p>';
    }

    renderKTable(comparison, data.selected_k, selection.elbow_k);

    // --- Cluster overview cards (built from the same artifact) --------------
    renderClusterCards(Array.isArray(data.cluster_profiles) ? data.cluster_profiles : data.clusters);

    el.modelStateDisplay.textContent = 'Model trained & ready';
    el.modelStateDisplay.className = 'meta-value text-emerald';
}

/** Loads the per-customer cluster assignments and draws the scatter plot. */
async function loadClusters() {
    const result = await apiFetch(CONFIG.CLUSTERS_ENDPOINT);
    if (!result.ok || !result.data || !result.data.success) return;

    // The scatter points live in metrics.json (where the PCA projection was
    // computed), so both artifacts are needed to draw the picture.
    const metricsResult = await apiFetch(CONFIG.METRICS_ENDPOINT);
    if (metricsResult.ok && metricsResult.data && metricsResult.data.success) {
        const visualization = metricsResult.data.visualization || {};
        lastScatterData = {
            points: visualization.points,
            centers: result.data.cluster_centers_projected || []
        };
        renderClusterScatter({
            visualization,
            cluster_sizes: result.data.cluster_sizes,
            cluster_labels: result.data.cluster_labels,
            cluster_centers_projected: result.data.cluster_centers_projected
        });
    }
}

/** Loads the detailed per-cluster profiles. */
async function loadProfiles() {
    const result = await apiFetch(CONFIG.PROFILES_ENDPOINT);
    if (!result.ok || !result.data || !result.data.success) return;
    renderProfiles(result.data);
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
    attachScatterResize();

    await loadSchema();
    await loadMetrics();
    await loadClusters();
    await loadProfiles();

    // If the metrics failed to load, surface the banner so the user knows why.
    if (el.modelStateDisplay.textContent === 'Metrics unavailable') {
        el.alertTitle.textContent = 'Model Metrics Unavailable';
        el.alertMessage.innerHTML =
            'Could not load model metrics. Train the model with ' +
            '<code class="code-pill">Rscript r_models/kmeans/train.R</code> ' +
            'and ensure the Flask backend is running.';
        el.alertBanner.style.display = 'flex';
    }
}

document.addEventListener('DOMContentLoaded', () => {
    el.form.addEventListener('submit', submitAssignment);
    el.btnSample.addEventListener('click', fillSampleCustomer);
    el.btnRetryModel.addEventListener('click', () => {
        el.alertBanner.style.display = 'none';
        initialize();
    });

    initialize();
});
