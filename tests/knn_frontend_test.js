/**
 * Headless behavioural test for frontend/js/knn.js
 *
 * Runs the real page script in Node against a minimal DOM shim, with fetch
 * pointed at the live Flask server, and asserts the actual user-facing
 * behaviour: formatting, client-side validation, the prediction flow, the
 * nearest-neighbour trace, the K-comparison table, the metrics and the confusion
 * matrix.
 *
 * Run:  node tests/knn_frontend_test.js
 * Requires the Flask backend to be running on 127.0.0.1:5000.
 */
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const API = process.env.API_BASE || 'http://127.0.0.1:5000';
const SCRIPT = path.join(__dirname, '..', 'frontend', 'js', 'knn.js');

// --- Minimal DOM shim -------------------------------------------------------
const elements = new Map();

function makeElement(id) {
    const el = {
        id,
        value: '',
        _text: '',
        innerHTML: '',
        className: '',
        style: {},
        title: '',
        dataset: {},
        attributes: {},
        classList: {
            _set: new Set(),
            add(c) { this._set.add(c); },
            remove(c) { this._set.delete(c); },
            toggle(c, on) { on ? this._set.add(c) : this._set.delete(c); },
            contains(c) { return this._set.has(c); }
        },
        setAttribute(name, value) { this.attributes[name] = value; },
        addEventListener() {},
        appendChild(child) { el.children.push(child); },
        children: [],
        getBoundingClientRect: () => ({ left: 0, top: 0 }),
        querySelector: () => makeElement('sub'),
        closest: () => null
    };
    // A real DOM's textContent is always a string, so coerce assignments.
    Object.defineProperty(el, 'textContent', {
        get() { return el._text; },
        set(v) { el._text = v === null || v === undefined ? '' : String(v); }
    });
    return el;
}

for (const id of [
    'predictionForm', 'btnPredict', 'btnPredictLabel', 'predictIcon', 'btnSample',
    'btnRetryModel', 'modelStateDisplay', 'datasetRangeNote', 'selectedKDisplay',
    'modelAlertBanner', 'modelAlertTitle', 'modelAlertMessage', 'resultPanel',
    'resultEmpty', 'resultBody', 'resultError', 'resultErrorTitle', 'resultErrorList',
    'resultPerformanceBadge', 'confidenceBlock', 'resultConfidence', 'confidenceFill',
    'confidenceNote', 'voteList', 'inputSummary', 'resultWarnings', 'resultModelName',
    'resultAlgorithm', 'resultSelectedK', 'resultTrainedAt', 'resultExecTime',
    'metricAccuracy', 'metricPrecision', 'metricRecall', 'metricF1',
    'metricSupportedF1', 'metricTestRows', 'metricTrainRows', 'performanceNote',
    'balanceNote', 'neighborPanel', 'neighborEmpty', 'neighborBody', 'neighborList',
    'neighborSummary', 'kTableBody', 'kTableWrap', 'matrixWrap', 'matrixNote'
]) {
    elements.set(id, makeElement(id));
}

// Form fields + their hint/error spans
const FIELDS = ['study_hours', 'attendance', 'previous_score',
                'assignments_completed', 'practical_score'];
for (const key of FIELDS) {
    const suffix = key.charAt(0).toUpperCase() + key.slice(1);
    elements.set('input' + suffix, makeElement('input' + suffix));
    elements.set('hint' + suffix, makeElement('hint' + suffix));
    elements.set('error' + suffix, makeElement('error' + suffix));
}

const documentStub = {
    getElementById: id => elements.get(id) || null,
    createElement: tag => makeElement('created-' + tag),
    createElementNS: (ns, tag) => makeElement('svg-' + tag),
    createTextNode: text => ({ nodeValue: text }),
    addEventListener: () => {}   // skip the page's own auto-initialisation
};

// --- Load the real script ---------------------------------------------------
const source = fs.readFileSync(SCRIPT, 'utf8');
const context = {
    document: documentStub,
    window: {},
    console,
    fetch: (url, opts) => fetch(url, opts),
    AbortController,
    setTimeout,
    clearTimeout,
    Object, Array, Number, String, Math, JSON, isFinite, Date, Promise
};
context.globalThis = context;
vm.createContext(context);
vm.runInContext(source, context);

// --- Test harness -----------------------------------------------------------
let passed = 0;
let failed = 0;

function check(name, condition, detail) {
    if (condition) {
        passed++;
        console.log('  PASS  ' + name);
    } else {
        failed++;
        console.log('  FAIL  ' + name + (detail ? ' -> ' + detail : ''));
    }
}

function eq(name, actual, expected) {
    check(name, actual === expected,
        'expected ' + JSON.stringify(expected) + ', got ' + JSON.stringify(actual));
}

function setForm(values) {
    for (const key of FIELDS) {
        const suffix = key.charAt(0).toUpperCase() + key.slice(1);
        elements.get('input' + suffix).value = values[key] !== undefined ? String(values[key]) : '';
        elements.get('error' + suffix).textContent = '';
    }
}

const VALID = {
    study_hours: 6.5, attendance: 88, previous_score: 74,
    assignments_completed: 9, practical_score: 81
};

(async () => {
    console.log('\n=== AI Insight Hub - Frontend Logic Test (knn.js) ===\n');

    // --- 1. Formatting -----------------------------------------------------
    console.log('[1] Formatting');
    eq('formatNumber 4dp', context.formatNumber(0.6028, 4), '0.6028');
    eq('formatPercent converts a proportion', context.formatPercent(0.8), '80.0%');
    eq('formatPercent on 1.0', context.formatPercent(1), '100.0%');
    eq('formatNumber handles a non-number safely', context.formatNumber(null, 2), '—');
    eq('formatPercent handles a non-number safely', context.formatPercent('x'), '—');
    eq('assignments are shown as a whole number',
        context.formatFeatureValue('assignments_completed', 9), '9');
    eq('study hours keep two decimals',
        context.formatFeatureValue('study_hours', 6.5), '6.50');
    eq('attendance is shown to 2dp', context.formatFeatureValue('attendance', 88), '88.00');
    eq('fieldSuffix builds the DOM id suffix', context.fieldSuffix('study_hours'), 'Study_hours');

    // --- 2. Client-side validation ----------------------------------------
    console.log('\n[2] Client-side validation');
    const { validateForm, findExtrapolationWarnings, readFormValues } = context;

    setForm(VALID);
    let errors = validateForm();
    check('valid form produces no errors',
        Object.values(errors).every(e => !e), JSON.stringify(errors));

    setForm({ ...VALID, study_hours: '' });
    check('empty study_hours is rejected',
        validateForm().study_hours.includes('required'), validateForm().study_hours);

    setForm({ ...VALID, study_hours: -1 });
    check('negative study hours is rejected',
        validateForm().study_hours.includes('0 or greater'), validateForm().study_hours);

    setForm({ ...VALID, attendance: -5 });
    check('negative attendance is rejected',
        validateForm().attendance.includes('0 or greater'), validateForm().attendance);

    setForm({ ...VALID, attendance: 150 });
    check('attendance above 100 is rejected',
        validateForm().attendance.includes('at most 100'), validateForm().attendance);

    setForm({ ...VALID, previous_score: 120 });
    check('previous score above 100 is rejected',
        validateForm().previous_score.includes('at most 100'), validateForm().previous_score);

    setForm({ ...VALID, assignments_completed: -2 });
    check('negative assignments are rejected',
        validateForm().assignments_completed.includes('0 or greater'),
        validateForm().assignments_completed);

    setForm({ ...VALID, practical_score: 101 });
    check('practical score above 100 is rejected',
        validateForm().practical_score.includes('at most 100'), validateForm().practical_score);

    setForm({ ...VALID, attendance: 'absent' });
    check('non-numeric attendance is rejected',
        validateForm().attendance.includes('must be a number'), validateForm().attendance);

    setForm({});
    errors = validateForm();
    check('empty form flags all 5 fields',
        Object.values(errors).filter(e => e).length === 5, JSON.stringify(errors));

    // --- 3. Range validation (ranges loaded from the live API) ------------
    await context.loadSchema();
    const schemaRes = await fetch(API + '/api/knn/schema');
    const schema = await schemaRes.json();
    console.log('\n[3] Range validation (ranges loaded from the live API)');
    console.log('      ' + schema.features.map(f => f.name + ': ' + f.min + '-' + f.max).join(', '));
    eq('range note badge is updated',
        elements.get('datasetRangeNote').textContent, 'Ranges from training data');

    // A value outside the trained range is still classifiable: the form must
    // accept it and flag it as an extrapolation warning rather than block it.
    setForm({ ...VALID, study_hours: 60 });
    eq('out-of-range study hours passes validation', validateForm().study_hours, '');
    check('out-of-range study hours is flagged as extrapolation',
        findExtrapolationWarnings(readFormValues()).some(w => w.field === 'study_hours'),
        JSON.stringify(findExtrapolationWarnings(readFormValues())));

    setForm({ ...VALID, attendance: 100 });
    eq('attendance of 100 (the documented maximum) is accepted', validateForm().attendance, '');

    const hoursMin = schema.features.find(f => f.name === 'study_hours').min;
    setForm({ ...VALID, study_hours: hoursMin });
    check('in-range minimum study hours is accepted', validateForm().study_hours === '',
        validateForm().study_hours);

    setForm(VALID);
    eq('an all-in-range student has no extrapolation warnings',
        findExtrapolationWarnings(readFormValues()).length, 0);

    // --- 4. Live prediction flow ------------------------------------------
    console.log('\n[4] Live prediction flow (real Flask -> R engine)');
    setForm(VALID);
    await context.submitPrediction();

    eq('result panel reaches the success state', elements.get('resultPanel').dataset.state, 'success');
    const badge = elements.get('resultPerformanceBadge');
    check('performance badge carries a class',
        ['LOW', 'MEDIUM', 'HIGH'].includes(badge.dataset.performance), badge.dataset.performance);

    const liveData = await (await fetch(API + '/api/knn/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(VALID)
    })).json();

    eq('badge text matches the API prediction', badge.textContent, liveData.prediction);
    eq('model name is displayed', elements.get('resultModelName').textContent,
        'K-Nearest Neighbors');
    eq('selected K is displayed', elements.get('resultSelectedK').textContent, 'K = ' + liveData.k);

    // Confidence is only shown when the model actually reported one.
    eq('confidence matches the API value', elements.get('resultConfidence').textContent,
        context.formatPercent(liveData.confidence));
    eq('confidence block is visible', elements.get('confidenceBlock').style.display, 'block');
    check('the confidence basis is explained to the user',
        elements.get('confidenceNote').textContent.includes('nearest'),
        elements.get('confidenceNote').textContent);

    // Neighbour vote rows: one per performance tier.
    const voteRows = elements.get('voteList').children;
    eq('one vote row is drawn per performance tier', voteRows.length, 3);
    check('vote rows show real neighbour counts',
        voteRows.some(r => r.children[2].textContent.includes('/')),
        JSON.stringify(voteRows.map(r => r.children[2].textContent)));

    // Input summary: five key/value pairs.
    eq('input summary lists all five fields', elements.get('inputSummary').children.length, 10);

    // --- 5. Nearest-neighbour rendering ----------------------------------
    console.log('\n[5] Nearest-neighbour rendering');
    const cards = elements.get('neighborList').children;
    eq('a card is drawn per neighbour', cards.length, liveData.neighbors.length);
    eq('a card is drawn per K', cards.length, liveData.k);
    eq('neighbour panel reaches the success state',
        elements.get('neighborPanel').dataset.state, 'success');

    check('each card is labelled with its rank',
        cards[0].children[0].children[0].textContent.startsWith('Neighbor'),
        cards[0].children[0].children[0].textContent);
    eq('card 1 shows the API neighbour class',
        cards[0].children[0].children[1].textContent, liveData.neighbors[0].neighbor_class);
    check('card 1 shows the real distance from the API',
        cards[0].children[0].children[2].textContent.includes(
            liveData.neighbors[0].distance.toFixed(4)),
        cards[0].children[0].children[2].textContent + ' vs ' + liveData.neighbors[0].distance);

    // Five feature chips per card, one per model feature.
    eq('each card lists all five features', cards[0].children[1].children.length, 5);

    check('the neighbour summary mentions the real K',
        elements.get('neighborSummary').textContent.includes(String(liveData.k)),
        elements.get('neighborSummary').textContent);

    // --- 6. Different inputs produce different tiers ----------------------
    console.log('\n[6] Classification responds to input');
    setForm({ study_hours: 34, attendance: 98, previous_score: 95,
              assignments_completed: 9, practical_score: 95 });
    await context.submitPrediction();
    const strong = elements.get('resultPerformanceBadge').dataset.performance;
    console.log('      strong student      -> ' + strong);

    setForm({ study_hours: 3, attendance: 60, previous_score: 38,
              assignments_completed: 1, practical_score: 42 });
    await context.submitPrediction();
    const weak = elements.get('resultPerformanceBadge').dataset.performance;
    console.log('      weak student        -> ' + weak);

    check('a strong student is classified differently from a weak one',
        strong !== weak, 'both returned ' + strong);
    check('the weak student is not HIGH performance', weak !== 'HIGH', weak);

    // --- 7. Error rendering -----------------------------------------------
    console.log('\n[7] Error rendering');
    setForm(VALID);
    context.renderError('Invalid student details', ['Attendance must be at most 100.']);
    eq('error state is set', elements.get('resultPanel').dataset.state, 'error');
    eq('error title is rendered', elements.get('resultErrorTitle').textContent,
        'Invalid student details');
    eq('each error message becomes a list item',
        elements.get('resultErrorList').children.length, 1);

    // A client-side validation failure must never reach the server.
    setForm({ ...VALID, attendance: 150 });
    await context.submitPrediction();
    eq('client-side validation blocks the request',
        elements.get('resultPanel').dataset.state, 'error');
    check('inline field error is painted',
        elements.get('errorAttendance').textContent.includes('at most 100'),
        elements.get('errorAttendance').textContent);

    // A valid form must recover from the error state.
    setForm(VALID);
    await context.submitPrediction();
    eq('a valid form recovers from the error state',
        elements.get('resultPanel').dataset.state, 'success');

    // --- 8. Metrics + K rendering (live API) ----------------------------
    console.log('\n[8] Metrics, K values and matrix (live API)');
    await context.loadConfig();
    await context.loadMetrics();
    const metricsRes = await (await fetch(API + '/api/knn/metrics')).json();
    const configRes = await (await fetch(API + '/api/knn/config')).json();

    eq('accuracy tile shows the trained value', elements.get('metricAccuracy').textContent,
        metricsRes.metrics.accuracy.toFixed(4));
    eq('precision tile shows the trained value', elements.get('metricPrecision').textContent,
        metricsRes.metrics.precision.toFixed(4));
    eq('recall tile shows the trained value', elements.get('metricRecall').textContent,
        metricsRes.metrics.recall.toFixed(4));
    eq('f1 tile shows the trained value', elements.get('metricF1').textContent,
        metricsRes.metrics.f1_score.toFixed(4));
    eq('supported-only f1 tile shows the trained value',
        elements.get('metricSupportedF1').textContent,
        metricsRes.metrics.macro_supported_only.f1_score.toFixed(4));
    eq('test row count is displayed', elements.get('metricTestRows').textContent,
        String(metricsRes.dataset.rows_test));
    eq('training row count is displayed', elements.get('metricTrainRows').textContent,
        String(metricsRes.dataset.rows_train));
    eq('model state is reported ready', elements.get('modelStateDisplay').textContent,
        'Model trained & ready');
    check('performance note explains the macro averaging',
        elements.get('performanceNote').textContent.includes('macro averages'),
        elements.get('performanceNote').textContent);

    // The dataset's single-LOW-row limitation must be surfaced to the user.
    eq('class-balance caveat is displayed', elements.get('balanceNote').textContent,
        metricsRes.class_distribution_note.message);
    check('the caveat explains the LOW class limit',
        elements.get('balanceNote').textContent.includes('LOW'),
        elements.get('balanceNote').textContent);

    // K value rendering
    eq('the selected K is shown in the hero', elements.get('selectedKDisplay').textContent,
        'K = ' + configRes.selected_k);

    const kRows = elements.get('kTableBody').children;
    eq('one row is drawn per evaluated K', kRows.length, configRes.k_grid.length);
    eq('K rows are ordered ascending',
        kRows.map(r => Number(r.children[0].textContent)).join(','),
        configRes.k_grid.slice().sort((a, b) => a - b).join(','));
    eq('the selected K row is marked', kRows.filter(r => r.dataset.selected === 'true').length, 1);
    eq('the marked row is the model\'s K',
        Number(kRows.find(r => r.dataset.selected === 'true').children[0].textContent),
        configRes.selected_k);
    eq('the selected K row shows the real CV score',
        kRows.find(r => r.dataset.selected === 'true').children[1].textContent,
        (metricsRes.k_comparison.find(e => e.k === configRes.selected_k).cv_f1_mean).toFixed(4));

    // --- 9. Confusion matrix ----------------------------------------------
    console.log('\n[9] Confusion matrix');
    const grid = elements.get('matrixWrap').children[0];
    check('a matrix grid is rendered', !!grid, 'no grid element');
    const matrixCells = grid ? grid.children : [];
    // 1 corner + 3 column headers + 3 * (1 row header + 3 cells) = 16
    eq('matrix grid has a header and a cell for every class', matrixCells.length, 16);
    eq('three diagonal cells are marked correct',
        matrixCells.filter(c => c.dataset && c.dataset.diagonal === 'true').length, 3);
    check('matrix note reports the correct count',
        elements.get('matrixNote').textContent.includes('classified correctly'),
        elements.get('matrixNote').textContent);

    // --- Summary -----------------------------------------------------------
    console.log('\n=== Results: ' + passed + ' passed, ' + failed + ' failed ===\n');
    process.exit(failed === 0 ? 0 : 1);
})().catch(err => {
    console.error('\nFATAL: test harness error:', err);
    process.exit(1);
});
