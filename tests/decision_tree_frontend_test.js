/**
 * Headless behavioural test for frontend/js/decision-tree.js
 *
 * Runs the real page script in Node against a minimal DOM shim, with fetch
 * pointed at the live Flask server, and asserts the actual user-facing
 * behaviour: formatting, client-side validation, the prediction flow, the
 * decision-path trace, the tree visualisation and the confusion matrix.
 *
 * Run:  node tests/decision_tree_frontend_test.js
 * Requires the Flask backend to be running on 127.0.0.1:5000.
 */
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const API = process.env.API_BASE || 'http://127.0.0.1:5000';
const SCRIPT = path.join(__dirname, '..', 'frontend', 'js', 'decision-tree.js');

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
    'btnRetryModel', 'modelStateDisplay', 'datasetRangeNote', 'modelAlertBanner',
    'modelAlertTitle', 'modelAlertMessage', 'resultPanel', 'resultEmpty',
    'resultBody', 'resultError', 'resultErrorTitle', 'resultErrorList',
    'resultRiskBadge', 'confidenceBlock', 'resultConfidence', 'confidenceFill',
    'probList', 'inputSummary', 'resultModelName', 'resultAlgorithm',
    'resultTrainedAt', 'resultExecTime', 'metricAccuracy', 'metricPrecision',
    'metricRecall', 'metricF1', 'metricBalanced', 'metricTestRows',
    'metricTrainRows', 'performanceNote', 'pathPanel', 'pathEmpty', 'pathBody',
    'pathList', 'treeCanvas', 'treePlaceholder', 'treeNodeCount',
    'importanceGrid', 'matrixWrap', 'matrixNote'
]) {
    elements.set(id, makeElement(id));
}

// Form fields + their hint/error spans
for (const key of ['age', 'income', 'credit_score', 'existing_loans', 'employment_years']) {
    const suffix = key[0].toUpperCase() + key.slice(1);
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
    for (const key of ['age', 'income', 'credit_score', 'existing_loans', 'employment_years']) {
        const suffix = key[0].toUpperCase() + key.slice(1);
        elements.get('input' + suffix).value = values[key] !== undefined ? String(values[key]) : '';
        elements.get('error' + suffix).textContent = '';
    }
}

const FIELDS = ['age', 'income', 'credit_score', 'existing_loans', 'employment_years'];
const VALID = { age: 32, income: 95000, credit_score: 735, existing_loans: 1, employment_years: 5 };

(async () => {
    console.log('\n=== AI Insight Hub - Frontend Logic Test (decision-tree.js) ===\n');

    // --- 1. Formatting -----------------------------------------------------
    console.log('[1] Formatting');
    eq('formatInr uses the Indian grouping format', context.formatInr(78675, 0), '₹78,675');
    eq('formatInr rounds to 0dp', context.formatInr(135306.68, 0), '₹1,35,307');
    eq('formatNumber 4dp', context.formatNumber(0.6522, 4), '0.6522');
    eq('formatPercent converts a probability', context.formatPercent(0.8), '80.0%');
    eq('formatPercent on 1.0', context.formatPercent(1), '100.0%');
    eq('formatNumber handles a non-number safely', context.formatNumber(null, 2), '—');
    eq('formatPercent handles a non-number safely', context.formatPercent('x'), '—');
    eq('income is formatted as rupees', context.formatFeatureValue('income', 78675), '₹78,675');
    eq('credit score is shown as a whole number', context.formatFeatureValue('credit_score', 735), '735');
    eq('age is shown as a whole number', context.formatFeatureValue('age', 41), '41');

    // --- 2. Client-side validation ----------------------------------------
    console.log('\n[2] Client-side validation');
    const { validateForm, findExtrapolationWarnings, readFormValues } = context;

    setForm(VALID);
    let errors = validateForm();
    check('valid form produces no errors', Object.values(errors).every(e => !e), JSON.stringify(errors));

    setForm({ ...VALID, age: '' });
    check('empty age is rejected', validateForm().age.includes('required'), validateForm().age);

    setForm({ ...VALID, age: 0 });
    check('zero age is rejected', validateForm().age.includes('greater than 0'), validateForm().age);

    setForm({ ...VALID, age: -5 });
    check('negative age is rejected', validateForm().age.includes('greater than 0'), validateForm().age);

    setForm({ ...VALID, income: -1 });
    check('negative income is rejected', validateForm().income.includes('greater than 0'), validateForm().income);

    setForm({ ...VALID, credit_score: 999 });
    check('credit score above 900 is rejected', validateForm().credit_score.includes('at most'), validateForm().credit_score);

    setForm({ ...VALID, existing_loans: -1 });
    check('negative existing_loans is rejected', validateForm().existing_loans.includes('0 or greater'), validateForm().existing_loans);

    setForm({ ...VALID, employment_years: -3 });
    check('negative employment_years is rejected', validateForm().employment_years.includes('0 or greater'), validateForm().employment_years);

    setForm({ ...VALID, credit_score: 'excellent' });
    check('non-numeric credit score is rejected', validateForm().credit_score.includes('must be a number'), validateForm().credit_score);

    setForm({});
    errors = validateForm();
    check('empty form flags all 5 fields', Object.values(errors).filter(e => e).length === 5, JSON.stringify(errors));

    // Range validation is loaded from the live API by the page's own loader.
    await context.loadSchema();
    const schemaRes = await fetch(API + '/api/decision-tree/schema');
    const schema = await schemaRes.json();
    console.log('\n[3] Range validation (ranges loaded from the live API)');
    console.log('      ' + schema.features.map(f => f.name + ': ' + f.min + '-' + f.max).join(', '));
    eq('range note badge is updated',
        elements.get('datasetRangeNote').textContent, 'Ranges from training data');

    // A value outside the trained range is still classifiable: the form must
    // accept it and flag it as an extrapolation warning rather than block it.
    setForm({ ...VALID, income: 750000 });
    eq('out-of-range income passes validation', validateForm().income, '');
    check('out-of-range income is flagged as extrapolation',
        findExtrapolationWarnings(readFormValues()).some(w => w.field === 'income'),
        JSON.stringify(findExtrapolationWarnings(readFormValues())));

    setForm({ ...VALID, credit_score: 900 });
    eq('credit score of 900 (the documented maximum) is accepted', validateForm().credit_score, '');

    const ageMin = schema.features.find(f => f.name === 'age').min;
    setForm({ ...VALID, age: ageMin });
    check('in-range minimum age is accepted', validateForm().age === '', validateForm().age);

    setForm(VALID);
    eq('an all-in-range applicant has no extrapolation warnings',
        findExtrapolationWarnings(readFormValues()).length, 0);

    // --- 4. Live prediction flow ------------------------------------------
    console.log('\n[4] Live prediction flow (real Flask -> R engine)');
    setForm(VALID);
    await context.submitPrediction();

    eq('result panel reaches the success state', elements.get('resultPanel').dataset.state, 'success');
    const badge = elements.get('resultRiskBadge');
    check('risk badge text ends with RISK', badge.textContent.endsWith(' RISK'), badge.textContent);
    check('risk badge carries a risk class',
        ['LOW', 'MEDIUM', 'HIGH'].includes(badge.dataset.risk), badge.dataset.risk);
    eq('model name is displayed', elements.get('resultModelName').textContent, 'Decision Tree');

    // Confidence is only shown when the model actually reported one.
    const liveData = await (await fetch(API + '/api/decision-tree/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(VALID)
    })).json();
    eq('confidence matches the API value', elements.get('resultConfidence').textContent,
        context.formatPercent(liveData.confidence));
    eq('confidence block is visible', elements.get('confidenceBlock').style.display, 'block');

    // Class probability rows: one per risk class.
    const probRows = elements.get('probList').children;
    eq('one probability row is drawn per risk class', probRows.length, 3);
    check('probability rows carry the real percentages',
        probRows.some(r => r.children[2].textContent.endsWith('%')),
        JSON.stringify(probRows.map(r => r.children[2].textContent)));

    // Input summary: five key/value pairs.
    eq('input summary lists all five fields', elements.get('inputSummary').children.length, 10);

    // --- 5. Decision path ---------------------------------------------------
    console.log('\n[5] Decision path trace');
    const pathSteps = elements.get('pathList').children;
    check('a step is drawn per decision-path entry',
        pathSteps.length === liveData.decision_path.length,
        'got ' + pathSteps.length + ', expected ' + liveData.decision_path.length);
    eq('path panel reaches the success state', elements.get('pathPanel').dataset.state, 'success');
    eq('final path step is a leaf', pathSteps[pathSteps.length - 1].dataset.kind, 'leaf');
    check('the leaf step carries the predicted risk',
        pathSteps[pathSteps.length - 1].dataset.risk === liveData.prediction,
        pathSteps[pathSteps.length - 1].dataset.risk + ' vs ' + liveData.prediction);
    check('a decision step shows the real condition text',
        pathSteps[0].children[1].children[0].textContent.length > 0,
        pathSteps[0].children[1].children[0].textContent);

    // --- 6. Different inputs produce different classes ---------------------
    console.log('\n[6] Risk classification responds to input');
    setForm({ age: 41, income: 78675, credit_score: 666, existing_loans: 1, employment_years: 24 });
    await context.submitPrediction();
    const lowRisk = elements.get('resultRiskBadge').dataset.risk;
    console.log('      low-debt applicant   -> ' + lowRisk);

    setForm({ age: 24, income: 28361, credit_score: 412, existing_loans: 5, employment_years: 11 });
    await context.submitPrediction();
    const highRisk = elements.get('resultRiskBadge').dataset.risk;
    console.log('      high-debt applicant  -> ' + highRisk);

    check('a heavily indebted applicant is classified differently from a low-debt one',
        lowRisk !== highRisk, 'both returned ' + highRisk);
    check('the high-debt applicant is not LOW risk', highRisk !== 'LOW', highRisk);

    // --- 7. Error rendering -----------------------------------------------
    console.log('\n[7] Error rendering');
    setForm(VALID);
    context.renderError('Invalid applicant details', ['Annual Income must be between 1 and 2.']);
    eq('error state is set', elements.get('resultPanel').dataset.state, 'error');
    eq('error title is rendered', elements.get('resultErrorTitle').textContent, 'Invalid applicant details');

    // A client-side validation failure must never reach the server.
    setForm({ ...VALID, age: -5 });
    await context.submitPrediction();
    eq('client-side validation blocks the request', elements.get('resultPanel').dataset.state, 'error');
    check('inline field error is painted',
        elements.get('errorAge').textContent.includes('greater than 0'),
        elements.get('errorAge').textContent);

    // A valid form must recover from the error state.
    setForm(VALID);
    await context.submitPrediction();
    eq('a valid form recovers from the error state', elements.get('resultPanel').dataset.state, 'success');

    // --- 8. Metrics loader -------------------------------------------------
    console.log('\n[8] Metrics loader (live API)');
    await context.loadMetrics();
    const metricsRes = await (await fetch(API + '/api/decision-tree/metrics')).json();
    eq('accuracy tile shows the trained value', elements.get('metricAccuracy').textContent,
        metricsRes.metrics.accuracy.toFixed(4));
    eq('precision tile shows the trained value', elements.get('metricPrecision').textContent,
        metricsRes.metrics.precision.toFixed(4));
    eq('recall tile shows the trained value', elements.get('metricRecall').textContent,
        metricsRes.metrics.recall.toFixed(4));
    eq('f1 tile shows the trained value', elements.get('metricF1').textContent,
        metricsRes.metrics.f1_score.toFixed(4));
    eq('test row count is displayed', elements.get('metricTestRows').textContent,
        String(metricsRes.dataset.rows_test));
    eq('training row count is displayed', elements.get('metricTrainRows').textContent,
        String(metricsRes.dataset.rows_train));
    eq('model state is reported ready', elements.get('modelStateDisplay').textContent, 'Model trained & ready');
    check('performance note explains the imbalance',
        elements.get('performanceNote').textContent.includes('macro averages'),
        elements.get('performanceNote').textContent);

    // --- 9. Confusion matrix ----------------------------------------------
    console.log('\n[9] Confusion matrix');
    const grid = elements.get('matrixWrap').children[0];
    check('a matrix grid is rendered', !!grid, 'no grid element');
    const matrixCells = grid ? grid.children : [];
    // 1 corner + 3 column headers + 3 * (1 row header + 3 cells) = 16
    eq('matrix grid has a header and a cell for every class', matrixCells.length, 16);
    const diagonals = matrixCells.filter(c => c.dataset && c.dataset.diagonal === 'true');
    eq('three diagonal cells are marked correct', diagonals.length, 3);
    check('matrix note reports the correct count',
        elements.get('matrixNote').textContent.includes('classified correctly'),
        elements.get('matrixNote').textContent);

    // --- 10. Tree visualisation -------------------------------------------
    console.log('\n[10] Tree visualisation (live API)');
    await context.loadTree();
    const treeRes = await (await fetch(API + '/api/decision-tree/tree')).json();
    const treeChildren = elements.get('treeCanvas').children;
    // Cards are created with className=; SVG nodes get it via setAttribute().
    const nodeCards = treeChildren.filter(c => c.className === 'tree-node');
    const edgeLabels = treeChildren.filter(c => c.className === 'tree-edge');
    const links = treeChildren.filter(c => c.attributes.class === 'tree-links')[0];

    check('tree placeholder is hidden after load',
        elements.get('treePlaceholder').style.display === 'none',
        elements.get('treePlaceholder').style.display);
    eq('a card is drawn for every node in the fitted tree',
        nodeCards.length, treeRes.total_nodes);
    check('the node count badge is populated',
        elements.get('treeNodeCount').textContent.includes(String(treeRes.total_nodes)),
        elements.get('treeNodeCount').textContent);

    // rpart's binary layout: a tree with L leaves has L-1 decision nodes, so
    // the number of drawn branches must equal 2 * (total nodes - leaves).
    const expectedEdges = 2 * (treeRes.total_nodes - treeRes.leaf_nodes);
    eq('a connector label is drawn per branch', edgeLabels.length, expectedEdges);
    check('an SVG connector layer is present', !!links, 'no .tree-links element');
    check('the connector layer holds one path per branch',
        links && links.children.length === expectedEdges,
        'got ' + (links ? links.children.length : 0) + ', expected ' + expectedEdges);

    // The root card must show the real root split.
    check('a decision node is drawn with a threshold',
        nodeCards.some(c => c.children[0] && c.children[0].textContent &&
            (c.children[0].textContent.includes('<') || c.children[0].textContent.includes('>='))),
        'no split rule found on any card');
    check('the root card splits on the most important feature',
        nodeCards[0].children[0].textContent.startsWith(
            { age: 'Age', income: 'Annual Income (INR)', credit_score: 'Credit Score',
              existing_loans: 'Existing Loans', employment_years: 'Employment Years' }[
                treeRes.variable_importance[0].feature]),
        nodeCards[0].children[0].textContent);

    // --- 11. Variable importance -----------------------------------------
    console.log('\n[11] Variable importance');
    const importanceRows = elements.get('importanceGrid').children;
    eq('one importance row is drawn per feature',
        importanceRows.length, treeRes.variable_importance.length);
    eq('importance is ordered strongest first',
        importanceRows[0].children[0].textContent, treeRes.variable_importance[0].feature);

    // --- Summary -----------------------------------------------------------
    console.log('\n=== Results: ' + passed + ' passed, ' + failed + ' failed ===\n');
    process.exit(failed === 0 ? 0 : 1);
})().catch(err => {
    console.error('\nFATAL: test harness error:', err);
    process.exit(1);
});
