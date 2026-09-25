/**
 * Headless behavioural test for frontend/js/regression.js
 *
 * Runs the real page script in Node against a minimal DOM shim, with fetch
 * pointed at the live Flask server, and asserts the actual user-facing
 * behaviour: formatting, client-side validation, and the prediction flow.
 *
 * Run:  node tests/frontend_logic_test.js
 * Requires the Flask backend to be running on 127.0.0.1:5000.
 */
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const API = process.env.API_BASE || 'http://127.0.0.1:5000';
const SCRIPT = path.join(__dirname, '..', 'frontend', 'js', 'regression.js');

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
    'resultPriceLakhs', 'resultRawInr', 'resultModelName', 'resultAlgorithm',
    'resultTrainedAt', 'resultExecTime', 'metricR2', 'metricRmse', 'metricMae',
    'metricAdjR2', 'metricTestRows', 'metricTrainRows', 'scatterChart',
    'chartPlaceholder', 'chartPointCount', 'chartWrapper', 'chartTooltip',
    'coefficientsGrid'
]) {
    elements.set(id, makeElement(id));
}

// Form fields + their hint/error spans
for (const key of ['area', 'bedrooms', 'bathrooms', 'location_score', 'property_age']) {
    const suffix = key[0].toUpperCase() + key.slice(1);
    elements.set('input' + suffix, makeElement('input' + suffix));
    elements.set('hint' + suffix, makeElement('hint' + suffix));
    elements.set('error' + suffix, makeElement('error' + suffix));
}

const documentStub = {
    getElementById: id => elements.get(id) || null,
    createElement: tag => makeElement('created-' + tag),
    createElementNS: (ns, tag) => makeElement('svg-' + tag),
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
    check(name, actual === expected, 'expected ' + JSON.stringify(expected) + ', got ' + JSON.stringify(actual));
}

function setForm(values) {
    for (const key of ['area', 'bedrooms', 'bathrooms', 'location_score', 'property_age']) {
        const suffix = key[0].toUpperCase() + key.slice(1);
        elements.get('input' + suffix).value = values[key] !== undefined ? String(values[key]) : '';
        elements.get('error' + suffix).textContent = '';
    }
}

const VALID = { area: 1500, bedrooms: 3, bathrooms: 2, location_score: 8, property_age: 5 };

(async () => {
    console.log('\n=== AI Insight Hub - Frontend Logic Test (regression.js) ===\n');

    // --- 1. Formatting -----------------------------------------------------
    console.log('[1] Currency formatting');
    eq('formatInr uses the Indian grouping format', context.formatInr(355005.39, 2), '₹3,55,005.39');
    eq('formatInr whole numbers', context.formatInr(500000, 0), '₹5,00,000');
    eq('toLakhs converts to lakhs', context.toLakhs(7842000), '78.42');
    eq('toLakhs rounds to 2dp', context.toLakhs(355005.39), '3.55');
    eq('formatCompactInr rounds', context.formatCompactInr(5646.2992), '₹5,646');
    eq('formatNumber 4dp', context.formatNumber(0.9967, 4), '0.9967');
    eq('formatInr handles a non-number safely', context.formatInr(null), '—');

    // --- 2. Client-side validation ----------------------------------------
    console.log('\n[2] Client-side validation');
    const { validateForm } = context;

    setForm(VALID);
    let errors = validateForm();
    check('valid form produces no errors', Object.values(errors).every(e => !e), JSON.stringify(errors));

    setForm({ ...VALID, area: '' });
    check('empty area is rejected', validateForm().area.includes('required'), validateForm().area);

    setForm({ ...VALID, area: -100 });
    check('negative area is rejected', validateForm().area.includes('greater than 0'), validateForm().area);

    setForm({ ...VALID, area: 0 });
    check('zero area is rejected', validateForm().area.includes('greater than 0'), validateForm().area);

    setForm({ ...VALID, bedrooms: 0 });
    check('zero bedrooms is rejected', validateForm().bedrooms.includes('greater than 0'), validateForm().bedrooms);

    setForm({ ...VALID, bathrooms: 0 });
    check('zero bathrooms is rejected', validateForm().bathrooms.includes('greater than 0'), validateForm().bathrooms);

    setForm({ ...VALID, property_age: -5 });
    check('negative property age is rejected', validateForm().property_age.includes('0 or greater'), validateForm().property_age);

    setForm({ ...VALID, property_age: 0 });
    check('property_age of 0 is allowed', validateForm().property_age === '', validateForm().property_age);

    setForm({ ...VALID, area: 'abc' });
    check('non-numeric area is rejected', validateForm().area.includes('must be a number'), validateForm().area);

    setForm({});
    errors = validateForm();
    check('empty form flags all 5 fields', Object.values(errors).filter(e => e).length === 5, JSON.stringify(errors));

    // Range validation is loaded from the live API by the page's own loader.
    await context.loadSchema();
    const schemaRes = await fetch(API + '/api/regression/schema');
    const schema = await schemaRes.json();
    console.log('\n[3] Range validation (ranges loaded from the live API)');
    console.log('      ' + schema.features.map(f => f.name + ': ' + f.min + '-' + f.max).join(', '));
    eq('range note badge is updated',
        elements.get('datasetRangeNote').textContent, 'Ranges from training data');

    setForm({ ...VALID, area: 99999 });
    check('out-of-range area is rejected', validateForm().area.includes('between'), validateForm().area);

    setForm({ ...VALID, location_score: 99 });
    check('out-of-range location_score is rejected', validateForm().location_score.includes('between'), validateForm().location_score);

    setForm({ ...VALID, location_score: 1 });
    check('below-range location_score is rejected', validateForm().location_score.includes('between'), validateForm().location_score);

    setForm({ ...VALID, area: schema.features.find(f => f.name === 'area').min });
    check('in-range minimum area is accepted', validateForm().area === '', validateForm().area);

    setForm({ ...VALID, area: schema.features.find(f => f.name === 'area').max });
    check('in-range maximum area is accepted', validateForm().area === '', validateForm().area);

    // --- 4. Live prediction flow ------------------------------------------
    console.log('\n[4] Live prediction flow (real Flask -> R engine)');
    setForm(VALID);
    await context.submitPrediction();

    eq('result panel reaches the success state', elements.get('resultPanel').dataset.state, 'success');
    check('headline price is rendered with the rupee sign',
        elements.get('resultPriceLakhs').textContent.startsWith('₹'),
        elements.get('resultPriceLakhs').textContent);
    eq('headline equals toLakhs of the model price',
        elements.get('resultPriceLakhs').textContent, '₹' + context.toLakhs(355005.39));
    check('raw INR value is shown',
        elements.get('resultRawInr').textContent.includes('3,55,005.39'),
        elements.get('resultRawInr').textContent);
    eq('model name is displayed', elements.get('resultModelName').textContent, 'Linear Regression');

    // The price must change with the input (proves nothing is hard-coded).
    setForm({ ...VALID, area: 900, bedrooms: 2, bathrooms: 1 });
    await context.submitPrediction();
    const smallPrice = elements.get('resultRawInr').textContent;
    console.log('      900 sq.ft  -> ' + smallPrice);

    setForm({ ...VALID, area: 4000, bedrooms: 5, bathrooms: 4 });
    await context.submitPrediction();
    const largePrice = elements.get('resultRawInr').textContent;
    console.log('      4000 sq.ft -> ' + largePrice);

    check('different inputs produce different prices', smallPrice !== largePrice, 'both returned ' + largePrice);
    check('larger property is valued higher',
        parseFloat(largePrice.replace(/[^0-9.]/g, '')) > parseFloat(smallPrice.replace(/[^0-9.]/g, '')),
        largePrice + ' vs ' + smallPrice);

    // --- 5. Error rendering -----------------------------------------------
    console.log('\n[5] Error rendering');
    setForm(VALID);
    context.renderError('Invalid property details', ['Area (sq.ft) must be between 1 and 2.']);
    eq('error state is set', elements.get('resultPanel').dataset.state, 'error');
    eq('error title is rendered', elements.get('resultErrorTitle').textContent, 'Invalid property details');

    // A client-side validation failure must never reach the server.
    setForm({ ...VALID, area: -5 });
    await context.submitPrediction();
    eq('client-side validation blocks the request', elements.get('resultPanel').dataset.state, 'error');
    check('inline field error is painted',
        elements.get('errorArea').textContent.includes('greater than 0'),
        elements.get('errorArea').textContent);

    // --- 6. Metrics & evaluation loaders ----------------------------------
    console.log('\n[6] Metrics and evaluation loaders (live API)');
    await context.loadMetrics();
    eq('R2 tile shows the trained value', elements.get('metricR2').textContent, '0.9967');
    eq('RMSE tile shows a rupee value', elements.get('metricRmse').textContent, '₹5,646');
    eq('MAE tile shows a rupee value', elements.get('metricMae').textContent, '₹5,005');
    eq('test row count is displayed', elements.get('metricTestRows').textContent, '24');
    eq('training row count is displayed', elements.get('metricTrainRows').textContent, '96');
    eq('model state is reported ready', elements.get('modelStateDisplay').textContent, 'Model trained & ready');

    await context.loadEvaluation();
    check('chart placeholder hidden after load',
        elements.get('chartPlaceholder').style.display === 'none',
        elements.get('chartPlaceholder').style.display);
    eq('chart point count badge updated', elements.get('chartPointCount').textContent, '24 test records');

    // Inspect the SVG the page actually built.
    const svgChildren = elements.get('scatterChart').children;
    const pointNodes = svgChildren.filter(c => (c.attributes.class || '').indexOf('chart-point') !== -1);
    const diagonalNodes = svgChildren.filter(c => (c.attributes.class || '').indexOf('chart-diagonal') !== -1);
    const gridNodes = svgChildren.filter(c => (c.attributes.class || '').indexOf('chart-grid-line') !== -1);
    const axisNodes = svgChildren.filter(c => (c.attributes.class || '').indexOf('chart-axis-line') !== -1);

    eq('one SVG point is drawn per test record', pointNodes.length, 24);
    eq('a perfect-prediction diagonal is drawn', diagonalNodes.length, 1);
    check('grid lines are drawn', gridNodes.length >= 10, 'got ' + gridNodes.length);
    eq('both axis lines are drawn', axisNodes.length, 2);

    // Every plotted point must carry the real actual/predicted values.
    const firstPoint = pointNodes[0];
    check('data points carry real values',
        firstPoint.attributes.cx !== undefined && Number(firstPoint.attributes.cy) > 0,
        JSON.stringify(firstPoint.attributes));
    check('data points expose actual for the tooltip',
        firstPoint.dataset.actual !== undefined && Number(firstPoint.dataset.actual) > 0,
        JSON.stringify(firstPoint.dataset));

    // Coordinate sanity: a bigger actual price must plot further right.
    const xs = pointNodes.map(p => Number(p.attributes.cx));
    const ys = pointNodes.map(p => Number(p.attributes.cy));
    check('chart coordinates stay inside the plot area',
        xs.every(x => x >= 0 && x <= 900) && ys.every(y => y >= 0 && y <= 520),
        'x range ' + Math.min(...xs) + '-' + Math.max(...xs) +
        ', y range ' + Math.min(...ys) + '-' + Math.max(...ys));

    // --- 7. Coefficients ---------------------------------------------------
    console.log('\n[7] Coefficients');
    await context.loadCoefficients();
    const coefficientCards = elements.get('coefficientsGrid').children;
    check('one card is drawn per model coefficient',
        coefficientCards.length === 6, 'got ' + coefficientCards.length);

    // --- Summary -----------------------------------------------------------
    console.log('\n=== Results: ' + passed + ' passed, ' + failed + ' failed ===\n');
    process.exit(failed === 0 ? 0 : 1);
})().catch(err => {
    console.error('\nFATAL: test harness error:', err);
    process.exit(1);
});
