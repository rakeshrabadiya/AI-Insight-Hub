/**
 * Headless behavioural test for frontend/js/kmeans.js
 *
 * Runs the real page script in Node against a minimal DOM shim, with fetch
 * pointed at the live Flask server, and asserts the actual user-facing
 * behaviour: formatting, client-side validation, the assignment flow, the
 * cluster overview cards, the profile tables, the cluster scatter plot's data
 * handling, the K-evaluation chart, the metrics tiles and the error paths.
 *
 * Run:  node tests/kmeans_frontend_test.js
 * Requires the Flask backend to be running on 127.0.0.1:5000.
 */
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const API = process.env.API_BASE || 'http://127.0.0.1:5000';
const SCRIPT = path.join(__dirname, '..', 'frontend', 'js', 'kmeans.js');

// --- Minimal DOM shim -------------------------------------------------------
const elements = new Map();

function makeElement(id) {
    const el = {
        id,
        value: '',
        _text: '',
        _html: '',
        className: '',
        style: { setProperty() {} },
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
        closest: () => null,
        // A real canvas returns the SAME 2-D context on every call, so the stub
        // memoizes it — otherwise a test could never see what was drawn.
        getContext: () => {
            if (!el._ctx) el._ctx = makeContext();
            return el._ctx;
        }
    };
    // A real DOM's textContent is always a string, so coerce assignments.
    Object.defineProperty(el, 'textContent', {
        get() { return el._text; },
        set(v) { el._text = v === null || v === undefined ? '' : String(v); }
    });
    // Setting innerHTML REPLACES the element's content, so the existing
    // children must go. Without this the shim would keep stale rows and a test
    // asserting "the list was cleared" would pass for the wrong reason.
    Object.defineProperty(el, 'innerHTML', {
        get() { return el._html; },
        set(v) {
            el._html = v === null || v === undefined ? '' : String(v);
            el.children = [];
        }
    });
    return el;
}

// A canvas context stub that records which drawing calls were made, so the
// chart tests can assert the plot really was drawn from the API data rather
// than silently skipped. Each entry is [method, ...args] — the method name has
// to be recorded, or a test could not tell an arc() from a fillText().
function makeContext() {
    const calls = [];
    const recorder = method => (...args) => { calls.push([method, ...args]); };
    return {
        calls,
        scale: recorder('scale'), clearRect: recorder('clearRect'),
        beginPath: recorder('beginPath'), moveTo: recorder('moveTo'),
        lineTo: recorder('lineTo'), arc: recorder('arc'), fill: recorder('fill'),
        stroke: recorder('stroke'), save: recorder('save'), restore: recorder('restore'),
        translate: recorder('translate'), rotate: recorder('rotate'),
        setLineDash: recorder('setLineDash'), fillText: recorder('fillText'),
        set fillStyle(v) {}, set strokeStyle(v) {}, set lineWidth(v) {},
        set font(v) {}, set textAlign(v) {}
    };
}

for (const id of [
    'predictionForm', 'btnPredict', 'btnPredictLabel', 'predictIcon', 'btnSample',
    'btnRetryModel', 'modelStateDisplay', 'datasetRangeNote', 'selectedKDisplay',
    'modelAlertBanner', 'modelAlertTitle', 'modelAlertMessage',
    'metricRecords', 'metricSelectedK', 'metricClusterCount', 'metricSilhouette',
    'metricWss', 'clusterCardGrid', 'resultPanel', 'resultEmpty', 'resultBody',
    'resultError', 'resultErrorTitle', 'resultErrorList', 'resultClusterBadge',
    'resultClusterLabel', 'resultDistance', 'separationFill', 'distanceNote',
    'distanceList', 'inputSummary', 'resultWarnings', 'resultModelName',
    'resultClusterSize', 'resultTrainedAt', 'resultExecTime',
    'vizSubtitle', 'vizLegend', 'scatterWrap', 'vizNote',
    'kSelectionBasis', 'elbowWrap', 'kSelectionNote', 'kTableWrap', 'kTableBody',
    'profileGrid'
]) {
    elements.set(id, makeElement(id));
}

// Form fields + their hint/error spans
const FIELDS = ['age', 'annual_income', 'spending_score', 'purchase_frequency'];
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
    window: { devicePixelRatio: 1, addEventListener: () => {} },
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

const VALID = { age: 42, annual_income: 85, spending_score: 55, purchase_frequency: 28 };

async function getJson(path) {
    return (await fetch(API + path)).json();
}

(async () => {
    console.log('\n=== AI Insight Hub - Frontend Logic Test (kmeans.js) ===\n');

    // --- 1. Formatting -----------------------------------------------------
    console.log('[1] Formatting');
    eq('formatNumber 4dp', context.formatNumber(0.439, 4), '0.4390');
    eq('formatPercent converts a proportion', context.formatPercent(0.425), '42.5%');
    eq('formatPercent on 1.0', context.formatPercent(1), '100.0%');
    eq('formatNumber handles a non-number safely', context.formatNumber(null, 2), '—');
    eq('formatPercent handles a non-number safely', context.formatPercent('x'), '—');
    eq('age is shown as a whole number', context.formatFeatureValue('age', 42), '42');
    eq('spending score keeps two decimals',
        context.formatFeatureValue('spending_score', 55), '55.00');
    eq('fieldSuffix builds the DOM id suffix',
        context.fieldSuffix('annual_income'), 'Annual_income');
    eq('cluster 4 maps to the categorical palette',
        context.clusterPalette(4).color, '#10b981');
    eq('an unknown cluster id still gets a colour',
        context.clusterPalette(99).color, '#38bdf8');

    // --- 2. Client-side validation ----------------------------------------
    console.log('\n[2] Client-side validation');
    const { validateForm, findExtrapolationWarnings, readFormValues } = context;

    setForm(VALID);
    let errors = validateForm();
    check('valid form produces no errors',
        Object.values(errors).every(e => !e), JSON.stringify(errors));

    setForm({ ...VALID, age: '' });
    check('empty age is rejected', validateForm().age.includes('required'), validateForm().age);

    setForm({ ...VALID, age: -1 });
    check('negative age is rejected',
        validateForm().age.includes('0 or greater'), validateForm().age);

    setForm({ ...VALID, age: 150 });
    check('age above 120 is rejected',
        validateForm().age.includes('at most 120'), validateForm().age);

    setForm({ ...VALID, annual_income: -5 });
    check('negative income is rejected',
        validateForm().annual_income.includes('0 or greater'), validateForm().annual_income);

    setForm({ ...VALID, spending_score: 150 });
    check('spending score above 100 is rejected',
        validateForm().spending_score.includes('at most 100'), validateForm().spending_score);

    setForm({ ...VALID, purchase_frequency: -3 });
    check('negative purchase frequency is rejected',
        validateForm().purchase_frequency.includes('0 or greater'),
        validateForm().purchase_frequency);

    setForm({ ...VALID, spending_score: 'frequent' });
    check('non-numeric spending score is rejected',
        validateForm().spending_score.includes('must be a number'), validateForm().spending_score);

    setForm({});
    errors = validateForm();
    check('empty form flags all 4 fields',
        Object.values(errors).filter(e => e).length === 4, JSON.stringify(errors));

    // --- 3. Range validation (ranges loaded from the live API) ------------
    await context.loadSchema();
    const schema = await getJson('/api/kmeans/schema');
    console.log('\n[3] Range validation (ranges loaded from the live API)');
    console.log('      ' + schema.features.map(f => f.name + ': ' + f.min + '-' + f.max).join(', '));
    eq('range note badge is updated',
        elements.get('datasetRangeNote').textContent, 'Ranges from training data');

    // A value outside the trained range is still assignable: the form must
    // accept it and flag it as an extrapolation rather than block it.
    setForm({ ...VALID, age: 110 });
    eq('out-of-range age passes validation', validateForm().age, '');
    check('out-of-range age is flagged as extrapolation',
        findExtrapolationWarnings(readFormValues()).some(w => w.field === 'age'),
        JSON.stringify(findExtrapolationWarnings(readFormValues())));

    setForm({ ...VALID, spending_score: 100 });
    eq('spending score of 100 (the documented maximum) is accepted',
        validateForm().spending_score, '');

    setForm({ ...VALID, age: schema.features.find(f => f.name === 'age').min });
    eq('the in-range minimum age is accepted', validateForm().age, '');

    setForm(VALID);
    eq('an all-in-range customer has no extrapolation warnings',
        findExtrapolationWarnings(readFormValues()).length, 0);

    // --- 4. Live assignment flow (real Flask -> R engine) -----------------
    console.log('\n[4] Live assignment flow (real Flask -> R engine)');
    setForm(VALID);
    await context.submitAssignment();

    eq('result panel reaches the success state', elements.get('resultPanel').dataset.state, 'success');
    const badge = elements.get('resultClusterBadge');
    check('the cluster badge carries a numeric cluster',
        /^\d+$/.test(badge.textContent.replace('Cluster ', '')), badge.textContent);

    const live = await (await fetch(API + '/api/kmeans/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(VALID)
    })).json();

    eq('badge text matches the API assignment', badge.textContent, 'Cluster ' + live.cluster);
    eq('badge data-cluster matches the API', badge.dataset.cluster, String(live.cluster));
    eq('distance shown matches the API value',
        elements.get('resultDistance').textContent, live.distance.toFixed(4));
    eq('the generated cluster label is displayed',
        elements.get('resultClusterLabel').textContent, live.cluster_label);
    eq('segment size is displayed', elements.get('resultClusterSize').textContent,
        live.cluster_size + ' customers');
    eq('model name is displayed', elements.get('resultModelName').textContent, 'K-Means');
    eq('trained-at timestamp is displayed',
        elements.get('resultTrainedAt').textContent, live.trained_at);
    check('the distance basis explains how it was measured',
        elements.get('distanceNote').textContent.includes('ratio'),
        elements.get('distanceNote').textContent);
    check('the separation ratio is quoted',
        elements.get('distanceNote').textContent.includes(live.separation_ratio.toFixed(4)),
        elements.get('distanceNote').textContent);

    // One distance row per cluster centre, the winner flagged.
    const distRows = elements.get('distanceList').children;
    eq('one distance row is drawn per cluster', distRows.length, live.k);
    eq('exactly one row is marked as the winner',
        distRows.filter(r => r.dataset.winner === 'true').length, 1);
    eq('the winner row is the assigned cluster',
        Number(distRows.find(r => r.dataset.winner === 'true').dataset.cluster), live.cluster);
    check('each row shows a real measured distance',
        distRows.every(r => /^\d+\.\d{4}$/.test(r.children[2].textContent)),
        JSON.stringify(distRows.map(r => r.children[2].textContent)));

    // Input summary: two elements per feature.
    eq('input summary lists all four fields',
        elements.get('inputSummary').children.length, 8);

    // --- 5. Different customers reach different clusters ------------------
    console.log('\n[5] Assignment responds to input');
    setForm({ age: 65, annual_income: 20, spending_score: 10, purchase_frequency: 3 });
    await context.submitAssignment();
    const contrastCluster = Number(elements.get('resultClusterBadge').dataset.cluster);
    console.log('      contrast customer -> cluster ' + contrastCluster);

    check('a contrasting customer is assigned to a different cluster',
        contrastCluster !== live.cluster,
        'both landed in cluster ' + contrastCluster);

    // --- 6. Reproducibility ------------------------------------------------
    console.log('\n[6] Repeated identical input');
    setForm(VALID);
    await context.submitAssignment();
    const firstCluster = elements.get('resultClusterBadge').textContent;
    await context.submitAssignment();
    eq('identical input returns the same cluster',
        elements.get('resultClusterBadge').textContent, firstCluster);
    eq('identical input returns the same distance',
        elements.get('resultDistance').textContent, live.distance.toFixed(4));

    // --- 7. Error rendering -----------------------------------------------
    console.log('\n[7] Error rendering');
    setForm(VALID);
    context.renderError('Invalid customer details', ['Age must be at most 120.']);
    eq('error state is set', elements.get('resultPanel').dataset.state, 'error');
    eq('error title is rendered', elements.get('resultErrorTitle').textContent,
        'Invalid customer details');
    eq('each error message becomes a list item',
        elements.get('resultErrorList').children.length, 1);

    // A client-side validation failure must never reach the server.
    setForm({ ...VALID, spending_score: 150 });
    await context.submitAssignment();
    eq('client-side validation blocks the request',
        elements.get('resultPanel').dataset.state, 'error');
    check('inline field error is painted',
        elements.get('errorSpending_score').textContent.includes('at most 100'),
        elements.get('errorSpending_score').textContent);

    // A valid form must recover from the error state.
    setForm(VALID);
    await context.submitAssignment();
    eq('a valid form recovers from the error state',
        elements.get('resultPanel').dataset.state, 'success');

    // --- 8. Metrics, stat tiles and K evaluation (live API) ---------------
    console.log('\n[8] Metrics, stat tiles and K evaluation (live API)');
    await context.loadMetrics();
    const metrics = await getJson('/api/kmeans/metrics');

    eq('dataset size tile shows the trained value',
        elements.get('metricRecords').textContent, String(metrics.dataset.rows_clean));
    eq('selected K tile shows the trained value',
        elements.get('metricSelectedK').textContent, String(metrics.selected_k));
    eq('cluster count tile shows the trained value',
        elements.get('metricClusterCount').textContent, String(metrics.metrics.cluster_count));
    eq('silhouette tile shows the trained value',
        elements.get('metricSilhouette').textContent, metrics.metrics.silhouette_score.toFixed(4));
    eq('WSS tile shows the trained value',
        elements.get('metricWss').textContent, metrics.metrics.wss.toFixed(4));
    eq('the selected K is shown in the hero',
        elements.get('selectedKDisplay').textContent, 'K = ' + metrics.selected_k);
    eq('model state is reported ready',
        elements.get('modelStateDisplay').textContent, 'Model trained & ready');
    eq('the K selection basis is displayed',
        elements.get('kSelectionBasis').textContent, metrics.k_selection.basis);
    check('the K reasoning is explained to the user',
        elements.get('kSelectionNote').textContent.length > 40,
        elements.get('kSelectionNote').textContent);

    // The elbow chart must actually be drawn from the K-comparison data.
    const elbowCanvas = elements.get('elbowWrap').children[0];
    check('an elbow chart canvas is created', !!elbowCanvas, 'no canvas element');
    check('the elbow canvas is tagged as an image',
        elbowCanvas && elbowCanvas.attributes.role === 'img', 'canvas has no role=img');
    check('the elbow chart is drawn with a 2-D context',
        elbowCanvas && elbowCanvas.getContext('2d').calls.length > 0,
        'no drawing calls were made');
    eq('one K row is drawn per evaluated K',
        elements.get('kTableBody').children.length, metrics.k_comparison.length);
    eq('exactly one K row is marked selected',
        elements.get('kTableBody').children.filter(r => r.dataset.selected === 'true').length, 1);
    eq('the marked K row is the model\'s K',
        elements.get('kTableBody').children
            .find(r => r.dataset.selected === 'true').children[0].textContent,
        String(metrics.selected_k));
    eq('the selected K row shows the real silhouette',
        elements.get('kTableBody').children
            .find(r => r.dataset.selected === 'true').children[2].textContent,
        metrics.k_comparison.find(e => e.k === metrics.selected_k).silhouette_mean.toFixed(4));

    // --- 9. Cluster overview cards ----------------------------------------
    console.log('\n[9] Cluster overview cards');
    const cards = elements.get('clusterCardGrid').children;
    eq('a card is drawn per cluster', cards.length, metrics.selected_k);
    metrics.cluster_profiles.forEach((profile, index) => {
        const card = cards[index];
        eq('card ' + (index + 1) + ' shows the real cluster id',
            card.children[0].children[0].textContent, 'CLUSTER ' + profile.cluster);
        eq('card ' + (index + 1) + ' shows the real customer count',
            card.children[1].textContent, String(profile.size));
        eq('card ' + (index + 1) + ' shows the generated label',
            card.children[3].textContent, profile.label);
    });
    const firstCardDescriptors = cards[0].children[4].children.length;
    check('each card lists the descriptors behind its label',
        firstCardDescriptors === metrics.cluster_profiles[0].descriptors.length,
        firstCardDescriptors + ' chips vs ' + metrics.cluster_profiles[0].descriptors.length);

    // --- 10. Cluster visualization ----------------------------------------
    console.log('\n[10] Cluster visualization');
    await context.loadClusters();
    // The per-cluster sizes and labels live on the clusters endpoint, which is
    // what the page reads them from.
    const clusterData = await getJson('/api/kmeans/clusters');

    const scatterCanvas = elements.get('scatterWrap').children[0];
    check('a scatter canvas is created', !!scatterCanvas, 'no canvas element');
    check('the scatter canvas is tagged as an image',
        scatterCanvas && scatterCanvas.attributes.role === 'img', 'canvas has no role=img');
    check('the scatter canvas is labelled for screen readers',
        scatterCanvas && (scatterCanvas.attributes['aria-label'] || '')
            .includes('principal components'),
        scatterCanvas ? scatterCanvas.attributes['aria-label'] : 'no canvas');

    // The chart must really be drawn from the API's projection data. Every
    // customer is one filled dot, and every cluster centre is drawn twice — a
    // halo behind a ringed marker — so two arcs per centre.
    const scatterCalls = scatterCanvas ? scatterCanvas.getContext('2d').calls : [];
    const arcCalls = scatterCalls.filter(a => a[0] === 'arc');
    const expectedArcs = metrics.dataset.rows_clean + metrics.selected_k * 2;
    check('every customer dot and every centre marker is drawn',
        arcCalls.length === expectedArcs,
        arcCalls.length + ' arcs for ' + metrics.dataset.rows_clean +
        ' customers and ' + metrics.selected_k + ' centres (expected ' + expectedArcs + ')');

    // Each centre is labelled with its number, so a reader can tell them apart.
    const labels = scatterCalls.filter(a => a[0] === 'fillText' && /^C\d+$/.test(a[1]));
    eq('every cluster centre is labelled on the plot', labels.length, metrics.selected_k);

    eq('one legend entry is drawn per cluster',
        elements.get('vizLegend').children.length, metrics.selected_k);
    Object.keys(clusterData.cluster_sizes).forEach(key => {
        const item = elements.get('vizLegend').children
            .find(i => i.dataset.cluster === key);
        check('legend entry for cluster ' + key + ' shows the real count',
            item.children[2].textContent === '(' + clusterData.cluster_sizes[key] + ')',
            item.children[2].textContent);
    });

    check('the projection is disclosed as display-only',
        elements.get('vizNote').textContent.includes('display only'),
        elements.get('vizNote').textContent);
    check('the explained variance is quoted',
        elements.get('vizNote').textContent.includes('%'),
        elements.get('vizNote').textContent);

    // --- 11. Cluster profiles ---------------------------------------------
    console.log('\n[11] Cluster profiles');
    await context.loadProfiles();
    const profiles = await getJson('/api/kmeans/profiles');

    const profileCards = elements.get('profileGrid').children;
    eq('a profile card is drawn per cluster', profileCards.length, profiles.selected_k);

    profiles.profiles.forEach((profile, index) => {
        const card = profileCards[index];
        eq('profile ' + (index + 1) + ' names the cluster',
            card.children[0].children[1].textContent, 'Cluster ' + profile.cluster);
        eq('profile ' + (index + 1) + ' shows the generated label',
            card.children[1].textContent, profile.label);
        eq('profile ' + (index + 1) + ' shows the real size and share',
            card.children[2].textContent,
            profile.size + ' customers · ' + (profile.size_share * 100).toFixed(1) + '% of the dataset');
    });

    // One table row per clustering feature, and the two-row footer.
    const firstProfileTable = profileCards[0].children[3];
    eq('the profile table lists every feature',
        firstProfileTable.children[1].children.length, FIELDS.length);
    check('each feature row is classified high, low or neutral',
        firstProfileTable.children[1].children.every(
            row => ['high', 'low', 'neutral'].includes(row.dataset.sign)),
        JSON.stringify(firstProfileTable.children[1].children.map(r => r.dataset.sign)));

    // --- 12. Empty and missing data handling ------------------------------
    console.log('\n[12] Empty and missing data handling');
    context.renderClusterCards([]);
    eq('no cluster cards are drawn for an empty profile list',
        elements.get('clusterCardGrid').children.length, 0);

    context.renderProfiles({ profiles: [] });
    check('an empty profile list shows a placeholder',
        elements.get('profileGrid').innerHTML.includes('unavailable'),
        elements.get('profileGrid').innerHTML);

    context.renderKTable([], 4, 5);
    check('an empty K comparison shows a placeholder',
        elements.get('kTableWrap').innerHTML.includes('unavailable'),
        elements.get('kTableWrap').innerHTML);

    context.renderClusterScatter({ visualization: {} });
    check('a missing projection shows a placeholder',
        elements.get('scatterWrap').innerHTML.includes('unavailable'),
        elements.get('scatterWrap').innerHTML);

    // --- 13. Distance ranking robustness ----------------------------------
    console.log('\n[13] Distance ranking robustness');
    context.renderDistanceRanking({ 'Cluster 1': 2, 'Cluster 2': 1 }, 2);
    const rows = elements.get('distanceList').children;
    eq('every reported centre gets a row', rows.length, 2);
    eq('the nearest centre is flagged as the winner',
        rows.find(r => r.dataset.winner === 'true').dataset.cluster, '2');

    context.renderDistanceRanking(null, 1);
    eq('a null distance map draws no rows',
        elements.get('distanceList').children.length, 0);

    // --- 14. Warnings -----------------------------------------------------
    console.log('\n[14] Extrapolation warnings');
    // A block is appended as real child elements, so the children are what
    // carry the message — not the host's own innerHTML.
    context.renderWarnings([{ message: 'Age (110) is outside the 18 - 68 range.' }]);
    const warningBlock = elements.get('resultWarnings').children[0];
    check('a warning block is built', !!warningBlock, 'no warning block');
    eq('the block is titled', warningBlock.children[0].textContent, 'Outside the training range');
    eq('each warning becomes a line',
        warningBlock.children[1].textContent, 'Age (110) is outside the 18 - 68 range.');
    eq('the warnings area is made visible',
        elements.get('resultWarnings').style.display, 'block');

    context.renderWarnings([]);
    check('no warnings block is shown when the list is empty',
        elements.get('resultWarnings').style.display === 'none',
        elements.get('resultWarnings').style.display);

    // --- 15. Model configuration endpoint ----------------------------------
    // The form is built from /api/kmeans/schema; /api/kmeans/config reports the
    // same trained ranges from the model's own artifact, so the two must agree
    // or the page would be describing a model it is not talking to.
    console.log('\n[15] Model configuration endpoint');
    const config = await getJson('/api/kmeans/config');
    console.log('      ' + config.model + ' | K = ' + config.selected_k +
                ' | grid [' + config.k_grid.join(', ') + ']');
    check('config reports the model name', config.model === 'K-Means', config.model);
    check('config reports the selected K', Number.isInteger(config.selected_k),
        String(config.selected_k));
    check('config reports a K grid containing the selected K',
        config.k_grid.includes(config.selected_k), JSON.stringify(config.k_grid));
    check('config reports four clustering features',
        config.feature_names.length === 4, JSON.stringify(config.feature_names));
    check('the identifier is excluded from the clustering features',
        !config.feature_names.includes('customer_id'),
        JSON.stringify(config.feature_names));
    check('config states the identifier it held out',
        config.excluded_from_clustering.includes('customer_id'),
        JSON.stringify(config.excluded_from_clustering));

    for (const feature of config.features) {
        check(`${feature.name} has a real min/max in the config`,
            feature.min < feature.max, `${feature.name}: ${feature.min}-${feature.max}`);
        const fromSchema = schema.features.find(f => f.name === feature.name);
        eq(`${feature.name} range matches the schema endpoint`, feature.min, fromSchema.min);
        eq(`${feature.name} max matches the schema endpoint`, feature.max, fromSchema.max);
    }

    // --- 16. Cluster profile artifact --------------------------------------
    // /api/kmeans/profiles is served from r_models/kmeans/profiles.json, and
    // must agree with the model metrics about K and the cluster sizes.
    console.log('\n[16] Cluster profile artifact');
    const profilesArtifact = await getJson('/api/kmeans/profiles');
    const metricsForProfiles = await getJson('/api/kmeans/metrics');

    eq('the profiles artifact is the one served',
        profilesArtifact.profiles_file, 'r_models/kmeans/profiles.json');
    eq('profiles report the same K as the metrics',
        profilesArtifact.selected_k, metricsForProfiles.selected_k);
    eq('there is one profile per cluster',
        profilesArtifact.profiles.length, metricsForProfiles.selected_k);
    check('profile sizes sum to the clustered record count',
        profilesArtifact.profiles.reduce((sum, p) => sum + p.size, 0) === profilesArtifact.total_records,
        String(profilesArtifact.profiles.reduce((sum, p) => sum + p.size, 0)));
    check('the label derivation rule is published',
        typeof profilesArtifact.naming === 'string' && profilesArtifact.naming.length > 0,
        String(profilesArtifact.naming));

    for (const profile of profilesArtifact.profiles) {
        check(`cluster ${profile.cluster} carries a size and a label`,
            Number.isInteger(profile.size) && profile.size > 0 && !!profile.label,
            `${profile.size} / ${profile.label}`);
        const metricsCluster = metrics.cluster_profiles.find(c => c.cluster === profile.cluster);
        check(`cluster ${profile.cluster} size matches the metrics artifact`,
            metricsCluster && metricsCluster.size === profile.size,
            `${profile.size} vs ${metricsCluster && metricsCluster.size}`);
    }

    // --- Summary -----------------------------------------------------------
    console.log('\n=== Results: ' + passed + ' passed, ' + failed + ' failed ===\n');
    process.exit(failed === 0 ? 0 : 1);
})().catch(err => {
    console.error('\nFATAL: test harness error:', err);
    process.exit(1);
});
