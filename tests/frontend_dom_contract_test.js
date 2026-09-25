/**
 * Static DOM contract test for the frontend pages.
 *
 * The behavioural tests (frontend_logic_test.js, knn_frontend_test.js,
 * decision_tree_frontend_test.js) run each page script in Node against a
 * synthetic DOM. That shim registers elements using the *same* fieldSuffix()
 * helper the page script uses, so it can never notice when the script and the
 * real HTML disagree about an element id — which is exactly the class of bug
 * that left the Phase 3 and Phase 4 buttons dead in a real browser.
 *
 * This test closes that gap: it reads the actual .html and .js files off disk
 * and asserts that every id the script looks up really exists in the markup.
 * It needs no server and no R engine.
 *
 * Run:  node tests/frontend_dom_contract_test.js
 */
const fs = require('fs');
const path = require('path');

const FRONTEND = path.join(__dirname, '..', 'frontend');

const PAGES = [
    { name: 'Linear Regression (Phase 3)', html: 'regression.html', js: 'js/regression.js' },
    { name: 'Decision Tree (Phase 4)', html: 'decision-tree.html', js: 'js/decision-tree.js' },
    { name: 'KNN (Phase 5)', html: 'knn.html', js: 'js/knn.js' },
    { name: 'K-Means (Phase 6)', html: 'kmeans.html', js: 'js/kmeans.js' }
];

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

/** Collects every id="..." declared in the markup. */
function collectHtmlIds(htmlSource) {
    const ids = new Set();
    const re = /id="([^"]+)"/g;
    let m;
    while ((m = re.exec(htmlSource)) !== null) ids.add(m[1]);
    return ids;
}

/** Collects every for="..." target declared in the markup. */
function collectLabelTargets(htmlSource) {
    const targets = [];
    const re = /<label[^>]*\bfor="([^"]+)"/g;
    let m;
    while ((m = re.exec(htmlSource)) !== null) targets.push(m[1]);
    return targets;
}

/** Collects statically-named getElementById('...') lookups in the script. */
function collectStaticLookups(jsSource) {
    const ids = new Set();
    const re = /getElementById\(\s*'([^']+)'\s*\)/g;
    let m;
    while ((m = re.exec(jsSource)) !== null) ids.add(m[1]);
    return ids;
}

/**
 * Extracts the FIELDS array the script validates and reads the form with.
 * Each entry must resolve to an input, a hint and an error span in the markup.
 */
function collectFieldKeys(jsSource) {
    const m = jsSource.match(/const\s+FIELDS\s*=\s*\[([\s\S]*?)\]/);
    if (!m) return null;
    const keys = [];
    const re = /'([^']+)'/g;
    let g;
    while ((g = re.exec(m[1])) !== null) keys.push(g[1]);
    return keys;
}

/** Mirrors the page scripts' fieldSuffix(): capitalise the first letter only. */
function fieldSuffix(key) {
    return key.charAt(0).toUpperCase() + key.slice(1);
}

/** Every <script src> and stylesheet href the page declares. */
function collectAssetRefs(htmlSource) {
    const refs = [];
    const re = /(?:src|href)="([^"]+\.(?:js|css))"/g;
    let m;
    while ((m = re.exec(htmlSource)) !== null) refs.push(m[1]);
    return refs;
}

console.log('\n=== AI Insight Hub - Frontend DOM Contract Test ===\n');

for (const page of PAGES) {
    console.log('[' + page.name + '] ' + page.html);

    const htmlPath = path.join(FRONTEND, page.html);
    const jsPath = path.join(FRONTEND, page.js);

    const htmlSource = fs.readFileSync(htmlPath, 'utf8');
    const jsSource = fs.readFileSync(jsPath, 'utf8');

    const htmlIds = collectHtmlIds(htmlSource);

    // --- 1. Every static getElementById must resolve in the real markup ---
    const staticLookups = [...collectStaticLookups(jsSource)]
        .filter(id => !id.startsWith('input') && !id.startsWith('hint') &&
                      !id.startsWith('error'));  // dynamic ones checked below

    const missingStatic = staticLookups.filter(id => !htmlIds.has(id));
    check('every static getElementById resolves to a real element',
        missingStatic.length === 0,
        missingStatic.length ? 'missing: ' + missingStatic.join(', ') : undefined);

    // --- 2. Every FIELDS key must resolve to input/hint/error elements ---
    const fieldKeys = collectFieldKeys(jsSource);
    check('the script declares a FIELDS list', Array.isArray(fieldKeys) && fieldKeys.length > 0);

    if (fieldKeys) {
        const missingDynamic = [];
        for (const key of fieldKeys) {
            const suffix = fieldSuffix(key);
            for (const prefix of ['input', 'hint', 'error']) {
                const id = prefix + suffix;
                if (!htmlIds.has(id)) missingDynamic.push(id);
            }
        }
        check('every form field resolves to its input, hint and error elements',
            missingDynamic.length === 0,
            missingDynamic.length ? 'missing: ' + missingDynamic.join(', ') : undefined);

        // The form controls must also be reachable through the <label for> wiring.
        const labelTargets = new Set(collectLabelTargets(htmlSource));
        const missingLabels = fieldKeys
            .map(k => 'input' + fieldSuffix(k))
            .filter(id => !labelTargets.has(id));
        check('every form control is targeted by its <label for>',
            missingLabels.length === 0,
            missingLabels.length ? 'unlabelled: ' + missingLabels.join(', ') : undefined);
    }

    // --- 3. The page's own script and stylesheet must exist on disk ---
    const assetRefs = collectAssetRefs(htmlSource);
    const brokenAssets = assetRefs.filter(ref => {
        if (/^https?:/.test(ref)) return false;      // CDN font, not local
        return !fs.existsSync(path.join(FRONTEND, ref));
    });
    check('every local script and stylesheet referenced by the page exists',
        brokenAssets.length === 0,
        brokenAssets.length ? 'missing files: ' + brokenAssets.join(', ') : undefined);

    // --- 4. The predict button must be a submit inside the wired form ---
    const formMatch = htmlSource.match(/<form[^>]*id="predictionForm"[\s\S]*?<\/form>/);
    check('a form#predictionForm exists', !!formMatch);
    if (formMatch) {
        const form = formMatch[0];
        check('the form wires the predict button as a submit control',
            /id="btnPredict"[^>]*type="submit"|type="submit"[^>]*id="btnPredict"/.test(form),
            'btnPredict is not a type="submit" control');
        check('the form contains the btnPredict control', form.includes('id="btnPredict"'));
    }

    console.log('');
}

console.log('=== Results: ' + passed + ' passed, ' + failed + ' failed ===\n');
process.exit(failed === 0 ? 0 : 1);
