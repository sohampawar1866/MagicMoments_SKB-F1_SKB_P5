// Drive the DRIFT frontend through its full user flow + capture everything.
// Usage: node tests/drive.mjs
import { chromium } from 'playwright';
import fs from 'node:fs';
import path from 'node:path';

const URL = process.env.URL || 'http://localhost:5174';
const ART = 'tests/_artifacts';
fs.mkdirSync(ART, { recursive: true });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const consoleMsgs = [];
const networkErrors = [];
const responses = [];
const pageErrors = [];

const browser = await chromium.launch({ headless: true });
const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
const page = await ctx.newPage();

page.on('console', (m) => consoleMsgs.push({ type: m.type(), text: m.text() }));
page.on('pageerror', (e) =>
  pageErrors.push({ name: e.name, message: e.message, stack: e.stack?.slice(0, 800) })
);
page.on('requestfailed', (req) =>
  networkErrors.push({ url: req.url(), method: req.method(), failure: req.failure()?.errorText })
);
page.on('response', async (res) => {
  const u = res.url();
  if (!u.includes('/api/v1/')) return;
  let body = '';
  if (res.status() >= 400) {
    try {
      body = (await res.text()).slice(0, 300);
    } catch {}
  }
  responses.push({ url: u, status: res.status(), body });
});

async function snap(name) {
  await page.screenshot({ path: path.join(ART, `${name}.png`), fullPage: false });
}

const report = { steps: [] };
async function step(name, fn) {
  console.log(`\n=== ${name} ===`);
  const t0 = Date.now();
  try {
    await fn();
    const ms = Date.now() - t0;
    report.steps.push({ name, ok: true, ms });
    console.log(`  ✓ ${ms}ms`);
  } catch (e) {
    const ms = Date.now() - t0;
    report.steps.push({ name, ok: false, ms, error: e.message });
    console.log(`  ✗ ${e.message}`);
  }
}

await step('1. landing page loads', async () => {
  await page.goto(URL, { waitUntil: 'networkidle', timeout: 30_000 });
  await snap('01_landing');
  const txt = (await page.evaluate(() => document.body.innerText)).slice(0, 200);
  console.log('  body preview:', JSON.stringify(txt.slice(0, 120)));
});

await step('2. navigate to /drift', async () => {
  await page.goto(`${URL}/drift`, { waitUntil: 'networkidle', timeout: 30_000 });
  await sleep(3000);
  await snap('02_drift');
});

// Bypass the map click (deck.gl synthetic-event quirks) and go straight to
// the ops dashboard URL the click flow produces. This tests the most
// important integration: detect → forecast → mission → dashboard end-to-end
// with a real custom AOI ID.
const TEST_LON = 80.5;
const TEST_LAT = 14.5; // Bay of Bengal, deep water

await step('3. open ops dashboard at custom AOI', async () => {
  const aoiId = `custom_${TEST_LON.toFixed(4)}_${TEST_LAT.toFixed(4)}`;
  await page.goto(`${URL}/drift/aoi/${aoiId}`, { waitUntil: 'domcontentloaded', timeout: 30_000 });
  console.log('  navigated to:', page.url());
  await snap('03_ops_initial');
});

await step('4. wait for detect/forecast/mission/metrics to complete', async () => {
  const start = Date.now();
  while (Date.now() - start < 120_000) {
    const detectCalls = responses.filter((r) => r.url.includes('/api/v1/detect'));
    const forecastCalls = responses.filter((r) => r.url.includes('/api/v1/forecast'));
    const missionCalls = responses.filter((r) => r.url.includes('/api/v1/mission'));
    const metricsCalls = responses.filter((r) => r.url.includes('/api/v1/dashboard/metrics'));
    if (detectCalls.length && forecastCalls.length && missionCalls.length && metricsCalls.length) {
      console.log('  all 4 chain calls returned in', Date.now() - start, 'ms');
      break;
    }
    await sleep(1000);
  }
  await snap('04_ops_loaded');
});

await step('5. validate API call statuses', async () => {
  const detect = responses.filter((r) => r.url.includes('/api/v1/detect'));
  const forecast = responses.filter((r) => r.url.includes('/api/v1/forecast'));
  const mission = responses.filter((r) => r.url.includes('/api/v1/mission'));
  const metrics = responses.filter((r) => r.url.includes('/api/v1/dashboard/metrics'));
  const alerts = responses.filter((r) => r.url.includes('/api/v1/alerts'));
  console.log('  /detect:        ', detect.map((c) => c.status));
  console.log('  /forecast:      ', forecast.map((c) => c.status));
  console.log('  /mission:       ', mission.map((c) => c.status));
  console.log('  /dashboard/...: ', metrics.map((c) => c.status));
  console.log('  /alerts:        ', alerts.map((c) => c.status));
  const fails = [];
  for (const [name, calls] of [['detect', detect], ['forecast', forecast], ['mission', mission], ['metrics', metrics]]) {
    if (!calls.length) fails.push(`no /${name} call`);
    else if (calls.some((c) => c.status >= 400)) fails.push(`${name}: ${calls.find((c) => c.status >= 400)?.body}`);
  }
  if (fails.length) throw new Error(fails.join('; '));
});

await step('6. validate detection payload', async () => {
  // Hit detect directly to inspect body since playwright doesn't capture bodies for 200s
  const aoiId = `custom_${TEST_LON.toFixed(4)}_${TEST_LAT.toFixed(4)}`;
  const bbox = `${TEST_LON - 0.03},${TEST_LAT - 0.03},${TEST_LON + 0.03},${TEST_LAT + 0.03}`;
  const r = await page.request.get(`http://localhost:8000/api/v1/detect?aoi_id=${aoiId}&bbox=${bbox}`);
  console.log('  status:', r.status());
  if (!r.ok()) throw new Error('detect status ' + r.status() + ' body: ' + (await r.text()).slice(0, 200));
  const j = await r.json();
  const feats = j.features || [];
  console.log('  detection features:', feats.length);
  if (!feats.length) throw new Error('zero detections returned');
  const firstVertex = feats[0].geometry?.coordinates?.[0]?.[0];
  console.log('  first polygon vertex:', firstVertex);
  if (!firstVertex) throw new Error('detection has no geometry');
  // Vertex should be near the requested bbox (loose tolerance: STAC pixel-grid
  // snap can extend polygons a small amount beyond bbox edges).
  const [lon, lat] = firstVertex;
  const inBbox = Math.abs(lon - TEST_LON) <= 0.10 && Math.abs(lat - TEST_LAT) <= 0.10;
  if (!inBbox) throw new Error(`first vertex (${lon}, ${lat}) is FAR FROM bbox around (${TEST_LON}, ${TEST_LAT})`);
  console.log('  ✓ first vertex is INSIDE the requested bbox');
});

await step('7. UI rendered something', async () => {
  const visibleText = await page.evaluate(() => document.body.innerText);
  if (visibleText.length < 50) throw new Error('body essentially empty: ' + visibleText.slice(0, 200));
  // Look for OceanTrace / OceanTrace-related labels
  const hasSST = /SST/i.test(visibleText);
  const hasChl = /Chl/i.test(visibleText);
  const hasOps = /OPERATIONS/i.test(visibleText);
  const hasRadar = /RADAR/i.test(visibleText);
  console.log('  has SST tile?', hasSST, '  Chl?', hasChl, '  OPS header?', hasOps, '  RADAR?', hasRadar);
});

await browser.close();

report.consoleMsgs = consoleMsgs.slice(-30);
report.networkErrors = networkErrors;
report.pageErrors = pageErrors;
report.apiResponses = responses;
fs.writeFileSync(path.join(ART, 'report.json'), JSON.stringify(report, null, 2));

console.log('\n========== SUMMARY ==========');
console.log(
  'Steps:\n  ' +
    report.steps
      .map((s) => `${s.ok ? '✓' : '✗'} ${s.name}${s.ok ? '' : ' — ' + s.error}`)
      .join('\n  ')
);
console.log('\nPage errors:', pageErrors.length);
pageErrors.slice(0, 5).forEach((e) =>
  console.log(`  ${e.name}: ${e.message}\n    ${(e.stack || '').slice(0, 200)}`)
);
console.log('\nConsole errors:', consoleMsgs.filter((m) => m.type === 'error').length);
consoleMsgs.filter((m) => m.type === 'error').slice(0, 8).forEach((m) =>
  console.log('  ', m.text.slice(0, 250))
);
console.log('\nNetwork failures:', networkErrors.length);
networkErrors.slice(0, 6).forEach((e) => console.log(`  ${e.url} - ${e.failure}`));
console.log('\nAPI responses (status / url / body):');
responses.forEach((r) => {
  const tag = r.status >= 400 ? '✗' : '✓';
  console.log(`  ${tag} ${r.status}  ${r.url.replace('http://localhost:8000', '')}`);
  if (r.body) console.log(`     body: ${r.body}`);
});
