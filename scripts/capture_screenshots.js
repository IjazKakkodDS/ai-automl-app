/**
 * capture_screenshots.js
 *
 * Playwright headless-Chromium script for Phase 6B.3E.
 * Captures UI screenshots from the running AutoML platform.
 *
 * Usage:
 *   node scripts/capture_screenshots.js
 *
 * Prerequisites:
 *   - Backend running at http://localhost:8000
 *   - Frontend running at http://localhost:3000
 *   - Playwright browsers installed (npx playwright install chromium)
 */

const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');

const BASE_URL = 'http://localhost:3000';
const OUT_DIR = path.join(__dirname, '..', 'docs', 'evidence', 'screenshots');
const VIEWPORT = { width: 1440, height: 900 };
const WAIT_MS = 2500;

const pages = [
  { file: '01_homepage.png',          url: '/',                    label: 'Homepage / Platform Hero' },
  { file: '02_preprocessing.png',     url: '/preprocessing',       label: 'Preprocessing - Upload & Configure' },
  { file: '03_eda.png',               url: '/eda',                 label: 'EDA - Exploratory Data Analysis' },
  { file: '04_feature_engineering.png', url: '/feature-engineering', label: 'Feature Engineering' },
  { file: '05_model_training.png',    url: '/model-training',      label: 'Model Training' },
  { file: '06_evaluate.png',          url: '/evaluate',            label: 'Evaluate - Model Evaluation' },
  { file: '07_forecasting.png',       url: '/forecasting',         label: 'Forecasting' },
  { file: '08_insights.png',          url: '/insights',            label: 'AI Insights' },
  { file: '09_rag.png',               url: '/rag',                 label: 'RAG Query Interface' },
  { file: '10_reports.png',           url: '/reports',             label: 'Reports' },
];

async function capture() {
  fs.mkdirSync(OUT_DIR, { recursive: true });

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: VIEWPORT });
  const page = await context.newPage();

  const results = [];

  for (const target of pages) {
    const fullUrl = `${BASE_URL}${target.url}`;
    const outPath = path.join(OUT_DIR, target.file);
    console.log(`Capturing: ${target.label} -> ${target.file}`);
    try {
      await page.goto(fullUrl, { waitUntil: 'networkidle', timeout: 15000 });
      await page.waitForTimeout(WAIT_MS);
      await page.screenshot({ path: outPath, fullPage: false });
      const stat = fs.statSync(outPath);
      results.push({ file: target.file, label: target.label, url: fullUrl, status: 'OK', bytes: stat.size });
      console.log(`  OK  ${stat.size} bytes`);
    } catch (err) {
      results.push({ file: target.file, label: target.label, url: fullUrl, status: 'ERROR', error: err.message });
      console.log(`  ERROR: ${err.message}`);
    }
  }

  await browser.close();

  console.log('\n=== SUMMARY ===');
  results.forEach(r => {
    const s = r.status === 'OK' ? `OK (${r.bytes} bytes)` : `FAIL: ${r.error}`;
    console.log(`  ${r.file}: ${s}`);
  });

  // Write manifest
  const manifest = { captured: new Date().toISOString(), results };
  fs.writeFileSync(path.join(OUT_DIR, 'manifest.json'), JSON.stringify(manifest, null, 2));
  console.log('\nManifest written to docs/evidence/screenshots/manifest.json');
}

capture().catch(err => {
  console.error('Fatal:', err);
  process.exit(1);
});
