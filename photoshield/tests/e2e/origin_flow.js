// Browser end-to-end check of origin reports (PhotoShield v2), manual like browser_flow.js.
// Needs a local backend (PHOTOSHIELD_ANCHOR=local) on :5000, the frontend on :8080, and the
// images written by the snippet in tests/e2e/README.md: me.png, fake.jpg, fake_cropped.jpg in $PS_E2E.
const { chromium } = require('playwright');
const DIR = process.env.PS_E2E || '/tmp/ps_e2e';
(async () => {
  const user = 'origin' + Math.floor(Math.random() * 1e6);
  const reg = await (await fetch('http://localhost:5000/api/auth/register', { method: 'POST',
    headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ username: user, password: 'secret123' }) })).json();
  const browser = await chromium.launch();
  const ctx = await browser.newContext({ acceptDownloads: true, viewport: { width: 1440, height: 900 } });
  await ctx.addInitScript(([t, u]) => { localStorage.setItem('bv_token', t); localStorage.setItem('bv_user', u); localStorage.setItem('bv_tour_done', '1'); }, [reg.token, user]);
  const page = await ctx.newPage();
  const errors = [];
  page.on('pageerror', e => errors.push('pageerror: ' + e.message));
  page.on('dialog', d => d.accept());
  // In-app dialogs (bvConfirm / bvPrompt) replaced the browser's confirm()/prompt().
  const dialogOk = async (value) => {
    const ov = await page.waitForSelector('.bv-dialog-overlay', { timeout: 15000 });
    if (value !== undefined) await (await ov.$('#bvDlgInput')).fill(value);
    await (await ov.$('#bvDlgOk')).click();
    await page.waitForFunction(el => !el.isConnected, ov);   // this dialog closed (a chained one may already be open)
  };
  await page.goto('http://localhost:8080/#photoshield');
  await page.waitForSelector('#psKeyInfo');
  await page.click('#psKeyGenBtn');
  await page.waitForFunction(() => document.getElementById('psKeyInfo').textContent.includes('Public key'));
  // register with a region map
  await page.setInputFiles('#psRegFiles', DIR + '/me.png');
  await page.waitForFunction(() => document.getElementById('psRegChecks').textContent.includes('Region map'));
  const regionRow = await page.$eval('#psRegChecks', e => e.textContent.replace(/\s+/g, ' ').trim());
  const mdl = page.waitForEvent('download');
  await page.click('#psRegBtn');
  await dialogOk();                                    // the key is not backed up yet: continue anyway
  await page.waitForFunction(() => /ANCHORED|failed|rror/.test(document.getElementById('psRegResult').textContent), null, { timeout: 120000 });
  const regText = (await page.$eval('#psRegResult', e => e.textContent)).replace(/\s+/g, ' ').trim();
  const manifestPath = DIR + '/manifest.json';
  await (await mdl).saveAs(manifestPath);
  // make a report: original + edited copy, no manifest (rebuilt from the owner's record)
  await page.evaluate(() => psSection('origin'));
  await page.setInputFiles('#psOrFiles', [DIR + '/me.png', DIR + '/fake.jpg']);
  await page.waitForFunction(() => /DERIVED FROM|NOT SHOWN|✘/.test(document.getElementById('psOrMakeOut').textContent), null, { timeout: 60000 });
  const made = (await page.$eval('#psOrMakeOut', e => e.textContent)).replace(/\s+/g, ' ').trim();
  await page.locator('#psOriginCard').screenshot({ path: DIR + '/origin_make.png' });
  const rdl = page.waitForEvent('download');
  await page.click('text=Download report (JSON)');
  const reportPath = DIR + '/origin-report.json';
  await (await rdl).saveAs(reportPath);
  // check it in the UI, bound to the copy
  await page.click('#psOrCheckTab');
  await page.setInputFiles('#psOrCheckFiles', [reportPath, DIR + '/fake.jpg']);
  await page.waitForFunction(() => /REPORT VALID|REPORT INVALID|✘/.test(document.getElementById('psOrCheckOut').textContent), null, { timeout: 60000 });
  const checked = (await page.$eval('#psOrCheckOut', e => e.textContent)).replace(/\s+/g, ' ').trim();
  await page.locator('#psOriginCard').screenshot({ path: DIR + '/origin_check.png' });
  // a cropped copy: the page explains the CLI route
  await page.click('#psOrMakeTab');
  await page.setInputFiles('#psOrFiles', [DIR + '/me.png', DIR + '/fake_cropped.jpg', manifestPath]);
  await page.waitForFunction(() => /cropped or rotated|DERIVED|NOT SHOWN|✘/.test(document.getElementById('psOrMakeOut').textContent), null, { timeout: 60000 });
  const cropped = (await page.$eval('#psOrMakeOut', e => e.textContent)).replace(/\s+/g, ' ').trim();
  console.log(JSON.stringify({ regionRow, regText, made: made.slice(0, 300), checked: checked.slice(0, 900), cropped: cropped.slice(0, 220), errors }, null, 1));
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
