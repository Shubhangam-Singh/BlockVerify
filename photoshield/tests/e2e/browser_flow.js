const { chromium } = require('playwright');
const path = require('path');
const VEC = path.join(__dirname, '..', 'vectors') + '/';
const JPEG = process.env.PS_JPEG || '/tmp/chelsea_q60.jpg';
(async () => {
  const user = 'pwuser' + Math.floor(Math.random() * 1e6);
  const reg = await (await fetch('http://localhost:5000/api/auth/register', { method: 'POST',
    headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ username: user, password: 'secret123' }) })).json();
  const browser = await chromium.launch();
  const ctx = await browser.newContext({ acceptDownloads: true });
  await ctx.addInitScript(([t, u]) => { localStorage.setItem('bv_token', t); localStorage.setItem('bv_user', u); }, [reg.token, user]);
  const page = await ctx.newPage();
  const errors = [];
  page.on('pageerror', e => errors.push('pageerror: ' + e.message));
  page.on('console', m => { if (m.type() === 'error') errors.push('console: ' + m.text()); });
  const answers = [];
  page.on('dialog', d => { const a = answers.shift(); a === undefined ? d.accept() : d.accept(a); });
  // In-app dialogs (bvConfirm / bvPrompt) replaced the browser's confirm()/prompt().
  const dialogOk = async (value) => {
    const ov = await page.waitForSelector('.bv-dialog-overlay', { timeout: 15000 });
    if (value !== undefined) await (await ov.$('#bvDlgInput')).fill(value);
    await (await ov.$('#bvDlgOk')).click();
    await page.waitForFunction(el => !el.isConnected, ov);   // this dialog closed (a chained one may already be open)
  };
  await page.goto('http://localhost:8080/#photoshield');
  await page.evaluate(() => switchTab('photoshield'));
  await page.waitForSelector('#psKeyInfo');
  await page.click('#psKeyGenBtn');
  await page.waitForFunction(() => document.getElementById('psKeyInfo').textContent.includes('Public key'));
  const warnVisible = await page.isVisible('#psKeyWarn');
  // provenance-only registration
  await page.setInputFiles('#psRegFiles', VEC + 'photo_chelsea.png');
  await page.waitForFunction(() => document.getElementById('psRegChecks').textContent.includes('pHash'));
  const phashShown = await page.$eval('#psRegChecks', e => e.textContent);
  const dl = page.waitForEvent('download');
  await page.click('#psRegBtn');
  await dialogOk();                                    // the key is not backed up yet: continue anyway
  await page.waitForFunction(() => /ANCHORED|failed|Error|error/.test(document.getElementById('psRegResult').textContent), null, { timeout: 120000 });
  const regText = await page.$eval('#psRegResult', e => e.textContent);
  const manifestDl = await dl;
  // check on ledger (local anchor -> SERVER-TRUSTED)
  await page.click('#psRegResult button:not(.bv-copy)');
  await page.waitForFunction(() => /SERVER-TRUSTED|TRUSTLESS|FAILED/.test(document.getElementById('psRegResult').textContent));
  const ledgerText = await page.$eval('#psRegResult .ps-steps', e => e.textContent);
  // verify exact + JPEG copy
  await page.setInputFiles('#psVerFile', VEC + 'photo_chelsea.png');
  await page.waitForFunction(() => document.getElementById('psVerResult').textContent.includes('EXACT') || document.getElementById('psVerResult').textContent.includes('UNKNOWN'));
  const ver1 = await page.$eval('#psVerResult', e => e.textContent);
  await page.setInputFiles('#psVerFile', JPEG);
  await page.waitForFunction(() => document.getElementById('psVerFN').textContent.includes('q60') && /UNKNOWN|DERIVED|SIMILAR|EXACT/.test(document.getElementById('psVerResult').textContent));
  await page.waitForTimeout(500);
  const ver2 = await page.$eval('#psVerResult', e => e.textContent);
  // export an encrypted backup
  const dl2 = page.waitForEvent('download');
  await page.click('text=Export backup');
  await dialogOk('a long passphrase 123');
  await dialogOk('a long passphrase 123');
  const keyDl = await dl2;
  await page.waitForFunction(() => document.getElementById('psKeyInfo').textContent.includes('exported'));
  const keyPath = require('os').tmpdir() + '/ps-key.json'; await keyDl.saveAs(keyPath);
  // my photos + prove
  await page.evaluate(() => psSection('mine'));
  await page.waitForFunction(() => document.getElementById('psMine').textContent.includes('Prove'));
  await page.click('#psMine button:has-text("Prove")');
  await page.waitForFunction(() => [...document.querySelectorAll('.toast')].some(t => /Proved|does not match/.test(t.textContent)), null, { timeout: 15000 });
  const toasts = await page.$$eval('.toast', els => els.map(e => e.textContent));
  await page.screenshot({ path: process.env.PS_SHOT || '/tmp/photoshield_tab.png', fullPage: false });
  // other tabs still fine
  for (const t of ['register', 'federated', 'ledger', 'help']) await page.evaluate(n => switchTab(n), t);
  const fs = require('fs');
  const backup = JSON.parse(fs.readFileSync(keyPath, 'utf8'));
  console.log(JSON.stringify({ warnVisible, phashShown, regText, manifestName: manifestDl.suggestedFilename(), ledgerText,
    ver1, ver2, keyBackupEncrypted: !!backup.encryptedPrivateKey && !backup.privateKeyPkcs8, toasts, errors }, null, 1));
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
