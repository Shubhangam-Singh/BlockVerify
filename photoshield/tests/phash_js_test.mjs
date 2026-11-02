// Runs the SHIPPED browser pHash (extracted from frontend/index.html by test_hashing.py)
// over the decoded pixels of every PNG vector and prints the hashes as JSON.
//
//   node photoshield/tests/phash_js_test.mjs <js-source-file> <pixels.json>
//
// pixels.json: [{"name", "w", "h", "rgba": base64}]. PNG decoding is lossless, so the pixels
// PIL decodes are the pixels the browser decodes; what is under test is the hash itself.
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const [, , srcFile, pixFile] = process.argv;
const ctx = vm.createContext({});
vm.runInContext(readFileSync(srcFile, 'utf8') +
  '\nthis.psPhashFromRGBA = psPhashFromRGBA; this.psHamming = psHamming;', ctx);
const items = JSON.parse(readFileSync(pixFile, 'utf8'));
const out = {};
for (const it of items) {
  const rgba = new Uint8Array(Buffer.from(it.rgba, 'base64'));
  try { out[it.name] = ctx.psPhashFromRGBA(it.w, it.h, rgba); }
  catch (e) { out[it.name] = 'ERROR: ' + e.message; }
}
process.stdout.write(JSON.stringify(out));
