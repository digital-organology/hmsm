// Run against hmsm-web --static-dir web/dist. Optional argument: a full roll scan.
import { chromium, firefox } from '@playwright/test';
import assert from 'node:assert/strict';
import path from 'node:path';

const engine = process.env.HMSM_BROWSER === 'firefox' ? firefox : chromium;
const browser = await engine.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1360, height: 1120 } });
const errors = [];
page.on('pageerror', error => errors.push(error.message));
let liveNotes = false;
page.on('response', async response => {
  if (/\/api\/jobs\/[^/?]+\?version/.test(response.url())) {
    const state = await response.json().catch(() => ({}));
    if (state.status === 'processing' && state.notes?.length && state.safe_row > state.origin_row) liveNotes = true;
  }
});
try {
  await page.goto(process.env.HMSM_URL || 'http://127.0.0.1:8000');
  await page.waitForFunction(() => document.querySelector('#profile').options.length > 0);
  await page.screenshot({ path: '/tmp/hmsm-empty.png', fullPage: true });
  if (await page.locator('#close').isVisible()) {
    await page.locator('#close').click();
    await page.waitForFunction(() => !document.querySelector('#begin').disabled);
  }
  const scan = path.resolve(process.argv[2] || '../examples/phonola_roll_playback.tif');
  await page.locator('#file').setInputFiles(scan);
  await page.locator('#tempo').fill('80');
  await page.locator('#begin').click();
  await page.waitForFunction(() => !document.querySelector('#play').disabled, null, { timeout: 90000 });
  await page.locator('#play').click();
  await page.waitForFunction(() => document.querySelectorAll('.key.active').length > 0);
  await page.screenshot({ path: '/tmp/hmsm-playing.png', fullPage: true });
  assert.equal(await page.locator('#speedvalue').textContent(), '8.00 ft/min');
  await page.locator('#speed').focus();
  await page.keyboard.press('ArrowUp');
  assert.equal(await page.locator('#speedvalue').textContent(), '8.01 ft/min');
  const dial = await page.locator('#speed').boundingBox();
  await page.mouse.move(dial.x + dial.width / 2, dial.y + dial.height / 2);
  await page.mouse.down();
  await page.mouse.move(dial.x + dial.width / 2, dial.y + dial.height / 2 - 30);
  await page.mouse.up();
  assert(+(await page.locator('#speed').inputValue()) > 8.01, 'Dragging up turns tempo dial faster');
  await page.locator('#speed').fill('9');
  assert.equal(await page.locator('#speedvalue').textContent(), '9.00 ft/min');
  await page.locator('#play').click();
  const before = await page.locator('#seek').inputValue();
  await page.waitForTimeout(300);
  assert.equal(await page.locator('#seek').inputValue(), before, 'Pause holds scan position');
  await page.locator('#restart').click();
  await page.waitForFunction(() => !document.querySelector('#download').hidden, null, { timeout: 90000 });
  const midi = await page.request.get(new URL(await page.locator('#download').getAttribute('href'), page.url()).href);
  assert.equal((await midi.body()).subarray(0, 4).toString(), 'MThd');
  await page.reload();
  await page.waitForFunction(() => !document.querySelector('#download').hidden);
  assert.equal(await page.locator('#filename').textContent(), path.basename(scan));
  assert.equal(await page.locator('#speedvalue').textContent(), '8.00 ft/min', 'Restored tempo matches MIDI');
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({ path: '/tmp/hmsm-mobile.png', fullPage: true });
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'Mobile must not overflow');
  const chooserPromise = page.waitForEvent('filechooser');
  await page.locator('#different').click();
  const chooser = await chooserPromise;
  await chooser.setFiles(path.resolve('../examples/phonola_roll_playback.tif'));
  await page.waitForFunction(() => !document.querySelector('#begin').disabled);
  assert.equal(await page.locator('#filename').textContent(), 'phonola_roll_playback.tif');
  await page.locator('#begin').click();
  await page.waitForFunction(() => !document.querySelector('#play').disabled);
  await page.locator('#play').click();
  await page.waitForFunction(() => document.querySelectorAll('.key.active').length > 0);
  await page.locator('#close').click();
  await page.waitForFunction(() => !document.querySelector('#begin').disabled);
  assert.deepEqual(errors, []);
  if (process.argv[2]) assert(liveNotes, 'Full roll should become playable while still processing');
  console.log(JSON.stringify({ browserErrors: errors, liveNotes, scan, midiBytes: (await midi.body()).length }));
} finally { await browser.close(); }
