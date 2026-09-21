import { test } from 'node:test';
import assert from 'node:assert/strict';
import { advance, sounding, playbackRatio } from '../src/player.js';

test('transport obeys speed and never outruns digitization', () => {
  assert.equal(advance(100, 1, 2, 0.01, 500), 300);
  assert.equal(advance(100, 10, 2, 0.01, 500), 500);
  assert.equal(advance(500, 1, 1, 0.01, 500), 500);
  assert.equal(advance(500, 1, 1, 0.01, 900), 600);
});
test('control codes never become oscillator pitches; ends are exclusive', () => {
  const notes = [[0, 100, 60], [50, 150, 72], [0, 500, -1]];
  assert.deepEqual(sounding(notes, 75), notes.slice(0, 2));
  assert.deepEqual(sounding(notes, 100), [notes[1]]);
  assert.deepEqual(sounding(notes, 150), []);
});

test('audio schedules and stops without cancelAndHoldAtTime (Firefox)', async () => {
  const { Piano } = await import('../src/player.js');
  const oscillators = [];
  const gain = () => ({
    gain: { setValueAtTime() {}, linearRampToValueAtTime() {}, exponentialRampToValueAtTime() {}, cancelScheduledValues() {}, setTargetAtTime() {} },
    connect() { return this; }, disconnect() {},
  });
  const piano = new Piano();
  piano.context = {
    currentTime: 5,
    createGain: gain,
    createOscillator() {
      const osc = { frequency: {}, connect() { return this; }, disconnect() {}, start(at) { this.started = at; }, stop(at) { this.stopped = at; } };
      oscillators.push(osc); return osc;
    },
  };
  piano.output = gain();
  piano.schedule([[2, 3, 60], [15, 16, 72], [0, 20, -3]], 0, 0.001, 1, 10);
  assert.equal(oscillators.length, 3, 'only one audible note is buffered');
  assert.equal(oscillators[0].started, 5.002);
  assert(oscillators[0].stopped > 5.003);
  piano.schedule([[2, 3, 60]], 1, 0.001, 1, 10);
  assert.equal(oscillators.length, 3, 'successive frames do not retrigger scheduled notes');
  piano.stop();
  assert.equal(piano.voices.size, 0);
});

test('feet per minute maps to the marked MIDI tempo', () => {
  assert.equal(playbackRatio(6, 60), 1);
  assert.equal(playbackRatio(9, 60), 1.5);
  assert.equal(playbackRatio(2, 80), 0.25);
  assert.equal(playbackRatio(16, 80), 2);
});
