// Marked roll tempo is tenths of a foot per minute.
export function playbackRatio(feetPerMinute, markedTempo) {
  return feetPerMinute / (markedTempo / 10);
}

// Scan rows are the shared clock for the transport, sound and photograph.
export function advance(row, elapsed, speed, secondsPerRow, safeRow) {
  return Math.min(safeRow, row + elapsed * speed / secondsPerRow);
}

export function playbackSeconds(rows, secondsPerRow, speed) {
  return rows * secondsPerRow / speed;
}

export function sounding(notes, row) {
  return notes.filter(([start, end, tone]) => tone > 0 && tone <= 127 && start <= row && end > row);
}

export class Piano {
  constructor() { this.voices = new Map(); }
  async unlock() {
    this.context ??= new AudioContext();
    if (!this.output) {
      this.output = this.context.createGain();
      this.output.gain.value = 0.35;
      const limiter = this.context.createDynamicsCompressor();
      this.output.connect(limiter).connect(this.context.destination);
    }
    await this.context.resume();
  }
  schedule(notes, row, secondsPerRow, speed, safeRow) {
    if (!this.context) return;
    const now = this.context.currentTime;
    const seconds = secondsPerRow / speed;
    const horizon = Math.min(safeRow, row + 0.12 / seconds);
    // Schedule on the audio clock ahead of rendering, so short notes between
    // animation frames still sound. Nothing is scheduled beyond decoded rows.
    for (const [start, end, tone] of notes) {
      if (tone <= 0 || tone > 127 || end <= row || start >= horizon) continue;
      const key = `${start}:${tone}`;
      let voice = this.voices.get(key);
      if (!voice) {
        const at = now + Math.max(0, start - row) * seconds;
        const gain = this.context.createGain();
        gain.gain.setValueAtTime(0, at);
        gain.gain.linearRampToValueAtTime(0.12, at + 0.008);
        gain.gain.exponentialRampToValueAtTime(0.025, at + 1.4);
        // Keep the attack/decay automation intact and release through a
        // separate gain. This also works where cancelAndHoldAtTime is absent.
        const release = this.context.createGain();
        release.gain.value = 1;
        gain.connect(release).connect(this.output);
        const oscillators = [1, 2, 3].map((harmonic, i) => {
          const osc = this.context.createOscillator();
          const partial = this.context.createGain();
          partial.gain.value = [0.65, 0.23, 0.08][i];
          osc.frequency.value = 440 * 2 ** ((tone - 69) / 12) * harmonic;
          osc.connect(partial).connect(gain);
          osc.onended = () => {
            osc.disconnect(); partial.disconnect();
            if (i === 0) {
              gain.disconnect(); release.disconnect();
              if (this.voices.get(key) === voice) this.voices.delete(key);
            }
          };
          osc.start(at);
          return osc;
        });
        voice = { gain, release, oscillators, at, released: false };
        this.voices.set(key, voice);
      }
      if (end <= horizon && !voice.released) {
        const at = Math.max(voice.at + 0.009, now + (end - row) * seconds);
        voice.release.gain.setTargetAtTime(0, at, 0.025);
        voice.oscillators.forEach(o => o.stop(at + 0.15));
        voice.released = true;
      }
    }
  }
  stop() {
    if (!this.context) return;
    const now = this.context.currentTime;
    for (const voice of this.voices.values()) {
      voice.release.gain.cancelScheduledValues(now);
      voice.release.gain.setTargetAtTime(0, now, 0.008);
      voice.oscillators.forEach(o => o.stop(now + 0.04));
    }
    this.voices.clear();
  }
}
