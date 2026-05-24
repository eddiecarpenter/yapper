/**
 * Audio worklet source for `MicrophoneCapture` — decimates a Web Audio
 * input stream (typically 48 kHz) to 16 kHz frames suitable for Silero
 * VAD / Whisper STT downstream.
 *
 * Canonical specification: `docs/ARCHITECTURE.md` §6.1 + Design Plan KD-2.
 *
 * ## Shape — string, not module
 *
 * The worklet runs in `AudioWorkletGlobalScope` — a separate global
 * environment that does NOT share module loaders with the main thread,
 * cannot `import`, and cannot reference any of the project's types.
 * The canonical way to ship a worklet from a library that does not
 * yet have a bundler set up (Yapper is still tsc-only) is to keep the
 * source as a JavaScript string here and load it at runtime via a
 * Blob URL:
 *
 *   ```ts
 *   const url = URL.createObjectURL(
 *     new Blob([DECIMATOR_WORKLET_SOURCE], { type: "application/javascript" }),
 *   );
 *   await audioContext.audioWorklet.addModule(url);
 *   ```
 *
 * This is bundler-agnostic — when a future Vite (or other) setup
 * introduces `?url`-style worklet imports, swap the Blob construction
 * for the import and the rest of `MicrophoneCapture` stays untouched.
 *
 * ## DSP — polyphase FIR anti-aliased decimation
 *
 * The integer 48 kHz → 16 kHz path applies a 33-tap Hamming-windowed
 * sinc low-pass filter with cutoff at half the new Nyquist (≈ 8 kHz of
 * source bandwidth, normalised to source rate as `0.5 / decimationFactor`
 * = 1/6 ≈ 0.1667) before keeping every Nth sample. The FIR is built
 * once at processor construction inside the worklet — the resulting
 * `Float32Array` then serves as the "embedded coefficients constant"
 * the Design Plan KD-2 calls for, with the benefit that the coefficient
 * design is auditable in source rather than a magic-number blob.
 *
 * For non-integer ratios (device rate ≠ 48 kHz — typical on Bluetooth
 * headsets that surface 44.1 kHz, or browsers configured at custom
 * rates), the worklet falls back to a generic rational resampler that
 * tracks a fractional accumulator and applies the same anti-aliasing
 * FIR sized to the actual ratio. Output frames are still 512 samples
 * at exactly 16 kHz.
 */

/**
 * Processor name registered inside the worklet via
 * `registerProcessor(...)`. Re-exported as a named constant so the
 * main-thread code that constructs `AudioWorkletNode` references the
 * same string — no risk of a silent typo breaking the load.
 */
export const DECIMATOR_WORKLET_NAME = "yapper-decimator";

/**
 * Inline JavaScript source for the decimator worklet. Kept as a
 * template-literal string for the bundler-free reasons documented at
 * the top of the file. Newlines and whitespace are preserved so the
 * source is human-readable in DevTools when the Blob URL is opened.
 *
 * Exported for two reasons:
 *   1. Production: `MicrophoneCapture.start()` wraps it in a Blob and
 *      passes the resulting URL to `audioContext.audioWorklet.addModule`.
 *   2. Tests: the test suite asserts that the constant contains the
 *      `registerProcessor` call so a future refactor that moves the
 *      registration out of the string fails loudly.
 */
export const DECIMATOR_WORKLET_SOURCE = `
// Built once at processor construction; result is the "embedded
// coefficients" Float32Array the Design Plan KD-2 calls for. The
// helper is a Hamming-windowed sinc, DC-normalised so the filter has
// unity gain at 0 Hz (i.e. the decimated audio retains the same
// average level as the source).
function buildLowPassFIR(numTaps, cutoff) {
  const coeffs = new Float32Array(numTaps);
  const center = (numTaps - 1) / 2;
  let sum = 0;
  for (let n = 0; n < numTaps; n++) {
    const m = n - center;
    let sinc;
    if (m === 0) {
      sinc = 2 * cutoff;
    } else {
      sinc = Math.sin(2 * Math.PI * cutoff * m) / (Math.PI * m);
    }
    const window = 0.54 - 0.46 * Math.cos((2 * Math.PI * n) / (numTaps - 1));
    coeffs[n] = sinc * window;
    sum += coeffs[n];
  }
  for (let n = 0; n < numTaps; n++) {
    coeffs[n] /= sum;
  }
  return coeffs;
}

class YapperDecimator extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = (options && options.processorOptions) || {};
    // \`sampleRate\` is a worklet global supplied by the runtime; falling
    // back to it covers the (rare) case where the host omits the
    // processorOptions.
    this.deviceSampleRate = opts.deviceSampleRate || sampleRate;
    this.targetSampleRate = opts.targetSampleRate || 16000;
    this.frameSampleCount = opts.frameSampleCount || 512;

    const exactRatio = this.deviceSampleRate / this.targetSampleRate;
    const intRatio = Math.round(exactRatio);
    // Treat as integer decimation if the device rate is within 1e-6 of
    // an integer multiple of the target. 1e-6 is well below any device
    // tolerance — anything else is genuine fractional resampling.
    this.useInteger = Math.abs(exactRatio - intRatio) < 1e-6 && intRatio > 0;

    if (this.useInteger) {
      this.decimationFactor = intRatio;
      // Normalised cutoff = 0.5 / decimationFactor. At factor 3 this is
      // 1/6 ≈ 0.1667 — i.e. the filter passes everything up to the new
      // Nyquist (8 kHz at the 16 kHz output rate) and aggressively
      // attenuates the band the decimation would otherwise alias.
      this.coefficients = buildLowPassFIR(33, 0.5 / this.decimationFactor);
      this.inputCounter = 0;
    } else {
      // Generic rational resampler. The fractional accumulator advances
      // by 1 per input sample; whenever it reaches the source-per-target
      // ratio, we emit one output sample and roll the accumulator back
      // by that ratio (preserving sub-sample phase so we don't drift).
      this.fractionalRatio = exactRatio;
      this.fractionalIndex = 0;
      this.coefficients = buildLowPassFIR(33, 0.5 / Math.max(1, this.fractionalRatio));
    }

    this.delayLine = new Float32Array(this.coefficients.length);
    this.delayWriteIndex = 0;
    this.outputBuffer = new Float32Array(this.frameSampleCount);
    this.outputIndex = 0;
  }

  // Convolve the FIR taps against the current delay-line state. Reads
  // the delay line backwards from the most recently written sample so
  // the most recent input is multiplied by the first coefficient.
  convolve() {
    let sum = 0;
    const L = this.coefficients.length;
    // delayWriteIndex points at the NEXT slot to write — the last write
    // landed at delayWriteIndex - 1.
    const start = (this.delayWriteIndex - 1 + L) % L;
    for (let k = 0; k < L; k++) {
      const idx = (start - k + L) % L;
      sum += this.delayLine[idx] * this.coefficients[k];
    }
    return sum;
  }

  // \`process\` is called by the Web Audio runtime once per 128-sample
  // render quantum on each input channel. We only consume the first
  // channel (mono mic) — multi-channel mics will get their other
  // channels dropped, which is the intended behaviour for VAD/STT
  // input.
  process(inputs) {
    const input = inputs[0];
    const channel = input && input[0];
    if (!channel) {
      // No input this quantum — keep the worklet alive for the next call.
      return true;
    }

    for (let i = 0; i < channel.length; i++) {
      this.delayLine[this.delayWriteIndex] = channel[i];
      this.delayWriteIndex = (this.delayWriteIndex + 1) % this.delayLine.length;

      if (this.useInteger) {
        this.inputCounter++;
        if (this.inputCounter >= this.decimationFactor) {
          this.inputCounter = 0;
          this.outputBuffer[this.outputIndex++] = this.convolve();
          if (this.outputIndex >= this.frameSampleCount) {
            // Slice produces a non-aliased copy — required because the
            // worklet's port serialises the buffer and we are about to
            // overwrite outputBuffer for the next frame.
            this.port.postMessage({
              type: "frame",
              frame: this.outputBuffer.slice(0),
            });
            this.outputIndex = 0;
          }
        }
      } else {
        this.fractionalIndex += 1;
        if (this.fractionalIndex >= this.fractionalRatio) {
          this.fractionalIndex -= this.fractionalRatio;
          this.outputBuffer[this.outputIndex++] = this.convolve();
          if (this.outputIndex >= this.frameSampleCount) {
            this.port.postMessage({
              type: "frame",
              frame: this.outputBuffer.slice(0),
            });
            this.outputIndex = 0;
          }
        }
      }
    }

    return true;
  }
}

registerProcessor(${JSON.stringify(DECIMATOR_WORKLET_NAME)}, YapperDecimator);
`;
