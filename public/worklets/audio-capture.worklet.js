// Runs on the audio rendering thread; batches input into fixed-size frames.
class AudioCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    // 4096 frames per message keeps message traffic low.
    this.buffer = new Float32Array(4096);
    this.offset = 0;
  }

  process(inputs) {
    const channel = inputs[0]?.[0];
    if (!channel) return true;

    let inputOffset = 0;
    while (inputOffset < channel.length) {
      const count = Math.min(channel.length - inputOffset, this.buffer.length - this.offset);
      this.buffer.set(channel.subarray(inputOffset, inputOffset + count), this.offset);
      this.offset += count;
      inputOffset += count;

      if (this.offset === this.buffer.length) {
        // Transfers the buffer, which detaches it; a fresh one is allocated below.
        this.port.postMessage(this.buffer, [this.buffer.buffer]);
        this.buffer = new Float32Array(4096);
        this.offset = 0;
      }
    }

    return true;
  }
}

registerProcessor("audio-capture", AudioCaptureProcessor);