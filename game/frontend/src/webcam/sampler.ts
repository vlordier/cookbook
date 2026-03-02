// Step 4: frame pair sampler
// Keeps a rolling [prev, curr] frame pair and calls onPair on each interval.

export interface FramePair {
  prev: string // base64 JPEG data URL
  curr: string // base64 JPEG data URL
}

export function startSampler(
  _videoEl: HTMLVideoElement,
  _onPair: (pair: FramePair) => void,
  _intervalMs = 300,
): () => void {
  throw new Error('Not implemented yet - Step 4')
}
