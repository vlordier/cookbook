// Step 3: model loader
// Singleton pipeline for LFM2.5-VL-1.6B-ONNX with WebGPU backend.

import type { ImageToTextPipeline } from '@huggingface/transformers'

export type ModelStatus =
  | { state: 'idle' }
  | { state: 'loading'; progress: number }
  | { state: 'ready' }
  | { state: 'error'; message: string }

export async function loadModel(
  _onStatus: (s: ModelStatus) => void,
): Promise<void> {
  throw new Error('Not implemented yet - Step 3')
}

export function getModel(): ImageToTextPipeline {
  throw new Error('Not implemented yet - Step 3')
}
