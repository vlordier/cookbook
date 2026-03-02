// Step 6: game engine
// requestAnimationFrame loop that owns game state and calls render each frame.

import type { Direction } from '../inference/steering'

export function startGame(
  _canvas: HTMLCanvasElement,
  _getDirection: () => Direction,
): () => void {
  throw new Error('Not implemented yet - Step 6')
}
