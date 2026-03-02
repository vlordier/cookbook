// Step 3: steering classifier
// Thin wrapper: runs the VL model on a frame pair and returns a Direction.

export type Direction = 'left' | 'straight' | 'right'

export async function classify(
  _framePrev: string,
  _frameCurr: string,
): Promise<Direction> {
  throw new Error('Not implemented yet - Step 3')
}
