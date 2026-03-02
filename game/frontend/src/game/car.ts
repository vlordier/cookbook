// Step 6: car state and update
// Lateral position, speed, and steering.

export interface CarState {
  x: number      // lateral position: 0 = left edge, 1 = right edge
  speed: number  // forward speed in pixels/s (constant for MVP)
  steer: number  // current steering: -1 left, 0 straight, +1 right
}

export function updateCar(_car: CarState, _dt: number): CarState {
  throw new Error('Not implemented yet - Step 6')
}
