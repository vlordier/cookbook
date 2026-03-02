// Step 6: procedural road generation
// Scrolling road segments with gentle S-curves.

export interface RoadSegment {
  curveOffset: number // lateral offset of road center at this segment
}

export function generateRoad(_segmentCount: number): RoadSegment[] {
  throw new Error('Not implemented yet - Step 6')
}

export function advanceRoad(
  _segments: RoadSegment[],
  _speed: number,
  _dt: number,
): RoadSegment[] {
  throw new Error('Not implemented yet - Step 6')
}
