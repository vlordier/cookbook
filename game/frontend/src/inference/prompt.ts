// Step 3: multimodal message builder
// Constructs the message array passed to the VL pipeline.

export interface Message {
  role: 'user'
  content: Array<
    | { type: 'image'; image: string }
    | { type: 'text'; text: string }
  >
}

export function buildMessages(_framePrev: string, _frameCurr: string): Message[] {
  throw new Error('Not implemented yet - Step 3')
}
