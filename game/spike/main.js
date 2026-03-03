import { HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision'

// MediaPipe CDN paths (WASM must be loaded from CDN, not bundled)
const WASM_PATH = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'

const EMA_ALPHA = 0.2

// Hand skeleton connections (MediaPipe 21-landmark layout)
const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],          // thumb
  [0,5],[5,6],[6,7],[7,8],          // index
  [0,9],[9,10],[10,11],[11,12],     // middle
  [0,13],[13,14],[14,15],[15,16],   // ring
  [0,17],[17,18],[18,19],[19,20],   // pinky
  [5,9],[9,13],[13,17],             // palm
]

const videoEl   = document.getElementById('video')
const overlayEl = document.getElementById('overlay')
const wheelCanvas = document.getElementById('wheel')
const angleEl   = document.getElementById('angle')
const statusEl  = document.getElementById('status')
const btnStart  = document.getElementById('btnStart')

let handLandmarker = null
let animFrameId    = null
let smoothedAngle  = 0

function setStatus(msg) { statusEl.textContent = msg }

// ── Wheel visualisation ──────────────────────────────────────────────────────

function drawWheel(canvas, angleDeg) {
  const ctx = canvas.getContext('2d')
  const cx = canvas.width / 2
  const cy = canvas.height / 2
  const r  = Math.min(canvas.width, canvas.height) / 2 - 10
  const rad = angleDeg * Math.PI / 180

  ctx.clearRect(0, 0, canvas.width, canvas.height)

  // Rim
  ctx.beginPath()
  ctx.arc(cx, cy, r, 0, Math.PI * 2)
  ctx.strokeStyle = '#555'
  ctx.lineWidth = 6
  ctx.stroke()

  // 3 spokes rotated by current angle
  for (const offset of [0, 120, 240]) {
    const a = (offset + angleDeg) * Math.PI / 180
    ctx.beginPath()
    ctx.moveTo(cx, cy)
    ctx.lineTo(cx + r * Math.sin(a), cy - r * Math.cos(a))
    ctx.strokeStyle = '#888'
    ctx.lineWidth = 3
    ctx.stroke()
  }

  // Hub
  ctx.beginPath()
  ctx.arc(cx, cy, 8, 0, Math.PI * 2)
  ctx.fillStyle = '#aaa'
  ctx.fill()

  // 12-o'clock indicator dot
  ctx.beginPath()
  ctx.arc(cx + r * Math.sin(rad), cy - r * Math.cos(rad), 6, 0, Math.PI * 2)
  ctx.fillStyle = '#4af'
  ctx.fill()
}

// ── Hand skeleton overlay ────────────────────────────────────────────────────

function drawHandSkeleton(ctx, landmarks, W, H) {
  ctx.strokeStyle = 'rgba(255,255,255,0.35)'
  ctx.lineWidth = 1
  for (const [a, b] of HAND_CONNECTIONS) {
    ctx.beginPath()
    ctx.moveTo(landmarks[a].x * W, landmarks[a].y * H)
    ctx.lineTo(landmarks[b].x * W, landmarks[b].y * H)
    ctx.stroke()
  }
  // Wrist dot
  ctx.beginPath()
  ctx.arc(landmarks[0].x * W, landmarks[0].y * H, 5, 0, Math.PI * 2)
  ctx.fillStyle = '#ff6'
  ctx.fill()
}

// ── Detection loop ───────────────────────────────────────────────────────────

function loop() {
  animFrameId = requestAnimationFrame(loop)

  const W = overlayEl.width
  const H = overlayEl.height
  const ctx = overlayEl.getContext('2d')
  ctx.clearRect(0, 0, W, H)

  const results = handLandmarker.detectForVideo(videoEl, performance.now())

  if (results.landmarks.length === 2) {
    for (const hand of results.landmarks) drawHandSkeleton(ctx, hand, W, H)

    // Sort hands by x so index-0 is left side, index-1 is right side of frame
    const [left, right] = results.landmarks.slice().sort((a, b) => a[0].x - b[0].x)
    const lw = left[0]   // left wrist
    const rw = right[0]  // right wrist

    // Steering axis line
    ctx.beginPath()
    ctx.moveTo(lw.x * W, lw.y * H)
    ctx.lineTo(rw.x * W, rw.y * H)
    ctx.strokeStyle = '#4af'
    ctx.lineWidth = 2
    ctx.stroke()

    // Angle: 0 = horizontal (straight), positive = right turn, negative = left turn
    const raw = Math.atan2(rw.y - lw.y, rw.x - lw.x) * 180 / Math.PI
    const clamped = Math.max(-90, Math.min(90, raw))
    smoothedAngle = EMA_ALPHA * clamped + (1 - EMA_ALPHA) * smoothedAngle

    const display = Math.round(smoothedAngle)
    angleEl.textContent = `${display > 0 ? '+' : ''}${display}°`
    drawWheel(wheelCanvas, smoothedAngle)
    setStatus('Detecting hands...')
  } else {
    // Decay toward 0 when hands are lost
    smoothedAngle = smoothedAngle * (1 - EMA_ALPHA)
    const display = Math.round(smoothedAngle)
    angleEl.textContent = `${display > 0 ? '+' : ''}${display}°`
    drawWheel(wheelCanvas, smoothedAngle)
    setStatus(results.landmarks.length === 0 ? 'No hands detected — show both hands' : 'Show both hands')
  }
}

// ── Start / Stop ─────────────────────────────────────────────────────────────

async function start() {
  btnStart.disabled = true
  setStatus('Loading MediaPipe Hand Landmarker...')

  if (!handLandmarker) {
    const vision = await FilesetResolver.forVisionTasks(WASM_PATH)
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' },
      runningMode: 'VIDEO',
      numHands: 2,
    })
  }

  setStatus('Starting webcam...')
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480, facingMode: 'user' },
  })
  videoEl.srcObject = stream
  await new Promise(resolve => { videoEl.onloadedmetadata = resolve })
  videoEl.play()

  overlayEl.width  = videoEl.videoWidth
  overlayEl.height = videoEl.videoHeight

  drawWheel(wheelCanvas, 0)
  btnStart.textContent = 'Stop'
  btnStart.disabled = false
  btnStart.onclick = stop

  loop()
}

function stop() {
  if (animFrameId) { cancelAnimationFrame(animFrameId); animFrameId = null }
  videoEl.srcObject?.getTracks().forEach(t => t.stop())
  videoEl.srcObject = null
  overlayEl.getContext('2d').clearRect(0, 0, overlayEl.width, overlayEl.height)
  smoothedAngle = 0
  angleEl.textContent = ''
  drawWheel(wheelCanvas, 0)
  setStatus('Stopped.')
  btnStart.textContent = 'Start Webcam'
  btnStart.onclick = start
}

// ── Init ─────────────────────────────────────────────────────────────────────

btnStart.onclick = start
drawWheel(wheelCanvas, 0)
