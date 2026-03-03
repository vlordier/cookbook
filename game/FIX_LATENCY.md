# Latency Fix Plan

## Why the spike is slow (>30 seconds)

### 1. High-resolution file uploads trigger multi-tile processing (primary cause)

`vl-processor.js` splits images into up to 10 tiles of 512x512 when the image is too large. "Too large" is defined as exceeding 256 image tokens (maxImageTokens = 256). A typical phone photo or 1080p screenshot easily exceeds this threshold.

What happens step by step:
- You upload a photo (e.g., 2000x1500 px from a phone camera)
- `isImageTooLarge()` returns true
- `calculateTileGrid()` picks e.g. 3x2 = 6 tiles
- `useThumbnail: true` adds a 7th thumbnail tile
- The `embedImages` ONNX session receives `pixel_values` with shape `[7, 1024, 768]` instead of `[1, 1024, 768]`
- The decoder prefill step processes 7 x 256 = 1792 image tokens per image, times 2 images = 3584 image tokens
- The attention computation in the prefill step is O(n^2) in sequence length

**Fix:** Pre-resize both images to 320x240 (or 256x256) before passing to `model.generate()`. At 320x240, `isImageTooLarge()` returns false and the single-tile path is taken, producing exactly 1 tile per image with ~64 active patches.

### 2. No WebGPU warmup run (secondary cause)

WebGPU must compile compute shaders the first time each ONNX operation runs. These compilations happen lazily, on first use, and are not persisted to disk (unlike model weights which land in the Cache API). The first real inference call triggers compilation of all kernels across all three ONNX sessions (embed_tokens, embed_images, decoder), which can cost 10-20 seconds on its own.

The reference project at `LiquidAI/LFM2.5-VL-1.6B-WebGPU` runs a dummy inference pass immediately after `load()` returns and before the UI shows "model ready". This pays the shader compilation cost once, at load time, so every subsequent call is fast.

**Fix:** After `model.load()` succeeds, run one dummy call with a tiny black image (64x64) and a trivial prompt. Log it as "Compiling GPU shaders..." rather than "Model ready" until it completes.

### 3. Prefill cost scales with image sequence length

The decoder ONNX session processes the entire prompt in a single pass on the first generation step (the "prefill"). With large images producing many tiles and many image tokens, this first step dominates total latency. Generating only 1 output token (`maxNewTokens: 1`) does not avoid the prefill cost.

This is not a separate root cause but a consequence of cause 1. Fixing the image size also fixes prefill cost.

### 4. Two full-resolution images are in flight simultaneously

The spike sends two images in one `generate()` call. If both are large, tile counts multiply: 7 tiles for image 1 plus 7 tiles for image 2 = 14 combined tiles feeding the image embedder, and 3584 image tokens in the decoder sequence. The single-tile path reduces this to 1 + 1 = 2 tiles and ~128 image tokens total.

---

## Action plan (based on the reference project)

### Step A: Resize images before inference (highest impact)

Before calling `model.generate()` in `main.js`, pass each image source through a canvas resize to 320x240:

```js
function resizeImageToDataURL(src, width = 320, height = 240) {
  return new Promise((resolve) => {
    const img = new Image()
    img.onload = () => {
      const canvas = document.createElement('canvas')
      canvas.width = width
      canvas.height = height
      canvas.getContext('2d').drawImage(img, 0, 0, width, height)
      resolve(canvas.toDataURL('image/jpeg', 0.85))
    }
    img.src = src
  })
}
```

Then before the `model.generate()` call:

```js
const [resized1, resized2] = await Promise.all([
  resizeImageToDataURL(img1.src),
  resizeImageToDataURL(img2.src),
])
const images = [resized1, resized2]
```

Expected result: single-tile path, ~64 active image patches per image, prefill sequence under 200 tokens total. Inference should drop from 30+ seconds to 1-3 seconds.

### Step B: Add a GPU warmup pass after model load

After `model.load()` completes, run one silent dummy inference and report a "warming up GPU" status to the user:

```js
async function warmup(model) {
  // Smallest valid image: 64x64 black canvas
  const canvas = document.createElement('canvas')
  canvas.width = 64
  canvas.height = 64
  const dataUrl = canvas.toDataURL('image/jpeg')
  const images = [dataUrl]
  const messageImageMap = new Map([[0, images]])
  await model.generate(
    [{ role: 'user', content: 'Describe.' }],
    { maxNewTokens: 1, images, messageImageMap }
  )
}
```

Call it right after `model.load()` returns, before enabling the "Run Inference" button. This pays the shader compilation cost once and makes the first real inference fast.

### Step C: Verify with the image cache

`VLModel` has an `imageCache` map keyed by image source URL. If you run inference twice with the same images (same object URL), the second call skips `embedImages` entirely. In the spike, use this to confirm that step A and B worked: after the first (warmed-up) call, click "Run Inference" again. The second call should complete in under 500ms.

For the game, webcam frames change every 300ms so the cache will not hit, but inference should already be fast after step A since frames are always small (320x240).

### Step D: Capture at 320x240 in the game (already in the plan)

`src/webcam/capture.ts` already plans to draw to a 320x240 OffscreenCanvas before encoding as JPEG. This naturally avoids the multi-tile problem in production. No change needed here; just confirm the canvas size is enforced before the frame is passed to `classify()`.

---

## Expected outcome after A + B

| Scenario | Before | After |
|---|---|---|
| First inference (file upload, full-res) | 30+ s | 2-4 s (warmup already done) |
| Subsequent inferences (same images) | ~30 s (no warmup hit) | <0.5 s (image cache hit) |
| Game frame (320x240 webcam) | N/A | 1-2 s (single tile, warmed up) |

---

## Reference

The working project (`LiquidAI/LFM2.5-VL-1.6B-WebGPU`) uses the same `vl-model.js` and `vl-processor.js` but drives inference from webcam frames, which are inherently small. That is the entire reason it runs fast: input resolution, not model architecture.
