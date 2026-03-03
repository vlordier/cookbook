# Learnings

## 1. Pre-resize images to 320x240 before inference

**Context:** Running LFM2.5-VL-1.6B-ONNX in the browser via WebGPU.

**Problem:** Inference was taking 30+ seconds on full-resolution images (phone photos, screenshots).

**Cause:** `vl-processor.js` splits images into up to 10 tiles of 512x512 when the image exceeds the `maxImageTokens: 256` threshold. A typical 1080p or phone-camera image generates 7-10 tiles per image. With 2 images in one call, the decoder prefill sequence grows to ~3584 image tokens, making the first generation step extremely slow.

**Fix:** Resize both images to 320x240 (or any size that stays under the token threshold) before passing them to `model.generate()`. At 320x240 the processor takes the single-tile path, producing ~64 active image patches per image.

**Result:** Latency dropped from 30+ seconds to under 1 second.

**Takeaway:** For any VLM with a tiled image processor, input resolution is the primary latency lever. Always resize to the minimum resolution that preserves enough visual detail for your task before running inference.
