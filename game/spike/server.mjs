/**
 * Minimal dev server for the inference spike.
 * Adds COOP/COEP headers required by WebGPU + SharedArrayBuffer.
 *
 * Usage:
 *   node spike/server.mjs
 *   Then open http://localhost:3000 in Chrome 113+ or Edge 113+
 */

import http from 'node:http'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const DIR = path.dirname(fileURLToPath(import.meta.url))
const PORT = 3000

const MIME = {
  '.html': 'text/html',
  '.js': 'text/javascript',
  '.mjs': 'text/javascript',
  '.css': 'text/css',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
}

http.createServer((req, res) => {
  const urlPath = req.url === '/' ? '/index.html' : req.url
  const filePath = path.join(DIR, urlPath)

  // Security headers required for WebGPU
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin')
  res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp')
  res.setHeader('Access-Control-Allow-Origin', '*')

  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(404)
      res.end('Not found')
      return
    }
    const ext = path.extname(filePath)
    res.setHeader('Content-Type', MIME[ext] ?? 'application/octet-stream')
    res.writeHead(200)
    res.end(data)
  })
}).listen(PORT, () => {
  console.log(`Spike server: http://localhost:${PORT}`)
  console.log('Requires Chrome 113+ or Edge 113+ (WebGPU)')
})
