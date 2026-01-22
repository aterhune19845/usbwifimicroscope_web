#!/usr/bin/env python3
# Web-based viewer for WiFi/USB microscope
# Captures images and serves them via HTTP with MJPEG streaming
# ALL IMAGE PROCESSING DONE IN PYTHON - browser displays raw processed images
# Supports both WiFi (UDP) and USB (webcam) modes

import time
import socket
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import sys
import io
import os
import warnings
from datetime import datetime

# Try to import PIL for better JPEG encoding
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Try to import OpenCV for USB camera support
try:
    import cv2
    import numpy as np
    USB_AVAILABLE = True
except ImportError:
    USB_AVAILABLE = False
    print("OpenCV not available. Install with: pip3 install opencv-python")

HOST = "192.168.29.1"  # Microscope hard-wired IP address
SPORT = 20000          # Microscope command port
RPORT = 10900          # Receive port for JPEG frames
WEB_PORT = 8080        # Web server port

current_frame = None
current_frame_number = 0  # Increments with each new frame
frame_lock = threading.Lock()
frame_event = threading.Event()
running = True
target_fps = 29  # Default target FPS for streaming
connection_mode = None  # Will be 'wifi', 'usb', or None

# Image processing settings (all processing happens in Python)
processing_settings = {
    'brightness': 0,     # -100 to 100 (additive)
    'contrast': 1.0,     # 0.1 to 3.0 (multiplicative)
    'saturation': 1.0,   # 0.0 to 3.0
    'flip_h': True,      # Default flipped horizontal
    'flip_v': True,      # Default flipped vertical
    'rotate': 0,         # 0, 90, 180, 270
    'zoom': 1.0          # 0.5 to 4.0
}
processing_lock = threading.Lock()
usb_camera_cap = None  # Global reference to camera

# Capture settings
capture_fps = 15  # Capture FPS (independent of stream FPS)
jpeg_quality = 75  # JPEG encoding quality (1-100)
capture_settings_lock = threading.Lock()

# Context manager to suppress libjpeg warnings
class SuppressStderr:
    def __enter__(self):
        self.null_fd = os.open(os.devnull, os.O_RDWR)
        self.save_fd = os.dup(2)
        os.dup2(self.null_fd, 2)

    def __exit__(self, *_):
        os.dup2(self.save_fd, 2)
        os.close(self.null_fd)
        os.close(self.save_fd)

def apply_image_processing(frame):
    """Apply all image processing in Python using OpenCV"""
    global processing_settings

    with processing_lock:
        settings = processing_settings.copy()

    # Debug: Show what settings we're applying
    # print(f"[APPLY] Settings: B={settings['brightness']} C={settings['contrast']:.1f} S={settings['saturation']:.1f}")

    # Convert to float for processing
    processed = frame.astype(np.float32)

    # 1. Apply brightness and contrast
    # Formula: output = alpha * input + beta
    # alpha = contrast, beta = brightness
    if settings['brightness'] != 0 or settings['contrast'] != 1.0:
        processed = cv2.convertScaleAbs(
            processed,
            alpha=settings['contrast'],
            beta=settings['brightness']
        )
        processed = processed.astype(np.float32)

    # 2. Apply saturation
    if settings['saturation'] != 1.0:
        # Convert to HSV to adjust saturation
        hsv = cv2.cvtColor(processed.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * settings['saturation'], 0, 255)
        processed = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    # Convert back to uint8 for geometric transforms
    processed = np.clip(processed, 0, 255).astype(np.uint8)

    # 3. Apply flip
    if settings['flip_h'] and settings['flip_v']:
        processed = cv2.flip(processed, -1)  # Both axes
    elif settings['flip_h']:
        processed = cv2.flip(processed, 1)   # Horizontal
    elif settings['flip_v']:
        processed = cv2.flip(processed, 0)   # Vertical

    # 4. Apply rotation
    rotation = int(settings['rotate']) % 360
    if rotation == 90:
        processed = cv2.rotate(processed, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        processed = cv2.rotate(processed, cv2.ROTATE_180)
    elif rotation == 270:
        processed = cv2.rotate(processed, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # 5. Apply zoom (center crop and resize)
    zoom = settings['zoom']
    if zoom != 1.0:
        h, w = processed.shape[:2]

        if zoom > 1.0:
            # Zoom in - crop from center and resize back
            crop_h = int(h / zoom)
            crop_w = int(w / zoom)
            start_y = (h - crop_h) // 2
            start_x = (w - crop_w) // 2
            cropped = processed[start_y:start_y+crop_h, start_x:start_x+crop_w]
            processed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            # Zoom out - resize smaller and pad with black
            new_h = int(h * zoom)
            new_w = int(w * zoom)
            resized = cv2.resize(processed, (new_w, new_h), interpolation=cv2.INTER_AREA)
            processed = np.zeros((h, w, 3), dtype=np.uint8)
            start_y = (h - new_h) // 2
            start_x = (w - new_w) // 2
            processed[start_y:start_y+new_h, start_x:start_x+new_w] = resized

    return processed

class MicroscopeHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            # Serve the HTML viewer
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=5.0, user-scalable=yes">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <title>Microscope Viewer</title>
    <style>
        * {
            box-sizing: border-box;
        }
        body {
            margin: 0;
            padding: 0;
            background: #1a1a1a;
            color: #fff;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            -webkit-tap-highlight-color: transparent;
            -webkit-touch-callout: none;
            touch-action: manipulation;
        }
        .container {
            display: flex;
            height: 100vh;
            height: 100dvh;
        }
        .sidebar {
            width: 280px;
            background: #252525;
            padding: 20px;
            overflow-y: auto;
            -webkit-overflow-scrolling: touch;
            border-right: 1px solid #333;
            transition: transform 0.3s ease;
        }
        .sidebar.mobile-hidden {
            transform: translateX(-100%);
        }
        .mobile-toggle {
            display: none;
            position: fixed;
            top: 10px;
            left: 10px;
            z-index: 1000;
            background: #252525;
            border: 1px solid #444;
            color: #fff;
            padding: 12px 16px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            -webkit-user-select: none;
            user-select: none;
        }
        .mobile-toggle:active {
            background: #1a1a1a;
        }
        .mobile-overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.5);
            z-index: 998;
            opacity: 0;
            transition: opacity 0.3s ease;
        }
        .mobile-overlay.active {
            opacity: 1;
        }
        .main-content {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 20px;
            overflow: hidden;
        }
        h1 {
            margin: 0 0 30px 0;
            font-size: 20px;
            font-weight: 600;
            color: #fff;
        }
        .control-group {
            margin-bottom: 25px;
        }
        .control-group h3 {
            margin: 0 0 12px 0;
            font-size: 13px;
            font-weight: 600;
            color: #999;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .button-group {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
        }
        button {
            background: #333;
            border: 1px solid #444;
            color: #fff;
            padding: 10px 14px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 13px;
            transition: all 0.2s;
        }
        button:hover {
            background: #3a3a3a;
            border-color: #555;
        }
        button:active {
            background: #2a2a2a;
        }
        button.primary {
            background: #0066cc;
            border-color: #0066cc;
            grid-column: 1 / -1;
        }
        button.primary:hover {
            background: #0077ee;
        }
        .slider-control {
            margin-bottom: 15px;
        }
        .slider-control label {
            display: block;
            margin-bottom: 6px;
            font-size: 13px;
            color: #ccc;
        }
        .slider-control input[type="range"] {
            width: 100%;
            height: 6px;
            border-radius: 3px;
            background: #333;
            outline: none;
            -webkit-appearance: none;
        }
        .slider-control input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: #0066cc;
            cursor: pointer;
        }
        .slider-control input[type="range"]::-moz-range-thumb {
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: #0066cc;
            cursor: pointer;
            border: none;
        }
        .slider-value {
            display: inline-block;
            margin-left: 8px;
            color: #0066cc;
            font-weight: 600;
        }
        select {
            width: 100%;
            background: #333;
            border: 1px solid #444;
            color: #fff;
            padding: 10px;
            border-radius: 6px;
            font-size: 13px;
            cursor: pointer;
        }
        select:hover {
            border-color: #555;
        }
        .image-container {
            position: relative;
            width: calc(100vw - 280px - 40px);
            height: calc(100vh - 40px);
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }
        .image-wrapper {
            position: relative;
            max-width: 100%;
            max-height: 100%;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        #microscope-image {
            max-width: 100%;
            max-height: 100%;
            width: auto;
            height: auto;
            border: 2px solid #333;
            background: #000;
            display: block;
            image-rendering: auto;
        }
        .info {
            position: absolute;
            bottom: 30px;
            right: 30px;
            background: rgba(0, 0, 0, 0.7);
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 12px;
            color: #aaa;
        }
        .fps {
            color: #0066cc;
            font-weight: 600;
        }

        @media (min-width: 1025px) {
            .sidebar.mobile-hidden {
                transform: none;
            }
            .mobile-overlay {
                display: none !important;
            }
        }

        @media (max-width: 1024px) {
            .container {
                flex-direction: column;
            }

            .sidebar {
                position: fixed;
                top: 0;
                left: 0;
                width: 280px;
                height: 100vh;
                height: 100dvh;
                z-index: 999;
                box-shadow: 2px 0 10px rgba(0,0,0,0.5);
            }

            .mobile-toggle {
                display: block;
            }

            .main-content {
                width: 100%;
                padding: 10px;
                padding-top: 50px;
            }

            .image-container {
                width: 100%;
                height: calc(100vh - 60px);
                height: calc(100dvh - 60px);
            }

            .info {
                bottom: 10px;
                right: 10px;
                font-size: 10px;
                padding: 6px 10px;
            }
        }

        @media (max-width: 768px) {
            .sidebar {
                width: 100%;
                max-width: 320px;
            }

            h1 {
                font-size: 18px;
                margin-bottom: 20px;
            }

            .control-group h3 {
                font-size: 12px;
            }

            button {
                padding: 12px 14px;
                font-size: 14px;
                -webkit-tap-highlight-color: transparent;
            }

            .slider-control label {
                font-size: 12px;
            }
        }

        @media (hover: none) and (pointer: coarse) {
            button {
                min-height: 44px;
            }

            input[type="range"] {
                height: 44px;
                padding: 10px 0;
            }
        }
    </style>
</head>
<body>
    <button class="mobile-toggle" id="mobile-toggle" onclick="toggleSidebar()">☰ Controls</button>
    <div class="mobile-overlay" id="mobile-overlay" onclick="closeSidebar()"></div>
    <div class="container">
        <div class="sidebar" id="sidebar">
            <h1>Microscope Controls</h1>

            <div class="control-group">
                <h3>Image Processing</h3>
                <div class="slider-control">
                    <label>Brightness: <span class="slider-value" id="brightness-value">0</span></label>
                    <input type="range" id="brightness-slider" min="-100" max="100" value="0" step="1">
                </div>
                <div class="slider-control">
                    <label>Contrast: <span class="slider-value" id="contrast-value">1.0</span></label>
                    <input type="range" id="contrast-slider" min="10" max="300" value="100" step="1">
                </div>
                <div class="slider-control">
                    <label>Saturation: <span class="slider-value" id="saturation-value">1.0</span></label>
                    <input type="range" id="saturation-slider" min="0" max="300" value="100" step="1">
                </div>
            </div>

            <div class="control-group">
                <h3>Orientation</h3>
                <div class="button-group">
                    <button onclick="flipHorizontal()">Flip H</button>
                    <button onclick="flipVertical()">Flip V</button>
                    <button onclick="rotateLeft()">↶ 90°</button>
                    <button onclick="rotateRight()">↷ 90°</button>
                </div>
            </div>

            <div class="control-group">
                <h3>Zoom</h3>
                <div class="slider-control">
                    <label>Zoom Level: <span class="slider-value" id="zoom-value">100%</span></label>
                    <input type="range" id="zoom-slider" min="50" max="400" value="100" step="10">
                </div>
                <div class="button-group">
                    <button onclick="zoomIn()">Zoom In</button>
                    <button onclick="zoomOut()">Zoom Out</button>
                </div>
            </div>

            <div class="control-group">
                <h3>Capture Settings</h3>
                <div class="slider-control">
                    <label>Capture FPS: <span class="slider-value" id="capture-fps-value">15</span></label>
                    <input type="range" id="capture-fps-slider" min="1" max="30" value="15" step="1">
                </div>
                <div class="slider-control">
                    <label>JPEG Quality: <span class="slider-value" id="quality-value">75</span></label>
                    <input type="range" id="quality-slider" min="10" max="100" value="75" step="5">
                </div>
                <div class="slider-control">
                    <label>Stream FPS: <span class="slider-value" id="fps-value">29</span></label>
                    <input type="range" id="fps-slider" min="1" max="29" value="29" step="1">
                </div>
            </div>

            <div class="control-group">
                <h3>Reset</h3>
                <button class="primary" onclick="resetAll()">Reset All Settings</button>
            </div>

            <div class="control-group">
                <h3>Capture</h3>
                <button class="primary" onclick="takeScreenshot()">📸 Take Screenshot</button>
                <button class="primary" id="record-btn" onclick="toggleRecording()">⏺ Start Recording</button>
                <div id="recording-status" style="display: none; margin-top: 10px; text-align: center; color: #ff4444; font-weight: 600;">
                    ⏺ Recording: <span id="recording-time">0:00</span>
                </div>
            </div>
        </div>

        <div class="main-content">
            <div class="image-container">
                <div class="image-wrapper">
                    <img id="microscope-image" src="/stream.mjpg" alt="Microscope feed">
                </div>
                <div class="info">
                    Processed in Python • <span class="fps" id="actual-fps">-- FPS</span>
                </div>
            </div>
        </div>
    </div>

    <script>
        const img = document.getElementById('microscope-image');
        const zoomSlider = document.getElementById('zoom-slider');
        const zoomValue = document.getElementById('zoom-value');
        const fpsSlider = document.getElementById('fps-slider');
        const fpsValue = document.getElementById('fps-value');
        const actualFpsDisplay = document.getElementById('actual-fps');
        const sidebar = document.getElementById('sidebar');
        const mobileToggle = document.getElementById('mobile-toggle');
        const mobileOverlay = document.getElementById('mobile-overlay');

        let frameCount = 0;
        let lastFpsUpdate = Date.now();

        // Mobile sidebar toggle
        function toggleSidebar() {
            const isHidden = sidebar.classList.toggle('mobile-hidden');
            if (window.innerWidth <= 1024) {
                mobileOverlay.style.display = isHidden ? 'none' : 'block';
                setTimeout(() => {
                    mobileOverlay.classList.toggle('active', !isHidden);
                }, 10);
            }
        }

        function closeSidebar() {
            sidebar.classList.add('mobile-hidden');
            mobileOverlay.classList.remove('active');
            setTimeout(() => {
                mobileOverlay.style.display = 'none';
            }, 300);
        }

        function handleResize() {
            if (window.innerWidth <= 1024) {
                if (!sidebar.classList.contains('mobile-hidden')) {
                    closeSidebar();
                }
            } else {
                sidebar.classList.remove('mobile-hidden');
                mobileOverlay.style.display = 'none';
                mobileOverlay.classList.remove('active');
            }
        }

        handleResize();

        let resizeTimeout;
        window.addEventListener('resize', function() {
            clearTimeout(resizeTimeout);
            resizeTimeout = setTimeout(handleResize, 100);
        });

        // Track actual FPS
        img.addEventListener('load', function() {
            frameCount++;
            const now = Date.now();

            // Debug every 30 frames
            if (frameCount % 30 === 0) {
                const timestamp = new Date().toISOString();
                console.log(`[${timestamp}] BROWSER: Received and displayed frame ${frameCount}`);
            }

            if (now - lastFpsUpdate >= 1000) {
                actualFpsDisplay.textContent = frameCount + ' FPS';
                frameCount = 0;
                lastFpsUpdate = now;
            }
        });

        // Brightness slider
        const brightnessSlider = document.getElementById('brightness-slider');
        const brightnessValue = document.getElementById('brightness-value');
        brightnessSlider.addEventListener('input', function() {
            brightnessValue.textContent = this.value;
        });
        brightnessSlider.addEventListener('change', function() {
            const value = this.value;
            const timestamp = new Date().toISOString();
            console.log(`[${timestamp}] BROWSER: Sending brightness=${value} to server`);
            fetch(`/process/brightness/${value}`, { method: 'POST' })
                .then(response => {
                    const ts = new Date().toISOString();
                    console.log(`[${ts}] BROWSER: Server responded OK to brightness change`);
                })
                .catch(err => {
                    const ts = new Date().toISOString();
                    console.error(`[${ts}] BROWSER: Failed to set brightness:`, err);
                });
        });

        // Contrast slider (display as 0.1-3.0, send as 10-300)
        const contrastSlider = document.getElementById('contrast-slider');
        const contrastValue = document.getElementById('contrast-value');
        contrastSlider.addEventListener('input', function() {
            const displayValue = (this.value / 100).toFixed(1);
            contrastValue.textContent = displayValue;
        });
        contrastSlider.addEventListener('change', function() {
            const value = this.value;
            fetch(`/process/contrast/${value}`, { method: 'POST' })
                .catch(err => console.error('Failed to set contrast:', err));
        });

        // Saturation slider (display as 0.0-3.0, send as 0-300)
        const saturationSlider = document.getElementById('saturation-slider');
        const saturationValue = document.getElementById('saturation-value');
        saturationSlider.addEventListener('input', function() {
            const displayValue = (this.value / 100).toFixed(1);
            saturationValue.textContent = displayValue;
        });
        saturationSlider.addEventListener('change', function() {
            const value = this.value;
            fetch(`/process/saturation/${value}`, { method: 'POST' })
                .catch(err => console.error('Failed to set saturation:', err));
        });

        // Flip controls
        function flipHorizontal() {
            const timestamp = new Date().toISOString();
            console.log(`[${timestamp}] BROWSER: Sending flip_h toggle to server`);
            fetch('/process/flip_h/toggle', { method: 'POST' })
                .then(response => {
                    const ts = new Date().toISOString();
                    console.log(`[${ts}] BROWSER: Server responded OK to flip_h`);
                })
                .catch(err => {
                    const ts = new Date().toISOString();
                    console.error(`[${ts}] BROWSER: Failed to flip horizontal:`, err);
                });
        }

        function flipVertical() {
            const timestamp = new Date().toISOString();
            console.log(`[${timestamp}] BROWSER: Sending flip_v toggle to server`);
            fetch('/process/flip_v/toggle', { method: 'POST' })
                .then(response => {
                    const ts = new Date().toISOString();
                    console.log(`[${ts}] BROWSER: Server responded OK to flip_v`);
                })
                .catch(err => {
                    const ts = new Date().toISOString();
                    console.error(`[${ts}] BROWSER: Failed to flip vertical:`, err);
                });
        }

        // Rotation controls
        function rotateLeft() {
            fetch('/process/rotate/-90', { method: 'POST' })
                .catch(err => console.error('Failed to rotate:', err));
        }

        function rotateRight() {
            fetch('/process/rotate/90', { method: 'POST' })
                .catch(err => console.error('Failed to rotate:', err));
        }

        // Zoom controls
        zoomSlider.addEventListener('input', function() {
            zoomValue.textContent = this.value + '%';
        });
        zoomSlider.addEventListener('change', function() {
            const value = this.value;
            fetch(`/process/zoom/${value}`, { method: 'POST' })
                .catch(err => console.error('Failed to set zoom:', err));
        });

        function zoomIn() {
            const currentZoom = parseInt(zoomSlider.value);
            zoomSlider.value = Math.min(400, currentZoom + 25);
            zoomSlider.dispatchEvent(new Event('input'));
            zoomSlider.dispatchEvent(new Event('change'));
        }

        function zoomOut() {
            const currentZoom = parseInt(zoomSlider.value);
            zoomSlider.value = Math.max(50, currentZoom - 25);
            zoomSlider.dispatchEvent(new Event('input'));
            zoomSlider.dispatchEvent(new Event('change'));
        }

        // Capture FPS slider
        const captureFpsSlider = document.getElementById('capture-fps-slider');
        const captureFpsValue = document.getElementById('capture-fps-value');
        captureFpsSlider.addEventListener('input', function() {
            captureFpsValue.textContent = this.value;
        });
        captureFpsSlider.addEventListener('change', function() {
            const value = this.value;
            fetch(`/capture/fps/${value}`, { method: 'POST' })
                .catch(err => console.error('Failed to set capture FPS:', err));
        });

        // JPEG quality slider
        const qualitySlider = document.getElementById('quality-slider');
        const qualityValue = document.getElementById('quality-value');
        qualitySlider.addEventListener('input', function() {
            qualityValue.textContent = this.value;
        });
        qualitySlider.addEventListener('change', function() {
            const value = this.value;
            fetch(`/capture/quality/${value}`, { method: 'POST' })
                .catch(err => console.error('Failed to set JPEG quality:', err));
        });

        // Stream FPS slider
        fpsSlider.addEventListener('input', function() {
            const value = this.value;
            fpsValue.textContent = value;
            img.src = '/stream.mjpg?fps=' + value + '&t=' + Date.now();
        });

        // Reset all settings
        function resetAll() {
            // Reset sliders to defaults
            brightnessSlider.value = 0;
            brightnessValue.textContent = '0';
            contrastSlider.value = 100;
            contrastValue.textContent = '1.0';
            saturationSlider.value = 100;
            saturationValue.textContent = '1.0';
            zoomSlider.value = 100;
            zoomValue.textContent = '100%';

            fetch('/process/reset', { method: 'POST' })
                .catch(err => console.error('Failed to reset:', err));
        }

        function takeScreenshot() {
            // Fetch current processed frame from server
            fetch('/current.jpg')
                .then(response => response.blob())
                .then(blob => {
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
                    a.download = `microscope_${timestamp}.jpg`;
                    a.click();
                    URL.revokeObjectURL(url);
                })
                .catch(err => console.error('Failed to take screenshot:', err));
        }

        // Video recording functionality
        let mediaRecorder = null;
        let recordedChunks = [];
        let recordingStartTime = null;
        let recordingInterval = null;

        function toggleRecording() {
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                stopRecording();
            } else {
                startRecording();
            }
        }

        async function startRecording() {
            try {
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');

                canvas.width = img.naturalWidth;
                canvas.height = img.naturalHeight;

                const stream = canvas.captureStream(29);

                let animationId;
                function drawFrame() {
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    ctx.drawImage(img, 0, 0);

                    if (mediaRecorder && mediaRecorder.state === 'recording') {
                        animationId = requestAnimationFrame(drawFrame);
                    }
                }

                recordedChunks = [];
                const options = {
                    mimeType: 'video/webm;codecs=vp9',
                    videoBitsPerSecond: 8000000
                };

                if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                    options.mimeType = 'video/webm;codecs=vp8';
                }

                mediaRecorder = new MediaRecorder(stream, options);

                mediaRecorder.ondataavailable = function(e) {
                    if (e.data.size > 0) {
                        recordedChunks.push(e.data);
                    }
                };

                mediaRecorder.onstop = function() {
                    const blob = new Blob(recordedChunks, { type: 'video/webm' });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
                    a.download = `microscope_video_${timestamp}.webm`;
                    a.click();
                    URL.revokeObjectURL(url);

                    cancelAnimationFrame(animationId);
                };

                mediaRecorder.start(1000);
                drawFrame();

                document.getElementById('record-btn').textContent = '⏹ Stop Recording';
                document.getElementById('recording-status').style.display = 'block';

                recordingStartTime = Date.now();
                recordingInterval = setInterval(updateRecordingTime, 1000);

            } catch (err) {
                console.error('Error starting recording:', err);
                alert('Failed to start recording: ' + err.message);
            }
        }

        function stopRecording() {
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                mediaRecorder.stop();

                document.getElementById('record-btn').textContent = '⏺ Start Recording';
                document.getElementById('recording-status').style.display = 'none';

                if (recordingInterval) {
                    clearInterval(recordingInterval);
                    recordingInterval = null;
                }
            }
        }

        function updateRecordingTime() {
            if (recordingStartTime) {
                const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
                const minutes = Math.floor(elapsed / 60);
                const seconds = elapsed % 60;
                document.getElementById('recording-time').textContent =
                    `${minutes}:${seconds.toString().padStart(2, '0')}`;
            }
        }

        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {
            if (e.key === 's' || e.key === 'S') {
                takeScreenshot();
            } else if (e.key === 'r' || e.key === 'R') {
                if (e.ctrlKey || e.metaKey) {
                    toggleRecording();
                } else {
                    resetAll();
                }
            } else if (e.key === 'h' || e.key === 'H') {
                flipHorizontal();
            } else if (e.key === 'v' || e.key === 'V') {
                flipVertical();
            } else if (e.key === '+' || e.key === '=') {
                zoomIn();
            } else if (e.key === '-' || e.key === '_') {
                zoomOut();
            }
        });
    </script>
</body>
</html>"""
            self.wfile.write(html.encode('utf-8'))

        elif self.path.startswith('/stream.mjpg'):
            # Parse FPS from query string
            fps = target_fps
            if '?' in self.path:
                query = self.path.split('?')[1]
                for param in query.split('&'):
                    if param.startswith('fps='):
                        try:
                            fps = int(param.split('=')[1])
                            fps = max(1, min(29, fps))
                        except:
                            pass

            # Serve MJPEG stream
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '0')
            self.send_header('Connection', 'close')
            self.end_headers()

            # Disable buffering on the socket for immediate frame delivery
            try:
                import socket
                self.wfile.flush()
                self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except:
                pass

            frame_delay = 1.0 / fps
            last_frame_number = -1
            stream_frame_count = 0

            try:
                last_send_time = time.time()

                while running:
                    with frame_lock:
                        frame = current_frame
                        frame_number = current_frame_number

                    # Send frame if it's new (different frame number)
                    if frame and frame_number != last_frame_number:
                        # Check if enough time has elapsed
                        elapsed = time.time() - last_send_time
                        if elapsed < frame_delay:
                            # Sleep in small increments to stay responsive
                            time.sleep(min(0.01, frame_delay - elapsed))
                            continue

                        last_send_time = time.time()

                        # Write MJPEG frame part by part with flushes
                        try:
                            self.wfile.write(b'--frame\r\n')
                            self.wfile.flush()

                            self.wfile.write(b'Content-Type: image/jpeg\r\n')
                            self.wfile.flush()

                            self.wfile.write(f'Content-Length: {len(frame)}\r\n\r\n'.encode())
                            self.wfile.flush()

                            self.wfile.write(frame)
                            self.wfile.flush()

                            self.wfile.write(b'\r\n')
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError):
                            # Client disconnected
                            break

                        last_frame_number = frame_number
                        stream_frame_count += 1

                        # Debug: Print every 30 frames sent
                        if stream_frame_count % 30 == 0:
                            timestamp = datetime.now().isoformat()
                            print(f"[{timestamp}] PYTHON STREAM: Sent frame {stream_frame_count} to browser, frame_num={frame_number}")
                    else:
                        # No new frame, sleep briefly
                        time.sleep(0.01)
            except:
                pass

        elif self.path.startswith('/current.jpg'):
            # Serve a single current frame (for screenshot/debugging)
            with frame_lock:
                if current_frame:
                    self.send_response(200)
                    self.send_header('Content-type', 'image/jpeg')
                    self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                    self.send_header('Pragma', 'no-cache')
                    self.send_header('Expires', '0')
                    self.end_headers()
                    self.wfile.write(current_frame)
                else:
                    self.send_response(404)
                    self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        global processing_settings, capture_fps, jpeg_quality

        # Handle image processing settings
        if self.path.startswith('/process/'):
            parts = self.path.split('/')

            if parts[2] == 'reset':
                # Reset all processing settings to defaults
                with processing_lock:
                    processing_settings['brightness'] = 0
                    processing_settings['contrast'] = 1.0
                    processing_settings['saturation'] = 1.0
                    processing_settings['flip_h'] = True
                    processing_settings['flip_v'] = True
                    processing_settings['rotate'] = 0
                    processing_settings['zoom'] = 1.0
                print("[PROCESS] Reset all settings to defaults")
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(b'{"status": "ok"}')
                return

            elif len(parts) == 4:
                setting = parts[2]
                value_str = parts[3]

                try:
                    timestamp = datetime.now().isoformat()
                    if setting == 'brightness':
                        value = int(value_str)
                        with processing_lock:
                            processing_settings['brightness'] = value
                        print(f"[{timestamp}] PYTHON: Received brightness={value}, updated settings")

                    elif setting == 'contrast':
                        value = int(value_str) / 100.0  # Convert 10-300 to 0.1-3.0
                        with processing_lock:
                            processing_settings['contrast'] = value
                        print(f"[{timestamp}] PYTHON: Received contrast={value:.2f}, updated settings")

                    elif setting == 'saturation':
                        value = int(value_str) / 100.0  # Convert 0-300 to 0.0-3.0
                        with processing_lock:
                            processing_settings['saturation'] = value
                        print(f"[{timestamp}] PYTHON: Received saturation={value:.2f}, updated settings")

                    elif setting == 'flip_h' and value_str == 'toggle':
                        with processing_lock:
                            processing_settings['flip_h'] = not processing_settings['flip_h']
                            new_state = processing_settings['flip_h']
                        print(f"[{timestamp}] PYTHON: Received flip_h toggle, new state={new_state}")

                    elif setting == 'flip_v' and value_str == 'toggle':
                        with processing_lock:
                            processing_settings['flip_v'] = not processing_settings['flip_v']
                            new_state = processing_settings['flip_v']
                        print(f"[{timestamp}] PYTHON: Received flip_v toggle, new state={new_state}")

                    elif setting == 'rotate':
                        delta = int(value_str)
                        with processing_lock:
                            processing_settings['rotate'] = (processing_settings['rotate'] + delta) % 360
                            new_rotation = processing_settings['rotate']
                        print(f"[{timestamp}] PYTHON: Received rotation delta={delta}, new rotation={new_rotation}°")

                    elif setting == 'zoom':
                        value = int(value_str) / 100.0  # Convert 50-400 to 0.5-4.0
                        with processing_lock:
                            processing_settings['zoom'] = value
                        print(f"[{timestamp}] PYTHON: Received zoom={value:.2f}x, updated settings")

                    else:
                        self.send_response(400)
                        self.end_headers()
                        return

                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(b'{"status": "ok"}')
                    return
                except Exception as e:
                    print(f"[PROCESS] Error: {e}")
                    self.send_response(400)
                    self.end_headers()
                    return

        # Handle capture settings API
        elif self.path.startswith('/capture/'):
            parts = self.path.split('/')
            if len(parts) == 4:
                setting = parts[2]
                try:
                    value = int(parts[3])

                    if setting == 'fps' and 1 <= value <= 30:
                        global capture_fps
                        capture_fps = value
                        print(f"[CAPTURE] Capture FPS set to: {value}")

                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        self.wfile.write(b'{"status": "ok"}')
                        return

                    elif setting == 'quality' and 10 <= value <= 100:
                        global jpeg_quality
                        jpeg_quality = value
                        print(f"[CAPTURE] JPEG quality set to: {value}")

                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        self.wfile.write(b'{"status": "ok"}')
                        return
                except:
                    pass

            self.send_response(400)
            self.end_headers()

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *arguments):
        # Suppress logging
        pass

def capture_usb():
    """Capture frames from USB microscope and apply processing"""
    global current_frame, running, connection_mode, usb_camera_cap

    if not USB_AVAILABLE:
        print("ERROR: OpenCV not available for USB capture")
        return

    print("Attempting to connect to USB microscope...")

    # Try to open the USB camera
    cap = None
    for device_id in range(5):
        cap = cv2.VideoCapture(device_id)
        if cap.isOpened():
            print(f"Found USB microscope on device {device_id}")
            connection_mode = 'usb'
            break
        cap.release()
        cap = None

    if not cap or not cap.isOpened():
        print("No USB microscope found")
        return

    usb_camera_cap = cap

    # Set camera properties optimized for Raspberry Pi 3
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print(f"Capturing from USB microscope...")
    print(f"All image processing done in Python - sliders affect image in real-time\n")

    last_capture_time = time.time()
    frame_counter = 0
    last_debug_time = time.time()

    while running:
        # Read capture settings
        global capture_fps, jpeg_quality
        current_fps = capture_fps
        current_quality = jpeg_quality

        # Check if enough time has elapsed for next frame
        if current_fps > 0:
            frame_delay = 1.0 / current_fps
            elapsed = time.time() - last_capture_time
            if elapsed < frame_delay:
                # Sleep in small increments to remain responsive
                time.sleep(min(0.01, frame_delay - elapsed))
                continue  # Check again next iteration

        last_capture_time = time.time()

        ret, frame = cap.read()
        if ret:
            frame_counter += 1

            # Debug every 30 frames
            if frame_counter % 30 == 0:
                actual_fps = 30.0 / (time.time() - last_debug_time)
                last_debug_time = time.time()
                timestamp = datetime.now().isoformat()
                with processing_lock:
                    settings_str = f"B:{processing_settings['brightness']} C:{processing_settings['contrast']:.1f} S:{processing_settings['saturation']:.1f} Z:{processing_settings['zoom']:.1f}x"
                print(f"[{timestamp}] PYTHON CAPTURE: Processing frame {frame_counter} with settings: {settings_str}, {actual_fps:.1f} fps")

            # Apply all image processing in Python
            processed_frame = apply_image_processing(frame)

            # Encode to JPEG
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, current_quality]

            with SuppressStderr():
                ret, jpeg = cv2.imencode('.jpg', processed_frame, encode_params)

            if ret:
                global current_frame_number
                new_frame = jpeg.tobytes()
                with frame_lock:
                    current_frame = new_frame
                    current_frame_number += 1
                    frame_num = current_frame_number
                frame_event.set()
                # Debug: Print when we update the frame
                if frame_counter % 30 == 0:
                    timestamp = datetime.now().isoformat()
                    print(f"[{timestamp}] PYTHON CAPTURE: Encoded and stored frame_num={frame_num}")
        else:
            time.sleep(0.01)

    cap.release()
    print("USB capture stopped")

def capture_wifi():
    """Capture frames from WiFi microscope and apply processing"""
    global current_frame, running, connection_mode

    print(f"Attempting to connect to WiFi microscope at {HOST}...")

    try:
        # Open command socket for sending
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.settimeout(2)
            s.sendto(b"JHCMD\x10\x00", (HOST, SPORT))
            s.sendto(b"JHCMD\x20\x00", (HOST, SPORT))
            s.sendto(b"JHCMD\xd0\x01", (HOST, SPORT))
            s.sendto(b"JHCMD\xd0\x01", (HOST, SPORT))

            # Open receive socket and bind to receive port
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as r:
                r.bind(("", RPORT))
                r.setblocking(0)

                frame_buffer = bytearray()
                last_framecount = -1
                heartbeat_counter = 0

                connection_mode = 'wifi'
                print("Connected to WiFi microscope, receiving data...")
                print("All image processing done in Python - sliders affect image in real-time\n")

                frame_counter = 0
                last_debug_time = time.time()

                while running:
                    try:
                        data = r.recv(1450)
                        if len(data) > 8:
                            framecount = data[0] + data[1]*256
                            packetcount = data[3]

                            if packetcount == 0:
                                # New frame - process and save previous
                                if frame_buffer and last_framecount != framecount:
                                    if len(frame_buffer) >= 4 and frame_buffer[0:2] == b'\xff\xd8':
                                        if frame_buffer[-2:] == b'\xff\xd9':
                                            # Decode JPEG to apply processing
                                            global current_frame_number
                                            if USB_AVAILABLE:
                                                nparr = np.frombuffer(frame_buffer, np.uint8)
                                                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                                                if frame is not None:
                                                    frame_counter += 1

                                                    # Debug every 30 frames
                                                    if frame_counter % 30 == 0:
                                                        actual_fps = 30.0 / (time.time() - last_debug_time)
                                                        last_debug_time = time.time()
                                                        with processing_lock:
                                                            settings_str = f"B:{processing_settings['brightness']} C:{processing_settings['contrast']:.1f} S:{processing_settings['saturation']:.1f} Z:{processing_settings['zoom']:.1f}x"
                                                        print(f"[DEBUG] {actual_fps:.1f} fps | {settings_str}")

                                                    # Apply processing
                                                    processed_frame = apply_image_processing(frame)

                                                    # Re-encode
                                                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
                                                    with SuppressStderr():
                                                        ret, jpeg = cv2.imencode('.jpg', processed_frame, encode_params)

                                                    if ret:
                                                        with frame_lock:
                                                            current_frame = jpeg.tobytes()
                                                            current_frame_number += 1
                                                        frame_event.set()
                                            else:
                                                # No OpenCV - just pass through raw frames
                                                with frame_lock:
                                                    current_frame = bytes(frame_buffer)
                                                    current_frame_number += 1
                                                frame_event.set()

                                frame_buffer = bytearray()
                                last_framecount = framecount
                                frame_buffer.extend(data[24:])

                                heartbeat_counter += 1
                                if heartbeat_counter % 50 == 0:
                                    s.sendto(b"JHCMD\xd0\x01", (HOST, SPORT))
                            else:
                                frame_buffer.extend(data[8:])
                    except:
                        time.sleep(0.001)

            s.sendto(b"JHCMD\xd0\x02", (HOST, SPORT))
            print("WiFi capture stopped")
    except Exception as e:
        print(f"WiFi connection failed: {e}")

def capture_microscope():
    """Capture frames from microscope - tries WiFi first, then USB"""
    global connection_mode, current_frame

    print("\n" + "="*50)
    print("Detecting microscope connection...")
    print("="*50)

    # Check WiFi first
    wifi_available = False
    try:
        test_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        test_sock.settimeout(0.5)
        test_sock.sendto(b"JHCMD\x10\x00", (HOST, SPORT))
        test_sock.sendto(b"JHCMD\x20\x00", (HOST, SPORT))
        test_sock.sendto(b"JHCMD\xd0\x01", (HOST, SPORT))
        test_sock.close()

        recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        recv_sock.settimeout(2)
        recv_sock.bind(("", RPORT))

        data = recv_sock.recv(1450)
        if len(data) > 8:
            wifi_available = True
        recv_sock.close()
    except:
        pass

    if wifi_available:
        print("✓ WiFi microscope detected")
        capture_wifi()
    elif USB_AVAILABLE:
        print("✗ WiFi microscope not found, trying USB...")
        capture_usb()
    else:
        print("✗ No microscope found (WiFi not reachable, OpenCV not available for USB)")
        print("\nTo enable USB support, install OpenCV:")
        print("  pip3 install opencv-python")

def main():
    global running, connection_mode

    print("="*50)
    print("Microscope Web Viewer (Python Image Processing)")
    print("="*50)
    print("All image processing happens in Python")
    print("Sliders affect the image in real-time")
    print("="*50)

    # Start microscope capture in a separate thread
    capture_thread = threading.Thread(target=capture_microscope, daemon=True)
    capture_thread.start()

    # Give it a moment to detect and connect
    time.sleep(3)

    if connection_mode:
        mode_str = "WiFi (UDP)" if connection_mode == 'wifi' else "USB Camera"
        print(f"\n✓ Connected via: {mode_str}")
    else:
        print("\n⚠ No microscope detected - waiting for connection...")

    # Start web server with threading for better concurrent connection handling
    server = ThreadingHTTPServer(('', WEB_PORT), MicroscopeHandler)
    print(f"\nWeb viewer running at: http://localhost:{WEB_PORT}")
    print("Press Ctrl+C to stop\n")
    print("Keyboard shortcuts:")
    print("  S - Take screenshot")
    print("  Ctrl+R - Start/Stop recording")
    print("  H - Flip horizontal")
    print("  V - Flip vertical")
    print("  R - Reset all settings")
    print("  +/- - Zoom in/out\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        running = False
        server.shutdown()

if __name__ == "__main__":
    main()
