#!/usr/bin/env python3
# Web-based viewer for WiFi/USB microscope
# Captures images and serves them via HTTP with MJPEG streaming
# Supports both WiFi (UDP) and USB (webcam) modes

import time
import socket
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
import sys
import io
import os
import warnings

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
frame_lock = threading.Lock()
frame_event = threading.Event()
running = True
target_fps = 29  # Default target FPS for streaming
connection_mode = None  # Will be 'wifi', 'usb', or None

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
            height: 100dvh; /* Dynamic viewport height for mobile */
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
            will-change: transform;
            backface-visibility: hidden;
            -webkit-backface-visibility: hidden;
            transform-origin: center center;
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

        /* Desktop-specific styles */
        @media (min-width: 1025px) {
            .sidebar.mobile-hidden {
                transform: none; /* Prevent hiding on desktop */
            }
            .mobile-overlay {
                display: none !important; /* Never show overlay on desktop */
            }
        }

        /* Mobile and Tablet Responsive Styles */
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

        /* Touch-friendly improvements */
        @media (hover: none) and (pointer: coarse) {
            button {
                min-height: 44px; /* iOS touch target size */
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
                <h3>Image Controls</h3>
                <div class="button-group">
                    <button onclick="flipHorizontal()">Flip H</button>
                    <button onclick="flipVertical()">Flip V</button>
                    <button onclick="rotateLeft()">↶ 90°</button>
                    <button onclick="rotateRight()">↷ 90°</button>
                    <button onclick="resetTransform()">Reset</button>
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
                <h3>Stream Settings</h3>
                <div class="slider-control">
                    <label>Target FPS: <span class="slider-value" id="fps-value">29</span></label>
                    <input type="range" id="fps-slider" min="1" max="29" value="29" step="1">
                </div>
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
                    Using MJPEG stream • <span class="fps" id="actual-fps">-- FPS</span>
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

        let transform = {
            scaleX: -1,  // Default flipped horizontal
            scaleY: -1,  // Default flipped vertical
            rotate: 0,
            zoom: 1
        };

        let frameCount = 0;
        let lastFpsUpdate = Date.now();

        // Mobile sidebar toggle
        function toggleSidebar() {
            const isHidden = sidebar.classList.toggle('mobile-hidden');
            if (window.innerWidth <= 1024) {
                mobileOverlay.style.display = isHidden ? 'none' : 'block';
                // Trigger reflow for transition
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

        // Handle initial sidebar state and window resize
        function handleResize() {
            if (window.innerWidth <= 1024) {
                // Mobile/tablet - start with sidebar hidden
                if (!sidebar.classList.contains('mobile-hidden')) {
                    closeSidebar();
                }
            } else {
                // Desktop - ensure sidebar is visible and overlay is hidden
                sidebar.classList.remove('mobile-hidden');
                mobileOverlay.style.display = 'none';
                mobileOverlay.classList.remove('active');
            }
        }

        // Set initial state
        handleResize();

        // Handle window resize
        let resizeTimeout;
        window.addEventListener('resize', function() {
            clearTimeout(resizeTimeout);
            resizeTimeout = setTimeout(handleResize, 100);
        });

        // Track actual FPS
        img.addEventListener('load', function() {
            frameCount++;
            const now = Date.now();
            if (now - lastFpsUpdate >= 1000) {
                actualFpsDisplay.textContent = frameCount + ' FPS';
                frameCount = 0;
                lastFpsUpdate = now;
            }
        });

        function updateTransform() {
            const transforms = [];
            transforms.push(`scale(${transform.scaleX * transform.zoom}, ${transform.scaleY * transform.zoom})`);
            transforms.push(`rotate(${transform.rotate}deg)`);
            img.style.transform = transforms.join(' ');
        }

        // Apply default transform on load
        updateTransform();

        function flipHorizontal() {
            transform.scaleX *= -1;
            updateTransform();
        }

        function flipVertical() {
            transform.scaleY *= -1;
            updateTransform();
        }

        function rotateLeft() {
            transform.rotate -= 90;
            updateTransform();
        }

        function rotateRight() {
            transform.rotate += 90;
            updateTransform();
        }

        function resetTransform() {
            transform.scaleX = 1;
            transform.scaleY = 1;
            transform.rotate = 0;
            updateTransform();
        }

        function zoomIn() {
            const currentZoom = parseInt(zoomSlider.value);
            zoomSlider.value = Math.min(400, currentZoom + 25);
            zoomSlider.dispatchEvent(new Event('input'));
        }

        function zoomOut() {
            const currentZoom = parseInt(zoomSlider.value);
            zoomSlider.value = Math.max(50, currentZoom - 25);
            zoomSlider.dispatchEvent(new Event('input'));
        }

        zoomSlider.addEventListener('input', function() {
            const value = this.value;
            zoomValue.textContent = value + '%';
            transform.zoom = value / 100;
            updateTransform();
        });

        fpsSlider.addEventListener('input', function() {
            const value = this.value;
            fpsValue.textContent = value;
            // Reconnect stream with new FPS
            img.src = '/stream.mjpg?fps=' + value + '&t=' + Date.now();
        });

        function takeScreenshot() {
            // Create a canvas to capture the current frame
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');

            // Set canvas size to match image natural size
            canvas.width = img.naturalWidth;
            canvas.height = img.naturalHeight;

            // Apply transforms to canvas
            ctx.save();
            ctx.translate(canvas.width / 2, canvas.height / 2);
            ctx.scale(transform.scaleX, transform.scaleY);
            ctx.rotate(transform.rotate * Math.PI / 180);
            ctx.drawImage(img, -canvas.width / 2, -canvas.height / 2);
            ctx.restore();

            // Download the image
            canvas.toBlob(function(blob) {
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
                a.download = `microscope_${timestamp}.jpg`;
                a.click();
                URL.revokeObjectURL(url);
            }, 'image/jpeg', 0.95);
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
                // Create a canvas to capture the stream with transforms
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');

                // Set canvas size to match image
                canvas.width = img.naturalWidth;
                canvas.height = img.naturalHeight;

                // Capture stream from canvas
                const stream = canvas.captureStream(29); // 29 fps

                // Draw frames to canvas continuously
                let animationId;
                function drawFrame() {
                    ctx.save();
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    ctx.translate(canvas.width / 2, canvas.height / 2);
                    ctx.scale(transform.scaleX, transform.scaleY);
                    ctx.rotate(transform.rotate * Math.PI / 180);
                    ctx.drawImage(img, -canvas.width / 2, -canvas.height / 2);
                    ctx.restore();

                    if (mediaRecorder && mediaRecorder.state === 'recording') {
                        animationId = requestAnimationFrame(drawFrame);
                    }
                }

                // Set up media recorder
                recordedChunks = [];
                const options = {
                    mimeType: 'video/webm;codecs=vp9',
                    videoBitsPerSecond: 8000000 // 8 Mbps for high quality
                };

                // Fallback for browsers that don't support vp9
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

                mediaRecorder.start(1000); // Collect data every second
                drawFrame();

                // Update UI
                document.getElementById('record-btn').textContent = '⏹ Stop Recording';
                document.getElementById('recording-status').style.display = 'block';

                // Start timer
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

                // Update UI
                document.getElementById('record-btn').textContent = '⏺ Start Recording';
                document.getElementById('recording-status').style.display = 'none';

                // Stop timer
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
                    resetTransform();
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

        // Touch gesture controls for mobile
        let touchStartDistance = 0;
        let touchStartZoom = 1;
        let lastTap = 0;

        img.addEventListener('touchstart', function(e) {
            if (e.touches.length === 2) {
                // Pinch to zoom
                e.preventDefault();
                const touch1 = e.touches[0];
                const touch2 = e.touches[1];
                touchStartDistance = Math.hypot(
                    touch2.clientX - touch1.clientX,
                    touch2.clientY - touch1.clientY
                );
                touchStartZoom = transform.zoom;
            } else if (e.touches.length === 1) {
                // Double tap to reset zoom
                const currentTime = new Date().getTime();
                const tapLength = currentTime - lastTap;
                if (tapLength < 300 && tapLength > 0) {
                    e.preventDefault();
                    resetTransform();
                }
                lastTap = currentTime;
            }
        }, { passive: false });

        img.addEventListener('touchmove', function(e) {
            if (e.touches.length === 2) {
                e.preventDefault();
                const touch1 = e.touches[0];
                const touch2 = e.touches[1];
                const touchDistance = Math.hypot(
                    touch2.clientX - touch1.clientX,
                    touch2.clientY - touch1.clientY
                );

                const scale = touchDistance / touchStartDistance;
                const newZoom = Math.max(0.5, Math.min(4, touchStartZoom * scale));

                transform.zoom = newZoom;
                zoomSlider.value = newZoom * 100;
                zoomValue.textContent = Math.round(newZoom * 100) + '%';
                updateTransform();
            }
        }, { passive: false });

        // Prevent default zoom behavior on iOS
        document.addEventListener('gesturestart', function(e) {
            e.preventDefault();
        });

        document.addEventListener('gesturechange', function(e) {
            e.preventDefault();
        });

        document.addEventListener('gestureend', function(e) {
            e.preventDefault();
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
                            fps = max(1, min(29, fps))  # Clamp between 1-29
                        except:
                            pass

            # Serve MJPEG stream
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '0')
            self.end_headers()

            frame_delay = 1.0 / fps
            last_frame = None

            try:
                while running:
                    with frame_lock:
                        frame = current_frame

                    if frame and frame != last_frame:
                        # Write MJPEG frame with proper Content-Length header
                        self.wfile.write(b'--frame\r\n')
                        self.wfile.write(b'Content-Type: image/jpeg\r\n')
                        self.wfile.write(f'Content-Length: {len(frame)}\r\n\r\n'.encode())
                        self.wfile.write(frame)
                        self.wfile.write(b'\r\n')
                        self.wfile.flush()  # Force flush to prevent buffering issues
                        last_frame = frame

                    time.sleep(frame_delay)
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

    def log_message(self, fmt, *arguments):
        # Suppress logging
        pass

def capture_usb():
    """Capture frames from USB microscope"""
    global current_frame, running, connection_mode

    if not USB_AVAILABLE:
        print("ERROR: OpenCV not available for USB capture")
        return

    print("Attempting to connect to USB microscope...")

    # Try to open the USB camera
    cap = None
    for device_id in range(5):  # Try first 5 camera devices
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

    # Set camera properties for best performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer to reduce lag

    encoding_method = "PIL" if PIL_AVAILABLE else "OpenCV"
    print(f"Capturing from USB microscope (using {encoding_method} encoding)...")

    frame_count = 0
    debug_frame_count = 0

    while running:
        ret, frame = cap.read()
        if ret:
            frame_count += 1

            # Debug: Print frame info every 30 frames
            if frame_count % 30 == 1:
                height, width = frame.shape[:2]
                print(f"\n[DEBUG] Frame #{frame_count}")
                print(f"  Original dimensions: {width}x{height}")
                print(f"  Frame dtype: {frame.dtype}, shape: {frame.shape}")

            # Ensure frame dimensions are multiples of 16 (JPEG MCU size)
            height, width = frame.shape[:2]
            new_height = (height // 16) * 16
            new_width = (width // 16) * 16

            if new_height != height or new_width != width:
                if frame_count % 30 == 1:
                    print(f"  Resizing to: {new_width}x{new_height}")
                frame = cv2.resize(frame, (new_width, new_height))

            # Use PIL for better JPEG encoding if available, otherwise OpenCV
            if PIL_AVAILABLE:
                try:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Create PIL Image
                    pil_image = Image.fromarray(frame_rgb)
                    # Encode to JPEG with settings that avoid libjpeg warnings
                    buffer = io.BytesIO()
                    # Use 4:4:4 subsampling (no chroma subsampling) to avoid warnings
                    pil_image.save(buffer,
                                 format='JPEG',
                                 quality=80,
                                 optimize=False,
                                 progressive=False,
                                 subsampling=0)  # 0 = 4:4:4 (no subsampling)
                    jpeg_bytes = buffer.getvalue()

                    # Debug JPEG output
                    if frame_count % 30 == 1:
                        print(f"  PIL JPEG size: {len(jpeg_bytes)} bytes")
                        print(f"  Start marker: {jpeg_bytes[:2].hex()}")
                        print(f"  End marker: {jpeg_bytes[-2:].hex()}")
                        # Check for extra bytes after end marker
                        end_pos = jpeg_bytes.rfind(b'\xff\xd9')
                        extra_bytes = len(jpeg_bytes) - (end_pos + 2)
                        print(f"  Bytes after end marker: {extra_bytes}")

                    # Validate JPEG structure
                    if len(jpeg_bytes) < 4 or jpeg_bytes[:2] != b'\xff\xd8' or jpeg_bytes[-2:] != b'\xff\xd9':
                        print(f"[WARNING] Invalid JPEG structure on frame {frame_count}")
                        continue
                except Exception as e:
                    print(f"[ERROR] PIL encoding failed on frame {frame_count}: {e}")
                    continue
            else:
                # Fallback to OpenCV encoding with baseline settings
                encode_params = [
                    cv2.IMWRITE_JPEG_QUALITY, 85,
                    cv2.IMWRITE_JPEG_OPTIMIZE, 0,
                    cv2.IMWRITE_JPEG_PROGRESSIVE, 0,
                    cv2.IMWRITE_JPEG_LUMA_QUALITY, 85,
                    cv2.IMWRITE_JPEG_CHROMA_QUALITY, 85
                ]
                ret, jpeg = cv2.imencode('.jpg', frame, encode_params)
                if not ret:
                    print(f"[ERROR] OpenCV encoding failed on frame {frame_count}")
                    continue
                jpeg_bytes = jpeg.tobytes()

                # Debug JPEG output
                if frame_count % 30 == 1:
                    print(f"  OpenCV JPEG size: {len(jpeg_bytes)} bytes")
                    print(f"  Start marker: {jpeg_bytes[:2].hex()}")
                    end_pos = jpeg_bytes.rfind(b'\xff\xd9')
                    print(f"  End marker position: {end_pos} (total: {len(jpeg_bytes)})")
                    extra_bytes = len(jpeg_bytes) - (end_pos + 2)
                    print(f"  Bytes after end marker: {extra_bytes}")

                # Strip any extraneous bytes after JPEG end marker (FF D9)
                end_marker_pos = jpeg_bytes.rfind(b'\xff\xd9')
                if end_marker_pos == -1:
                    print(f"[ERROR] No end marker found on frame {frame_count}")
                    continue
                jpeg_bytes = jpeg_bytes[:end_marker_pos + 2]

                # Validate start marker
                if len(jpeg_bytes) < 4 or jpeg_bytes[:2] != b'\xff\xd8':
                    print(f"[ERROR] No start marker found on frame {frame_count}")
                    continue

            with frame_lock:
                current_frame = jpeg_bytes
            frame_event.set()

            # Rate limiting
            time.sleep(0.05)  # ~20 fps max
        else:
            time.sleep(0.01)

    cap.release()
    print("USB capture stopped")

def capture_wifi():
    """Capture frames from WiFi microscope"""
    global current_frame, running, connection_mode

    print(f"Attempting to connect to WiFi microscope at {HOST}...")

    try:
        # Open command socket for sending
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.settimeout(2)
            # Send commands like naInit_Re() would do
            s.sendto(b"JHCMD\x10\x00", (HOST, SPORT))
            s.sendto(b"JHCMD\x20\x00", (HOST, SPORT))
            # Heartbeat command, starts the transmission of data from the scope
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

                while running:
                    try:
                        data = r.recv(1450)
                        if len(data) > 8:
                            # Header
                            framecount = data[0] + data[1]*256
                            packetcount = data[3]

                            # Data
                            if packetcount == 0:
                                # New frame started - validate and save previous frame
                                if frame_buffer and last_framecount != framecount:
                                    # Validate JPEG: check for proper start (FF D8) and end (FF D9) markers
                                    if len(frame_buffer) >= 4 and frame_buffer[0:2] == b'\xff\xd8':
                                        # Check for end marker in last few bytes
                                        if frame_buffer[-2:] == b'\xff\xd9':
                                            with frame_lock:
                                                current_frame = bytes(frame_buffer)
                                            frame_event.set()

                                # Reset buffer for new frame
                                frame_buffer = bytearray()
                                last_framecount = framecount

                                # First packet has extra 16-byte header, skip 24 bytes total
                                frame_buffer.extend(data[24:])

                                # Send heartbeat every 50 frames
                                heartbeat_counter += 1
                                if heartbeat_counter % 50 == 0:
                                    s.sendto(b"JHCMD\xd0\x01", (HOST, SPORT))
                            else:
                                # Subsequent packets only have standard 8-byte header
                                frame_buffer.extend(data[8:])
                    except:
                        time.sleep(0.001)

            # Stop data command
            s.sendto(b"JHCMD\xd0\x02", (HOST, SPORT))
            print("WiFi capture stopped")
    except Exception as e:
        print(f"WiFi connection failed: {e}")

def capture_microscope():
    """Capture frames from microscope - tries WiFi first, then USB"""
    global connection_mode, current_frame

    # Try WiFi first
    print("\n" + "="*50)
    print("Detecting microscope connection...")
    print("="*50)

    # Check if WiFi microscope is reachable by trying to get actual data
    wifi_available = False
    try:
        # Send init commands
        test_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        test_sock.settimeout(0.5)
        test_sock.sendto(b"JHCMD\x10\x00", (HOST, SPORT))
        test_sock.sendto(b"JHCMD\x20\x00", (HOST, SPORT))
        test_sock.sendto(b"JHCMD\xd0\x01", (HOST, SPORT))
        test_sock.close()

        # Try to receive data on the receive port
        recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        recv_sock.settimeout(2)
        recv_sock.bind(("", RPORT))

        # Wait for actual data
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
    print("Microscope Web Viewer (WiFi/USB)")
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

    # Start web server
    server = HTTPServer(('', WEB_PORT), MicroscopeHandler)
    print(f"\nWeb viewer running at: http://localhost:{WEB_PORT}")
    print("Press Ctrl+C to stop\n")
    print("Keyboard shortcuts:")
    print("  S - Take screenshot")
    print("  Ctrl+R - Start/Stop recording")
    print("  H - Flip horizontal")
    print("  V - Flip vertical")
    print("  R - Reset transform")
    print("  +/- - Zoom in/out\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        running = False
        server.shutdown()

if __name__ == "__main__":
    main()
