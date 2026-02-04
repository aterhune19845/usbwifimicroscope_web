#!/usr/bin/env python3
import time
import threading
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import cv2
import numpy as np
import usb.core
import usb.util

WEB_PORT = 1337
current_frame = None
frame_lock = threading.Lock()
running = True

settings = {
    'brightness': 0,
    'contrast': 1.0,
    'saturation': 1.0,
    'gain': 1.0,
    'flip_h': True,
    'flip_v': True,
    'rotate': 0,
    'zoom': 1.0,
    'use_lanczos': True,       # High-quality Lanczos upscaling when zooming
    'use_ai_upscale': False,   # AI super-resolution (slower but best quality)
    'stabilize': False,
    'stab_noise': 0,       # Noise filter OFF
    'stab_smooth': 0,      # Smoothing OFF
    'stab_decay': 0,       # Decay OFF
    'stab_blend': 3,       # Frame blending
    'stab_use_lowpass': False,   # Low-pass OFF
    'stab_use_kalman': False,   # Kalman OFF
    'stab_use_ema': False,     # EMA blending OFF
    'stab_ema_type': 'regular',  # EMA type: 'regular', 'tema', 'hma'
    'stab_use_crop': False,     # Warp mode (better for panning)
    'stab_lowpass_alpha': 0,   # Low-pass strength OFF
    'stab_crop_margin': 10,      # Crop margin percentage (5-15%)
    'jpeg_quality': 75,    # Lower quality = faster encoding, minimal visual difference
    'capture_fps': 30,
}
settings_lock = threading.Lock()

# Stabilization state
stab_prev_gray = None
stab_accumulated_x = 0.0
stab_accumulated_y = 0.0
stab_smooth_correction_x = 0.0
stab_smooth_correction_y = 0.0
stab_frame_buffer = []

# Low-pass filter state
stab_filtered_dx = 0.0
stab_filtered_dy = 0.0

# Kalman filter state
stab_kalman_x = None
stab_kalman_y = None

# EMA state
stab_ema_frame = None
# TEMA state (Triple Exponential Moving Average)
stab_tema_ema1 = None
stab_tema_ema2 = None
stab_tema_ema3 = None
# HMA state (Hull Moving Average)
stab_hma_short = None
stab_hma_long = None

# AI Super-Resolution state
ai_upscaler = None
ai_model_loaded = False

def init_ai_upscaler():
    global ai_upscaler, ai_model_loaded
    try:
        import urllib.request
        import os

        # Model file path
        model_dir = os.path.expanduser('~/.cache/opencv_superres')
        os.makedirs(model_dir, exist_ok=True)

        # Try FSRCNN first (best balance of speed and quality for real-time), then fall back to ESPCN
        # EDSR is too slow for real-time microscope viewing
        models_to_try = [
            {
                'name': 'fsrcnn',
                'scale': 2,
                'file': 'FSRCNN_x2.pb',
                'url': 'https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x2.pb'
            },
            {
                'name': 'espcn',
                'scale': 2,
                'file': 'ESPCN_x2.pb',
                'url': 'https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x2.pb'
            }
        ]

        model_loaded = False
        for model_info in models_to_try:
            model_path = os.path.join(model_dir, model_info['file'])

            try:
                # Download model if it doesn't exist
                if not os.path.exists(model_path):
                    print(f"Downloading {model_info['name'].upper()} {model_info['scale']}x super-resolution model...")
                    urllib.request.urlretrieve(model_info['url'], model_path)
                    print(f"Model downloaded to {model_path}")

                # Initialize and load model
                ai_upscaler = cv2.dnn_superres.DnnSuperResImpl_create()
                ai_upscaler.readModel(model_path)
                ai_upscaler.setModel(model_info['name'], model_info['scale'])
                ai_model_loaded = True
                print(f"AI super-resolution model loaded: {model_info['name'].upper()} {model_info['scale']}x (excellent quality!)")
                model_loaded = True
                break
            except Exception as e:
                print(f"Failed to load {model_info['name'].upper()}: {e}")
                continue

        if not model_loaded:
            raise Exception("All AI upscaling models failed to load")

    except Exception as e:
        print(f"Failed to load AI upscaler: {e}")
        print("AI upscaling will not be available. Using Lanczos instead.")
        ai_upscaler = None
        ai_model_loaded = False

# Initialize AI upscaler on startup
init_ai_upscaler()

def apply_stabilization(frame, s):
    global stab_prev_gray, stab_accumulated_x, stab_accumulated_y
    global stab_smooth_correction_x, stab_smooth_correction_y, stab_frame_buffer
    global stab_filtered_dx, stab_filtered_dy
    global stab_kalman_x, stab_kalman_y, stab_ema_frame
    global stab_tema_ema1, stab_tema_ema2, stab_tema_ema3
    global stab_hma_short, stab_hma_long

    if not s['stabilize']:
        stab_prev_gray = None
        stab_accumulated_x = 0.0
        stab_accumulated_y = 0.0
        stab_smooth_correction_x = 0.0
        stab_smooth_correction_y = 0.0
        stab_frame_buffer = []
        stab_filtered_dx = 0.0
        stab_filtered_dy = 0.0
        stab_kalman_x = None
        stab_kalman_y = None
        stab_ema_frame = None
        stab_tema_ema1 = None
        stab_tema_ema2 = None
        stab_tema_ema3 = None
        stab_hma_short = None
        stab_hma_long = None
        return frame

    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_small = cv2.resize(gray, (int(w/2), int(h/2)))

    if stab_prev_gray is None or stab_prev_gray.shape != gray_small.shape:
        stab_prev_gray = gray_small.copy()
        stab_frame_buffer = []
        stab_filtered_dx = 0.0
        stab_filtered_dy = 0.0
        stab_kalman_x = None
        stab_kalman_y = None
        stab_ema_frame = None
        return frame

    try:
        # Detect motion using phase correlation
        shift, _ = cv2.phaseCorrelate(stab_prev_gray.astype(np.float32), gray_small.astype(np.float32))
        dx, dy = shift
        dx *= 2
        dy *= 2

        stab_prev_gray = gray_small.copy()

        # Apply noise threshold
        noise_threshold = s['stab_noise'] / 10.0
        if abs(dx) < noise_threshold:
            dx = 0
        if abs(dy) < noise_threshold:
            dy = 0

        # Low-pass filter on motion (smooth out high-frequency jitter)
        if s['stab_use_lowpass'] and s['stab_lowpass_alpha'] > 0:
            # Higher alpha = MORE filtering (keeps more history)
            # At 95, this will create VERY obvious lag
            alpha = (s['stab_lowpass_alpha'] / 100.0) ** 0.3  # Even more aggressive curve
            stab_filtered_dx = alpha * stab_filtered_dx + (1 - alpha) * dx
            stab_filtered_dy = alpha * stab_filtered_dy + (1 - alpha) * dy
            dx = stab_filtered_dx
            dy = stab_filtered_dy

            # Debug: Show how much filtering is happening
            if abs(dx) > 1 or abs(dy) > 1:
                raw_motion = (dx / (1 - alpha) if alpha < 0.99 else dx)
                filtering_ratio = (dx / raw_motion * 100) if raw_motion != 0 else 100
                # This shows how much motion is being filtered out

        # Kalman filter for periodic motion prediction
        if s['stab_use_kalman']:
            # Initialize Kalman filters if needed (simple 2-state: position, velocity)
            if stab_kalman_x is None:
                stab_kalman_x = cv2.KalmanFilter(2, 1)  # 2 state vars (position, velocity), 1 measurement (position)
                stab_kalman_y = cv2.KalmanFilter(2, 1)

                # State transition matrix: x_new = x + v*dt, v_new = v (constant velocity model)
                dt = 1.0
                stab_kalman_x.transitionMatrix = np.array([[1, dt], [0, 1]], dtype=np.float32)
                stab_kalman_y.transitionMatrix = stab_kalman_x.transitionMatrix.copy()

                # Measurement matrix: we only measure position
                stab_kalman_x.measurementMatrix = np.array([[1, 0]], dtype=np.float32)
                stab_kalman_y.measurementMatrix = stab_kalman_x.measurementMatrix.copy()

                # Process noise (how much we trust the model) - lower = trust model more
                stab_kalman_x.processNoiseCov = np.eye(2, dtype=np.float32) * 0.001
                stab_kalman_y.processNoiseCov = np.eye(2, dtype=np.float32) * 0.001

                # Measurement noise (how much we trust the measurements) - higher = smooth more
                stab_kalman_x.measurementNoiseCov = np.array([[1.0]], dtype=np.float32)
                stab_kalman_y.measurementNoiseCov = np.array([[1.0]], dtype=np.float32)

            # Predict next state
            stab_kalman_x.predict()
            stab_kalman_y.predict()

            # Correct with measurement
            measurement_x = np.array([[dx]], dtype=np.float32)
            measurement_y = np.array([[dy]], dtype=np.float32)

            stab_kalman_x.correct(measurement_x)
            stab_kalman_y.correct(measurement_y)

            # Use filtered position (smoother than raw measurement)
            dx = stab_kalman_x.statePost[0, 0]
            dy = stab_kalman_y.statePost[0, 0]

        max_shift = min(w, h) * 0.3
        if abs(dx) < max_shift and abs(dy) < max_shift:
            # Use traditional accumulation and smoothing for both modes
            # This provides better vibration filtering than direct tracking
            stab_accumulated_x += dx
            stab_accumulated_y += dy
            stab_accumulated_x *= s['stab_decay'] / 100.0
            stab_accumulated_y *= s['stab_decay'] / 100.0

            max_accum = min(w, h) * 0.5
            stab_accumulated_x = max(-max_accum, min(max_accum, stab_accumulated_x))
            stab_accumulated_y = max(-max_accum, min(max_accum, stab_accumulated_y))

            smooth = s['stab_smooth'] / 100.0
            stab_smooth_correction_x = smooth * stab_smooth_correction_x + (1 - smooth) * (-stab_accumulated_x)
            stab_smooth_correction_y = smooth * stab_smooth_correction_y + (1 - smooth) * (-stab_accumulated_y)

            # Crop-based stabilization (no motion blur!)
            if s['stab_use_crop']:
                # Calculate crop size based on margin percentage
                margin_pct = s['stab_crop_margin'] / 100.0
                crop_w = int(w * (1 - margin_pct * 2))
                crop_h = int(h * (1 - margin_pct * 2))

                # Calculate the center of the crop window
                # Start from the frame center
                center_x = w // 2
                center_y = h // 2

                # Offset by the correction (negate to move crop window opposite of shake)
                # If camera shakes RIGHT, we need to crop from the RIGHT side
                # Keep as float for sub-pixel accuracy
                offset_x = -stab_smooth_correction_x
                offset_y = -stab_smooth_correction_y

                # Calculate crop boundaries with float precision
                crop_x1_float = center_x - crop_w / 2.0 + offset_x
                crop_y1_float = center_y - crop_h / 2.0 + offset_y
                crop_x2_float = crop_x1_float + crop_w
                crop_y2_float = crop_y1_float + crop_h

                # Clamp to frame boundaries
                crop_x1_float = max(0, min(w - crop_w, crop_x1_float))
                crop_y1_float = max(0, min(h - crop_h, crop_y1_float))
                crop_x2_float = crop_x1_float + crop_w
                crop_y2_float = crop_y1_float + crop_h

                # Convert to int only for final crop extraction
                crop_x1 = int(crop_x1_float)
                crop_y1 = int(crop_y1_float)
                crop_x2 = int(crop_x2_float)
                crop_y2 = int(crop_y2_float)

                # Extract crop and resize back to original size
                cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                stabilized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                # Traditional warp-based stabilization
                M = np.float32([[1, 0, stab_smooth_correction_x], [0, 1, stab_smooth_correction_y]])
                stabilized = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        else:
            stab_accumulated_x = 0.0
            stab_accumulated_y = 0.0
            stab_smooth_correction_x = 0.0
            stab_smooth_correction_y = 0.0
            stab_frame_buffer = []
            stabilized = frame.copy()
    except:
        stabilized = frame.copy()

    # Temporal blending (only for non-crop modes)
    if not s['stab_use_crop'] and s['stab_use_ema']:
        ema_type = s['stab_ema_type']

        if ema_type == 'regular':
            # Regular Exponential Moving Average (lower latency but very smooth)
            # Lower alpha = MORE smoothing (more ghosting/trails but smoother)
            alpha = 1.0 / (s['stab_blend'] * 2)  # More aggressive than standard EMA
            if stab_ema_frame is None:
                stab_ema_frame = stabilized.astype(np.float32)
            else:
                stab_ema_frame = alpha * stabilized.astype(np.float32) + (1 - alpha) * stab_ema_frame
            stabilized = stab_ema_frame.astype(np.uint8)

        elif ema_type == 'tema':
            # Triple Exponential Moving Average (reduced lag, more responsive)
            # TEMA = 3*EMA1 - 3*EMA2 + EMA3
            alpha = 1.0 / (s['stab_blend'] * 2)
            current_frame = stabilized.astype(np.float32)

            if stab_tema_ema1 is None:
                stab_tema_ema1 = current_frame
                stab_tema_ema2 = current_frame
                stab_tema_ema3 = current_frame
            else:
                stab_tema_ema1 = alpha * current_frame + (1 - alpha) * stab_tema_ema1
                stab_tema_ema2 = alpha * stab_tema_ema1 + (1 - alpha) * stab_tema_ema2
                stab_tema_ema3 = alpha * stab_tema_ema2 + (1 - alpha) * stab_tema_ema3

            tema_result = 3 * stab_tema_ema1 - 3 * stab_tema_ema2 + stab_tema_ema3
            stabilized = np.clip(tema_result, 0, 255).astype(np.uint8)

        elif ema_type == 'hma':
            # Hull Moving Average (very responsive while smooth)
            # HMA approximation: 2*EMA(short) - EMA(long)
            alpha_short = 1.0 / (s['stab_blend'] * 1.5)  # Faster response
            alpha_long = 1.0 / (s['stab_blend'] * 3)  # Slower response
            current_frame = stabilized.astype(np.float32)

            if stab_hma_short is None:
                stab_hma_short = current_frame
                stab_hma_long = current_frame
            else:
                stab_hma_short = alpha_short * current_frame + (1 - alpha_short) * stab_hma_short
                stab_hma_long = alpha_long * current_frame + (1 - alpha_long) * stab_hma_long

            hma_result = 2 * stab_hma_short - stab_hma_long
            stabilized = np.clip(hma_result, 0, 255).astype(np.uint8)
    elif not s['stab_use_crop']:
        # Weighted frame blending (original method)
        blend = s['stab_blend']
        if blend > 1:
            stab_frame_buffer.append(stabilized.astype(np.float32))
            if len(stab_frame_buffer) > blend:
                stab_frame_buffer.pop(0)
            if len(stab_frame_buffer) >= 2:
                blended = np.zeros_like(stabilized, dtype=np.float32)
                for i, f in enumerate(stab_frame_buffer):
                    blended += f * (i + 1)
                stabilized = (blended / sum(range(1, len(stab_frame_buffer) + 1))).astype(np.uint8)

    return stabilized

def apply_processing(frame, s):
    global stab_prev_gray, stab_accumulated_x, stab_accumulated_y
    global stab_smooth_correction_x, stab_smooth_correction_y, stab_frame_buffer
    global ai_upscaler, ai_model_loaded

    p = frame

    # Combine gain with brightness/contrast for single operation
    if s['gain'] != 1.0 or s['brightness'] != 0 or s['contrast'] != 1.0:
        combined_alpha = s['contrast'] * s['gain']
        p = cv2.convertScaleAbs(p, alpha=combined_alpha, beta=s['brightness'])

    # Skip expensive saturation if at default
    if s['saturation'] != 1.0:
        hsv = cv2.cvtColor(p, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s['saturation'], 0, 255)
        p = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Fast flip operations
    if s['flip_h'] and s['flip_v']:
        p = cv2.flip(p, -1)
    elif s['flip_h']:
        p = cv2.flip(p, 1)
    elif s['flip_v']:
        p = cv2.flip(p, 0)

    # Skip rotation if at 0
    r = int(s['rotate']) % 360
    if r != 0:
        center = (p.shape[1] // 2, p.shape[0] // 2)
        M = cv2.getRotationMatrix2D(center, r, 1.0)
        p = cv2.warpAffine(p, M, (p.shape[1], p.shape[0]), borderMode=cv2.BORDER_REPLICATE)

    # Skip zoom if at 1.0
    z = s['zoom']
    if z != 1.0:
        h, w = p.shape[:2]
        if z > 1.0:
            ch = int(h / z)
            cw = int(w / z)
            sy = (h - ch) // 2
            sx = (w - cw) // 2
            cropped = p[sy:sy+ch, sx:sx+cw]

            # Choose upscaling method based on settings
            if s['use_ai_upscale'] and ai_model_loaded:
                try:
                    # AI super-resolution (best quality, slowest)
                    # ESPCN 2x upscaling - very slow but excellent quality
                    upscaled = ai_upscaler.upsample(cropped)
                    # Resize to final dimensions if needed (ESPCN does 2x, we may need different)
                    if upscaled.shape[:2] != (h, w):
                        p = cv2.resize(upscaled, (w, h), interpolation=cv2.INTER_LANCZOS4)
                    else:
                        p = upscaled
                except Exception as e:
                    # Fall back to Lanczos if AI upscaling fails
                    print(f"AI upscaling failed: {e}, falling back to Lanczos")
                    p = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LANCZOS4)
            elif s['use_lanczos']:
                # High-quality Lanczos interpolation (good quality, fast)
                p = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LANCZOS4)
            else:
                # Basic linear interpolation (fastest)
                p = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            nh = int(h * z)
            nw = int(w * z)
            resized = cv2.resize(p, (nw, nh), interpolation=cv2.INTER_AREA)
            p = np.zeros((h, w, 3), dtype=np.uint8)
            sy = (h - nh) // 2
            sx = (w - nw) // 2
            p[sy:sy+nh, sx:sx+nw] = resized

    return apply_stabilization(p, s)

def capture_loop():
    global current_frame, running
    
    dev = usb.core.find(idVendor=0x1b3f, idProduct=0x2002)
    if not dev:
        print("Microscope not found (VID: 1b3f, PID: 2002)")
        running = False
        return
    
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    # Camera ignores resolution settings and always outputs 1280x720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)  # Request higher FPS (camera will use what it supports)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    time.sleep(1)

    ret, frame = cap.read()
    if not ret or frame.shape[1] != 1280 or frame.shape[0] != 720:
        print("Failed to read from device")
        cap.release()
        running = False
        return
    
    print(f"Web viewer running at: http://localhost:{WEB_PORT}")
    print("Press Ctrl+C to stop\n")
    
    frame_count = 0
    last_time = time.time()
    cached_settings = None
    settings_check_count = 0
    consecutive_failures = 0
    
    while running:
        if frame_count > 0:
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures > 10:
                    print("Camera became stale")
                    cap.release()
                    running = False
                    return
                time.sleep(0.1)
                continue

        consecutive_failures = 0
        settings_check_count += 1
        if settings_check_count >= 10:
            with settings_lock:
                cached_settings = settings.copy()
            settings_check_count = 0
        
        if cached_settings is None:
            with settings_lock:
                cached_settings = settings.copy()
        
        frame_count += 1
        if frame_count % 30 == 0:
            stab_status = "OFF"
            if cached_settings['stabilize']:
                methods = []
                if cached_settings['stab_use_crop']:
                    methods.append(f"Crop:{cached_settings['stab_crop_margin']}%")
                else:
                    if cached_settings['stab_use_lowpass']:
                        methods.append(f"LP:{cached_settings['stab_lowpass_alpha']}")
                    if cached_settings['stab_use_kalman']:
                        methods.append("Kalman")
                    if cached_settings['stab_use_ema']:
                        methods.append("EMA")
                stab_status = f"ON [{','.join(methods) if methods else 'Basic'}]"
            print(f"Frame {frame_count}: Stab={stab_status}")
        processed = apply_processing(frame, cached_settings)
        ret, jpeg = cv2.imencode('.jpg', processed, [cv2.IMWRITE_JPEG_QUALITY, cached_settings['jpeg_quality']])
        if ret:
            with frame_lock:
                current_frame = jpeg.tobytes()
        
        if frame_count % 30 == 0:
            fps = 30 / (time.time() - last_time)
            print(f"Frame {frame_count}: {fps:.1f} fps")
            last_time = time.time()

class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            with open('/Users/x334478/personal/microscope-viewer/index.html', 'rb') as f:
                self.wfile.write(f.read())
        
        elif self.path.startswith('/stream.mjpg'):
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            
            last_frame = None
            try:
                while running:
                    with frame_lock:
                        frame = current_frame
                    
                    if frame and frame != last_frame:
                        self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n')
                        self.wfile.write(f'Content-Length: {len(frame)}\r\n\r\n'.encode())
                        self.wfile.write(frame)
                        self.wfile.write(b'\r\n')
                        last_frame = frame
                    
                    time.sleep(0.001)
            except:
                pass
        
        elif self.path == '/current.jpg':
            with frame_lock:
                if current_frame:
                    self.send_response(200)
                    self.send_header('Content-type', 'image/jpeg')
                    self.end_headers()
                    self.wfile.write(current_frame)
                else:
                    self.send_response(404)
                    self.end_headers()

        elif self.path == '/settings':
            # Return current settings as JSON
            with settings_lock:
                settings_json = json.dumps(settings)
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(settings_json.encode())

        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        if self.path.startswith('/process/'):
            parts = self.path.split('/')
            if len(parts) >= 3:
                setting = parts[2]
                value = parts[3] if len(parts) > 3 else ''
                
                with settings_lock:
                    if setting == 'reset':
                        settings.update({
                            'brightness': 0, 'contrast': 1.0, 'saturation': 1.0,
                            'gain': 1.0, 'flip_h': True, 'flip_v': True,
                            'rotate': 0, 'zoom': 1.0
                        })
                    elif setting == 'brightness':
                        settings['brightness'] = int(value)
                    elif setting == 'contrast':
                        settings['contrast'] = int(value) / 100.0
                    elif setting == 'saturation':
                        settings['saturation'] = int(value) / 100.0
                    elif setting == 'gain':
                        settings['gain'] = int(value) / 100.0
                    elif setting == 'zoom':
                        settings['zoom'] = int(value) / 100.0
                    elif setting == 'flip_h' and value == 'toggle':
                        settings['flip_h'] = not settings['flip_h']
                    elif setting == 'flip_v' and value == 'toggle':
                        settings['flip_v'] = not settings['flip_v']
                    elif setting == 'rotate':
                        new_rotate = int(value) % 360
                        print(f"Setting rotate to {new_rotate}")
                        settings['rotate'] = new_rotate
                    elif setting == 'stabilize' and value == 'toggle':
                        settings['stabilize'] = not settings['stabilize']
                    elif setting == 'stab_noise':
                        settings['stab_noise'] = int(value)
                    elif setting == 'stab_smooth':
                        settings['stab_smooth'] = int(value)
                    elif setting == 'stab_decay':
                        settings['stab_decay'] = int(value)
                    elif setting == 'stab_blend':
                        settings['stab_blend'] = int(value)
                    elif setting == 'stab_reset':
                        settings.update({
                            'stab_noise': 0, 'stab_smooth': 0,
                            'stab_decay': 0, 'stab_blend': 3,
                            'stab_lowpass_alpha': 0
                        })
                    elif setting == 'stab_use_lowpass' and value == 'toggle':
                        settings['stab_use_lowpass'] = not settings['stab_use_lowpass']
                    elif setting == 'stab_use_kalman' and value == 'toggle':
                        settings['stab_use_kalman'] = not settings['stab_use_kalman']
                    elif setting == 'stab_use_ema' and value == 'toggle':
                        settings['stab_use_ema'] = not settings['stab_use_ema']
                    elif setting == 'stab_use_crop' and value == 'toggle':
                        settings['stab_use_crop'] = not settings['stab_use_crop']
                    elif setting == 'stab_lowpass_alpha':
                        settings['stab_lowpass_alpha'] = int(value)
                    elif setting == 'stab_crop_margin':
                        settings['stab_crop_margin'] = int(value)
                    elif setting == 'stab_ema_type':
                        if value in ['regular', 'tema', 'hma']:
                            settings['stab_ema_type'] = value
                    elif setting == 'use_lanczos' and value == 'toggle':
                        settings['use_lanczos'] = not settings['use_lanczos']
                    elif setting == 'use_ai_upscale' and value == 'toggle':
                        settings['use_ai_upscale'] = not settings['use_ai_upscale']
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')
        
        elif self.path.startswith('/capture/'):
            parts = self.path.split('/')
            if len(parts) == 4:
                setting = parts[2]
                value = parts[3]
                with settings_lock:
                    if setting == 'quality':
                        settings['jpeg_quality'] = int(value)
                    elif setting == 'fps':
                        settings['capture_fps'] = int(value)
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')
        
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, fmt, *args):
        pass

def main():
    global running
    print("="*50)
    print("Microscope Viewer (Fast)")
    print("="*50)
    
    capture_thread = threading.Thread(target=capture_loop, daemon=True)
    capture_thread.start()
    
    time.sleep(1)
    
    server = ThreadingHTTPServer(('', WEB_PORT), Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    
    try:
        while running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        running = False
        server.shutdown()

if __name__ == "__main__":
    main()
