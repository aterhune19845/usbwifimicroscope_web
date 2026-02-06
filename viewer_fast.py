#!/usr/bin/env python3
import time
import threading
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import cv2
import numpy as np
import usb.core
import usb.util
import base64
import os

# Browser automation for Claude.ai
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
    print("✓ Playwright available for browser automation")
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("✗ Warning: playwright not available. Component analysis disabled.")
    print("  Install with: pip install playwright && playwright install chromium")

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
    # PCB/Circuit Board Enhancement
    'enhance_clahe': True,      # CLAHE contrast enhancement (ON by default - helps tracking!)
    'enhance_edges': False,     # Edge detection overlay
    'enhance_sharpen': True,    # Sharpening filter (ON by default for PCB inspection)
    'enhance_invert': False,    # Color inversion
    # Annotation Tracking
    'track_annotations': True,  # Track annotations with image motion
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

# Screenshot directory for browser automation
screenshot_dir = os.path.join(os.path.dirname(__file__), 'screenshots')
os.makedirs(screenshot_dir, exist_ok=True)

# Browser automation state - keep browser alive across requests (Claude)
playwright_instance = None
browser_instance = None
page_instance = None
browser_lock = threading.Lock()

# Gemini browser automation state
gemini_playwright_instance = None
gemini_browser_instance = None
gemini_page_instance = None
gemini_browser_lock = threading.Lock()

# Annotation tracking state
current_motion_matrix = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]  # Affine matrix [a, b, tx, c, d, ty]
motion_sequence = 0  # Increments each time motion is updated
motion_lock = threading.Lock()
tracking_prev_gray = None  # Separate from stabilization for different resolution
tracking_points = None  # Feature points for optical flow tracking
motion_history = []  # Smoothing history
motion_velocity = (0.0, 0.0)  # Velocity for prediction
motion_ema_matrix = None  # EMA-smoothed motion (syncs with stabilization EMA)

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
                    urllib.request.urlretrieve(model_info['url'], model_path)

                # Initialize and load model
                ai_upscaler = cv2.dnn_superres.DnnSuperResImpl_create()
                ai_upscaler.readModel(model_path)
                ai_upscaler.setModel(model_info['name'], model_info['scale'])
                ai_model_loaded = True
                print(f"✓ AI super-resolution loaded: {model_info['name'].upper()} {model_info['scale']}x")
                model_loaded = True
                break
            except Exception:
                continue

        # Silently fall back to Lanczos if AI upscaler unavailable

    except Exception:
        pass  # Silently use Lanczos instead

    # Set defaults if loading failed
    if not ai_model_loaded:
        ai_upscaler = None
        ai_model_loaded = False

# Initialize AI upscaler on startup
init_ai_upscaler()

def init_browser_automation():
    """Check if browser automation is available"""
    if PLAYWRIGHT_AVAILABLE:
        print("✓ Browser automation ready for Claude.ai")
    else:
        print("✗ Browser automation not available")
        print("  Install with: pip install playwright && playwright install chromium")

# Initialize browser automation on startup
print("\n=== Initializing Browser Automation ===")
init_browser_automation()
print("========================================\n")

def apply_stabilization(frame, s):
    global stab_prev_gray, stab_accumulated_x, stab_accumulated_y
    global stab_smooth_correction_x, stab_smooth_correction_y, stab_frame_buffer
    global stab_filtered_dx, stab_filtered_dy
    global stab_kalman_x, stab_kalman_y, stab_ema_frame
    global stab_tema_ema1, stab_tema_ema2, stab_tema_ema3
    global stab_hma_short, stab_hma_long
    global current_motion_matrix, motion_sequence, motion_lock, tracking_prev_gray, tracking_points, motion_history, motion_velocity, motion_ema_matrix

    # Always track motion for annotations using optical flow (even when stabilization is off)
    # Optical flow handles fast jerky movements much better than phase correlation
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if s['track_annotations']:
        if tracking_prev_gray is not None and tracking_prev_gray.shape == gray.shape:
            try:
                # If we don't have tracking points or too few remain, detect new ones
                if tracking_points is None or len(tracking_points) < 30:
                    # Detect good features distributed across the image
                    # More features = more robust tracking
                    tracking_points = cv2.goodFeaturesToTrack(
                        tracking_prev_gray,
                        maxCorners=200,  # Track more features
                        qualityLevel=0.01,
                        minDistance=20,  # Better distribution
                        blockSize=7,
                        useHarrisDetector=True,
                        k=0.04
                    )

                if tracking_points is not None and len(tracking_points) > 0:
                    # Use motion prediction to help with very fast movements
                    # Predict where features should be based on velocity
                    predicted_points = tracking_points.copy()
                    if motion_velocity[0] != 0 or motion_velocity[1] != 0:
                        predicted_points[:, 0, 0] += motion_velocity[0]
                        predicted_points[:, 0, 1] += motion_velocity[1]

                    # Track features using Lucas-Kanade optical flow
                    # OPTIMIZED FOR 24FPS: Larger window and more pyramid levels
                    new_points, status, err = cv2.calcOpticalFlowPyrLK(
                        tracking_prev_gray,
                        gray,
                        tracking_points,
                        predicted_points,  # Use predicted positions as initial guess
                        winSize=(71, 71),  # Massive search window for 24fps large displacements
                        maxLevel=6,  # 6 pyramid levels = can handle ~64x motions (up to 300+ pixels)
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.005),
                        flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,
                        minEigThreshold=0.00005  # Very permissive for extreme motion
                    )

                    # Select good points (successfully tracked)
                    if new_points is not None and status is not None:
                        good_old = tracking_points[status == 1]
                        good_new = new_points[status == 1]

                        if len(good_new) > 10:  # Need at least 10 points for reliable motion
                            # Use RANSAC to estimate rigid transform and reject outliers
                            # This handles rotation/scale and is more robust than median
                            try:
                                # Estimate affine transform using RANSAC
                                M, inliers = cv2.estimateAffinePartial2D(
                                    good_old, good_new,
                                    method=cv2.RANSAC,
                                    ransacReprojThreshold=3.0,
                                    maxIters=2000,
                                    confidence=0.99
                                )

                                if M is not None and inliers is not None and np.sum(inliers) > 5:
                                    # Store full affine matrix: [a, b, tx, c, d, ty]
                                    # where [[a, b, tx], [c, d, ty]] is the 2x3 affine matrix
                                    motion_matrix = [
                                        float(M[0, 0]), float(M[0, 1]), float(M[0, 2]),
                                        float(M[1, 0]), float(M[1, 1]), float(M[1, 2])
                                    ]
                                    # Keep only inlier points for next frame
                                    tracking_points = good_new[inliers.ravel() == 1].reshape(-1, 1, 2)
                                else:
                                    # RANSAC failed, fall back to median (pure translation)
                                    motion_vectors = good_new - good_old
                                    dx = float(np.median(motion_vectors[:, 0]))
                                    dy = float(np.median(motion_vectors[:, 1]))
                                    # Identity matrix with translation
                                    motion_matrix = [1.0, 0.0, dx, 0.0, 1.0, dy]
                                    tracking_points = good_new.reshape(-1, 1, 2)
                            except:
                                # RANSAC error, use median (pure translation)
                                motion_vectors = good_new - good_old
                                dx = float(np.median(motion_vectors[:, 0]))
                                dy = float(np.median(motion_vectors[:, 1]))
                                # Identity matrix with translation
                                motion_matrix = [1.0, 0.0, dx, 0.0, 1.0, dy]
                                tracking_points = good_new.reshape(-1, 1, 2)

                            # Apply temporal smoothing to reduce jitter
                            motion_history.append(motion_matrix)
                            if len(motion_history) > 3:  # Keep last 3 frames
                                motion_history.pop(0)

                            # Weighted average of matrix elements: more recent = higher weight
                            if len(motion_history) >= 2:
                                weights = [1, 2, 3][-len(motion_history):]
                                total_weight = sum(weights)
                                smooth_matrix = [
                                    sum(w * m[i] for w, m in zip(weights, motion_history)) / total_weight
                                    for i in range(6)
                                ]
                            else:
                                smooth_matrix = motion_matrix

                            # Update velocity for motion prediction (use translation components)
                            motion_velocity = (smooth_matrix[2], smooth_matrix[5])

                            # Apply additional EMA smoothing if stabilization is ON with EMA
                            # This syncs annotation movement with the stabilized image smoothing
                            global motion_ema_matrix
                            if s['stabilize'] and not s['stab_use_crop'] and s['stab_use_ema']:
                                # Use same alpha as stabilization EMA for perfect sync
                                alpha = 1.0 / (s['stab_blend'] * 2)
                                if motion_ema_matrix is None:
                                    motion_ema_matrix = smooth_matrix[:]
                                else:
                                    # Apply EMA to each matrix component
                                    motion_ema_matrix = [
                                        alpha * smooth_matrix[i] + (1 - alpha) * motion_ema_matrix[i]
                                        for i in range(6)
                                    ]
                                final_matrix = motion_ema_matrix
                            else:
                                # Use normal weighted-average smoothing
                                motion_ema_matrix = None
                                final_matrix = smooth_matrix

                            with motion_lock:
                                current_motion_matrix = final_matrix
                                motion_sequence += 1
                        else:
                            # Not enough good points, reset and try again next frame
                            tracking_points = None
                            motion_history.clear()
                            motion_velocity = (0.0, 0.0)
                            motion_ema_matrix = None
                            with motion_lock:
                                current_motion_matrix = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
                                motion_sequence += 1
                    else:
                        tracking_points = None
                        motion_history.clear()
                        motion_velocity = (0.0, 0.0)
                        motion_ema_matrix = None
                        with motion_lock:
                            current_motion_matrix = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
                            motion_sequence += 1
                else:
                    motion_history.clear()
                    motion_velocity = (0.0, 0.0)
                    motion_ema_matrix = None
                    with motion_lock:
                        current_motion_matrix = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
                        motion_sequence += 1
            except:
                tracking_points = None
                motion_history.clear()
                motion_velocity = (0.0, 0.0)
                motion_ema_matrix = None
                with motion_lock:
                    current_motion_matrix = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
                    motion_sequence += 1
        else:
            tracking_points = None
            motion_history.clear()
            motion_velocity = (0.0, 0.0)
            with motion_lock:
                current_motion_matrix = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
                motion_sequence += 1

        # Update tracking reference frame
        tracking_prev_gray = gray.copy()
    else:
        tracking_prev_gray = None
        tracking_points = None
        motion_history.clear()
        motion_velocity = (0.0, 0.0)
        motion_ema_matrix = None
        with motion_lock:
            current_motion_matrix = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            motion_sequence += 1

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
        stabilized = frame

    if s['stabilize']:
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

    # Apply PCB/Circuit Board enhancements
    enhanced = stabilized

    # CLAHE - Contrast Limited Adaptive Histogram Equalization (great for local details)
    if s['enhance_clahe']:
        # Convert to LAB color space for better contrast enhancement
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])  # Apply only to L channel
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Sharpening - Makes fine details (traces, vias) more visible
    if s['enhance_sharpen']:
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        enhanced = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)

    # Edge detection overlay - Highlights component boundaries and traces
    if s['enhance_edges']:
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        # Create colored edge overlay (cyan edges)
        edge_overlay = np.zeros_like(enhanced)
        edge_overlay[edges > 0] = [255, 255, 0]  # Cyan color for edges
        enhanced = cv2.addWeighted(enhanced, 0.85, edge_overlay, 0.15, 0)

    # Color inversion - Sometimes easier to see copper traces on dark background
    if s['enhance_invert']:
        enhanced = cv2.bitwise_not(enhanced)

    return enhanced

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

        elif self.path == '/motion':
            # Return full affine transformation matrix for annotation tracking
            with motion_lock:
                motion_data = json.dumps({
                    'matrix': current_motion_matrix,  # [a, b, tx, c, d, ty]
                    'seq': motion_sequence
                })
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(motion_data.encode())

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
                    elif setting == 'enhance_clahe' and value == 'toggle':
                        settings['enhance_clahe'] = not settings['enhance_clahe']
                    elif setting == 'enhance_edges' and value == 'toggle':
                        settings['enhance_edges'] = not settings['enhance_edges']
                    elif setting == 'enhance_sharpen' and value == 'toggle':
                        settings['enhance_sharpen'] = not settings['enhance_sharpen']
                    elif setting == 'enhance_invert' and value == 'toggle':
                        settings['enhance_invert'] = not settings['enhance_invert']

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

        elif self.path == '/analyze_component':
            # Save screenshot and open Claude.ai with browser automation
            if not PLAYWRIGHT_AVAILABLE:
                self.send_response(503)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    'error': 'Playwright not available. Install with: pip install playwright && playwright install chromium'
                }).encode())
                return

            # Read POST body
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            data = json.loads(body)

            screenshot_base64 = data.get('screenshot')  # Base64-encoded PNG

            if not screenshot_base64:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'No screenshot provided'}).encode())
                return

            try:
                # Save screenshot to file
                import tempfile
                timestamp = int(time.time())
                screenshot_path = os.path.join(screenshot_dir, f'component_{timestamp}.png')

                # Decode base64 and save
                screenshot_data = base64.b64decode(screenshot_base64)
                with open(screenshot_path, 'wb') as f:
                    f.write(screenshot_data)

                print(f"📸 Screenshot saved: {screenshot_path}")

                # Launch browser automation in a separate thread to not block
                def automate_claude():
                    global playwright_instance, browser_instance, page_instance

                    try:
                        with browser_lock:
                            # Check if browser context is still alive
                            reuse_browser = False
                            if browser_instance is not None:
                                try:
                                    # Check if context is still alive
                                    existing_pages = browser_instance.pages
                                    if existing_pages:
                                        # Reuse existing page or create new one
                                        page_instance = existing_pages[0]
                                        print("♻️  Reusing existing browser, navigating to new chat...")
                                        page_instance.goto('https://claude.ai/new', wait_until='domcontentloaded')
                                        time.sleep(2)
                                        reuse_browser = True
                                except:
                                    # Browser died or not responding, force cleanup
                                    print("🔄 Browser not responding, forcing cleanup...")
                                    try:
                                        if browser_instance:
                                            browser_instance.close()
                                        if playwright_instance:
                                            playwright_instance.stop()
                                    except:
                                        pass

                                    # Force kill any chromium processes using our profile
                                    try:
                                        import subprocess
                                        subprocess.run(['pkill', '-f', 'chromium.*browser_profile'],
                                                     stdout=subprocess.DEVNULL,
                                                     stderr=subprocess.DEVNULL)
                                        time.sleep(1)  # Wait for process to die
                                    except:
                                        pass

                                    playwright_instance = None
                                    browser_instance = None
                                    page_instance = None

                            if not reuse_browser:
                                # Start new browser with persistent profile to avoid CAPTCHA
                                playwright_instance = sync_playwright().start()

                                # Use persistent context to save cookies/login
                                user_data_dir = os.path.join(os.path.dirname(__file__), '.browser_profile')

                                browser_instance = playwright_instance.chromium.launch_persistent_context(
                                    user_data_dir=user_data_dir,
                                    headless=False,
                                    args=['--disable-blink-features=AutomationControlled']  # Hide automation
                                )
                                page_instance = browser_instance.pages[0] if browser_instance.pages else browser_instance.new_page()

                                # Navigate to Claude.ai
                                print("🌐 Opening Claude.ai...")
                                page_instance.goto('https://claude.ai/new', wait_until='domcontentloaded')

                                # Give time for CAPTCHA/login if needed
                                print("⏳ Waiting 3 seconds for page to load...")
                                print("   (Browser will stay open - solve any challenge if needed)")
                                time.sleep(3)

                            # Upload screenshot to the (possibly existing) conversation
                            print("📎 Uploading screenshot...")
                            file_input = page_instance.locator('input[type="file"]').first
                            file_input.set_input_files(screenshot_path, timeout=10000)

                            # Wait for upload to process
                            time.sleep(2)

                            # Find the textarea and type the prompt
                            print("⌨️  Typing prompt...")
                            prompt = "I've circled a component or components on this PCB with annotations. Please identify what component(s) are circled and provide technical details including: component type, likely part designation, function, any visible markings or identifiers, typical pinouts, test procedures, and expected voltage/resistance values.  Include description of which pins are where in the picture.  Eg.  Left/Right/Top/Bottom"

                            # Wait for textarea to be ready and visible
                            textarea = page_instance.locator('div[contenteditable="true"]').first
                            textarea.wait_for(state='visible', timeout=10000)
                            textarea.click()
                            time.sleep(0.5)
                            textarea.fill(prompt)

                            # Wait a moment for the message to be ready
                            time.sleep(1)

                            # Press Enter or click send button
                            print("🚀 Sending message...")
                            textarea.press('Enter')

                            if reuse_browser:
                                print("✓ New screenshot sent to existing conversation!")
                            else:
                                print("✓ Message sent! Browser window will stay open for additional screenshots.")

                    except Exception as e:
                        print(f"✗ Browser automation error: {e}")
                        import traceback
                        traceback.print_exc()
                        # Reset on error
                        with browser_lock:
                            playwright_instance = None
                            browser_instance = None
                            page_instance = None

                # Start automation in background thread
                automation_thread = threading.Thread(target=automate_claude, daemon=True)
                automation_thread.start()

                # Return success immediately
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    'status': 'success',
                    'message': 'Opening Claude.ai in browser...',
                    'screenshot': screenshot_path
                }).encode())

            except Exception as e:
                print(f"✗ Error: {e}")
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode())

        elif self.path == '/analyze_component_gemini':
            # Save screenshot and open Google Gemini with browser automation
            if not PLAYWRIGHT_AVAILABLE:
                self.send_response(503)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    'error': 'Playwright not available'
                }).encode())
                return

            # Read POST body
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            data = json.loads(body)
            screenshot_base64 = data.get('screenshot')

            if not screenshot_base64:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'No screenshot provided'}).encode())
                return

            try:
                # Save screenshot to file
                timestamp = int(time.time())
                screenshot_path = os.path.join(screenshot_dir, f'component_{timestamp}.png')
                screenshot_data = base64.b64decode(screenshot_base64)
                with open(screenshot_path, 'wb') as f:
                    f.write(screenshot_data)
                print(f"📸 Screenshot saved: {screenshot_path}")

                # Launch browser automation in a separate thread
                def automate_gemini():
                    global gemini_playwright_instance, gemini_browser_instance, gemini_page_instance

                    try:
                        with gemini_browser_lock:
                            # Check if browser context is still alive
                            reuse_browser = False
                            if gemini_browser_instance is not None:
                                try:
                                    existing_pages = gemini_browser_instance.pages
                                    if existing_pages:
                                        gemini_page_instance = existing_pages[0]
                                        print("♻️  Reusing existing Gemini browser...")
                                        gemini_page_instance.goto('https://aistudio.google.com/prompts/new_chat', wait_until='domcontentloaded')
                                        time.sleep(2)
                                        reuse_browser = True
                                except:
                                    print("🔄 Gemini browser not responding, forcing cleanup...")
                                    try:
                                        if gemini_browser_instance:
                                            gemini_browser_instance.close()
                                        if gemini_playwright_instance:
                                            gemini_playwright_instance.stop()
                                    except:
                                        pass

                                    # Force kill any chromium processes using gemini profile
                                    try:
                                        import subprocess
                                        subprocess.run(['pkill', '-f', 'chromium.*gemini_browser_profile'],
                                                     stdout=subprocess.DEVNULL,
                                                     stderr=subprocess.DEVNULL)
                                        time.sleep(1)
                                    except:
                                        pass

                                    gemini_playwright_instance = None
                                    gemini_browser_instance = None
                                    gemini_page_instance = None

                            if not reuse_browser:
                                # Start new browser with persistent profile
                                gemini_playwright_instance = sync_playwright().start()
                                user_data_dir = os.path.join(os.path.dirname(__file__), '.gemini_browser_profile')
                                gemini_browser_instance = gemini_playwright_instance.chromium.launch_persistent_context(
                                    user_data_dir=user_data_dir,
                                    headless=False,
                                    args=['--disable-blink-features=AutomationControlled']
                                )
                                gemini_page_instance = gemini_browser_instance.pages[0] if gemini_browser_instance.pages else gemini_browser_instance.new_page()

                                print("🌐 Opening Gemini AI Studio...")
                                gemini_page_instance.goto('https://aistudio.google.com/prompts/new_chat', wait_until='domcontentloaded')
                                print("⏳ Waiting 3 seconds for page to load...")
                                time.sleep(3)

                            # Upload screenshot using clipboard paste
                            print("📎 Uploading screenshot to Gemini via clipboard...")
                            upload_success = False

                            try:
                                # Step 1: Copy image to clipboard (macOS)
                                print("   Copying image to clipboard...")
                                import subprocess
                                # Use osascript to copy PNG to clipboard
                                subprocess.run([
                                    'osascript', '-e',
                                    f'set the clipboard to (read (POSIX file "{screenshot_path}") as «class PNGf»)'
                                ], check=True, capture_output=True)
                                print("   ✓ Image copied to clipboard")

                                # Step 2: Find and click textarea
                                print("   Finding prompt textarea...")
                                textarea_selectors = ['textarea', 'div[contenteditable="true"]', '[role="textbox"]']
                                prompt_element = None

                                for selector in textarea_selectors:
                                    try:
                                        prompt_element = gemini_page_instance.locator(selector).first
                                        prompt_element.wait_for(state='visible', timeout=3000)
                                        prompt_element.click()
                                        print(f"   Clicked prompt element: {selector}")
                                        time.sleep(0.5)
                                        break
                                    except Exception:
                                        continue

                                if prompt_element:
                                    # Step 3: Paste the image (Cmd+V on macOS)
                                    print("   Pasting image into prompt...")
                                    prompt_element.press('Meta+V')  # Meta = Cmd on macOS
                                    time.sleep(2)  # Wait for upload to process
                                    print("   ✓ Image pasted!")
                                    upload_success = True
                                else:
                                    print("   ⚠️  Could not find prompt textarea")

                            except Exception as upload_error:
                                print(f"⚠️  Clipboard upload error: {upload_error}")
                                import traceback
                                traceback.print_exc()

                            if not upload_success:
                                print("⚠️  Could not auto-upload screenshot")
                                print(f"   Please drag manually: {screenshot_path}")

                            # Now type the prompt
                            print("⌨️  Typing prompt...")
                            prompt = "I've circled a component or components on this PCB with annotations. Please identify what component(s) are circled and provide technical details including: component type, likely part designation, function, any visible markings or identifiers, typical pinouts, test procedures, and expected voltage/resistance values."

                            # Find the textarea - try multiple selectors
                            textarea = None
                            selectors = ['div[contenteditable="true"]', 'textarea', '[role="textbox"]']
                            for selector in selectors:
                                try:
                                    textarea = gemini_page_instance.locator(selector).first
                                    textarea.wait_for(state='visible', timeout=5000)
                                    break
                                except Exception:
                                    continue

                            if textarea:
                                textarea.click()
                                time.sleep(0.5)
                                textarea.fill(prompt)
                                time.sleep(1)

                            # Send message by clicking Run button
                            print("🚀 Clicking Run button to send message...")
                            try:
                                run_button = gemini_page_instance.locator('button[type="submit"]:has-text("Run")').first
                                run_button.click(timeout=5000)
                                print("✓ Message sent to Gemini!")
                            except Exception as e:
                                print(f"⚠️  Could not click Run button: {e}")
                                print("   Please click Run manually")

                    except Exception as e:
                        print(f"✗ Gemini browser automation error: {e}")
                        import traceback
                        traceback.print_exc()
                        with gemini_browser_lock:
                            gemini_playwright_instance = None
                            gemini_browser_instance = None
                            gemini_page_instance = None

                # Start automation in background thread
                automation_thread = threading.Thread(target=automate_gemini, daemon=True)
                automation_thread.start()

                # Return success immediately
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    'status': 'success',
                    'message': 'Opening Gemini in browser...',
                    'screenshot': screenshot_path
                }).encode())

            except Exception as e:
                print(f"✗ Error: {e}")
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode())

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
