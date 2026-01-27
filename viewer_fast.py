#!/usr/bin/env python3
import time
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import cv2
import numpy as np
import usb.core
import usb.util

WEB_PORT = 8080
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
    'stabilize': False,
    'stab_noise': 5,
    'stab_smooth': 30,
    'stab_decay': 60,
    'stab_blend': 4,
    'jpeg_quality': 100,
    'capture_fps': 30,
}
settings_lock = threading.Lock()

stab_prev_gray = None
stab_accumulated_x = 0.0
stab_accumulated_y = 0.0
stab_smooth_correction_x = 0.0
stab_smooth_correction_y = 0.0
stab_frame_buffer = []

def apply_stabilization(frame, s):
    global stab_prev_gray, stab_accumulated_x, stab_accumulated_y
    global stab_smooth_correction_x, stab_smooth_correction_y, stab_frame_buffer
    
    if not s['stabilize']:
        stab_prev_gray = None
        stab_accumulated_x = 0.0
        stab_accumulated_y = 0.0
        stab_smooth_correction_x = 0.0
        stab_smooth_correction_y = 0.0
        stab_frame_buffer = []
        return frame
    
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_small = cv2.resize(gray, (640, 360))
    
    if stab_prev_gray is None:
        stab_prev_gray = gray_small.copy()
        return frame
    
    try:
        shift, _ = cv2.phaseCorrelate(stab_prev_gray.astype(np.float32), gray_small.astype(np.float32))
        dx, dy = shift
        dx *= 2
        dy *= 2
        
        noise_threshold = s['stab_noise'] / 10.0
        if abs(dx) < noise_threshold:
            dx = 0
        if abs(dy) < noise_threshold:
            dy = 0
        
        max_shift = min(w, h) * 0.3
        if abs(dx) < max_shift and abs(dy) < max_shift:
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
    
    stab_prev_gray = gray_small.copy()
    
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
    p = frame
    
    if s['brightness'] != 0 or s['contrast'] != 1.0:
        p = cv2.convertScaleAbs(p, alpha=s['contrast'], beta=s['brightness'])
    
    if s['gain'] != 1.0:
        p = cv2.convertScaleAbs(p, alpha=s['gain'], beta=0)
    
    if s['saturation'] != 1.0:
        hsv = cv2.cvtColor(p, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s['saturation'], 0, 255)
        p = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    if s['flip_h'] and s['flip_v']:
        p = cv2.flip(p, -1)
    elif s['flip_h']:
        p = cv2.flip(p, 1)
    elif s['flip_v']:
        p = cv2.flip(p, 0)
    
    r = int(s['rotate']) % 360
    if r == 90:
        p = cv2.rotate(p, cv2.ROTATE_90_CLOCKWISE)
    elif r == 180:
        p = cv2.rotate(p, cv2.ROTATE_180)
    elif r == 270:
        p = cv2.rotate(p, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    z = s['zoom']
    if z != 1.0:
        h, w = p.shape[:2]
        if z > 1.0:
            ch = int(h / z)
            cw = int(w / z)
            sy = (h - ch) // 2
            sx = (w - cw) // 2
            p = cv2.resize(p[sy:sy+ch, sx:sx+cw], (w, h), interpolation=cv2.INTER_LINEAR)
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
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
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
                        settings['rotate'] = (settings['rotate'] + int(value)) % 360
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
                            'stab_noise': 5, 'stab_smooth': 30,
                            'stab_decay': 60, 'stab_blend': 2
                        })
            
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
