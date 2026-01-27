# Microscope Viewer

Web-based viewer for WiFi/USB microscopes with real-time video streaming and image manipulation controls.

![Microscope Viewer](https://img.shields.io/badge/Python-3.6+-blue.svg)
![License](https://img.shields.io/badge/license-BSD--2--Clause-green.svg)

## Features

- **Dual Connection Support**: Automatically detects and connects via WiFi (UDP) or USB
- **Real-time MJPEG Streaming**: Smooth video feed at up to 29 fps
- **Image Manipulation**:
  - Flip horizontal/vertical (default: both flipped)
  - Rotate in 90° increments
  - Zoom 50%-400% from center
  - Reset to defaults
- **Capture Capabilities**:
  - Take screenshots (auto-download)
  - Record video with WebM format (auto-download)
- **Touch Gesture Support**:
  - Pinch to zoom on mobile/tablet
  - Double-tap to reset view
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **FPS Control**: Adjustable frame rate (1-29 fps)

## Requirements

- Python 3.6 or higher
- For USB support: OpenCV (`opencv-python`)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd microscope-viewer
```

2. Install dependencies:
```bash
# For WiFi-only support (no additional dependencies needed)
python3 viewer.py

# For WiFi + USB support
pip3 install opencv-python
```

## Usage

### Quick Start

```bash
python3 viewer.py
```

Then open your browser to: **http://localhost:8080**

### Connection Modes

The viewer automatically detects your microscope:

**WiFi Mode:**
- Connect your computer to the microscope's WiFi network (192.168.29.1)
- Run `viewer.py`
- Uses UDP protocol on ports 20000/10900

**USB Mode:**
- Plug the microscope into a USB port
- Run `viewer.py`
- Automatically falls back to USB if WiFi is unavailable

### Controls

**Sidebar Controls:**
- **Flip H/V**: Toggle horizontal/vertical flip
- **↶/↷ 90°**: Rotate left/right
- **Zoom +/-**: Zoom in/out (50%-400%)
- **Reset**: Reset all transformations
- **📸 Take Screenshot**: Capture and download current frame
- **🎥 Record**: Start/stop video recording
- **FPS Slider**: Adjust frame rate (1-29 fps)

**Keyboard Shortcuts:**
- `S` - Take screenshot
- `Ctrl+R` - Start/Stop recording
- `H` - Flip horizontal
- `V` - Flip vertical
- `R` - Reset transform
- `+/-` - Zoom in/out

**Touch Gestures (Mobile/Tablet):**
- Pinch to zoom
- Double-tap to reset view

## Compatible Devices

### Tested Microscopes
- WiFi microscopes with UDP protocol (192.168.29.1)
- USB microscopes (standard UVC webcam protocol)

### Platforms
- macOS (tested on M2)
- Linux (including Raspberry Pi 4/5)
- Windows

### Browsers
- Chrome/Chromium (recommended)
- Safari
- Firefox
- Edge

## Raspberry Pi Setup

The viewer works great on Raspberry Pi 4/5 for a standalone microscope station:

```bash
# Install OpenCV
sudo apt-get update
sudo apt-get install python3-opencv

# Run viewer
python3 viewer.py

# Access on Pi's display or from another device
# http://<pi-ip-address>:8080
```

## Mobile/Tablet Usage

For iPad or mobile devices:

1. Run `viewer.py` on your computer
2. Connect mobile device to same network (or microscope WiFi)
3. Open browser to: `http://<computer-ip>:8080`
4. Use touch gestures for control

## Troubleshooting

**"No microscope detected"**
- WiFi: Ensure you're connected to microscope's WiFi network (192.168.29.1)
- USB: Install OpenCV with `pip3 install opencv-python`
- Check USB cable connection

**Low frame rate**
- Reduce FPS in sidebar controls
- Close other applications
- Try USB mode instead of WiFi (or vice versa)

**Image appears upside down**
- Use Flip H and Flip V buttons (defaults are already flipped for most microscopes)

**Video recording not working**
- Ensure browser supports MediaRecorder API (Chrome/Firefox recommended)
- Check available disk space
- Try reducing zoom before recording

## Architecture

- **viewer.py**: Main Python server with MJPEG streaming
- **WiFi capture**: UDP socket communication with custom protocol
- **USB capture**: OpenCV VideoCapture for standard webcams
- **Web interface**: Single-page HTML with vanilla JavaScript
- **Video encoding**: Browser-native MediaRecorder (VP9/VP8 WebM)

## Protocol Details

### WiFi Microscope Protocol

The WiFi microscope uses a custom UDP protocol:

**Command Port**: 20000
- `JHCMD\x10\x00` - Initialize
- `JHCMD\x20\x00` - Setup
- `JHCMD\xd0\x01` - Start/heartbeat
- `JHCMD\xd0\x02` - Stop

**Receive Port**: 10900
- Packet structure:
  - Bytes 0-1: Frame count (little-endian)
  - Byte 3: Packet count within frame
  - First packet: Skip 24 bytes (8-byte header + 16-byte metadata)
  - Subsequent packets: Skip 8 bytes (header only)

### Frame Validation

JPEG frames are validated for integrity:
- Start marker: `FF D8`
- End marker: `FF D9`
- Only complete, valid frames are displayed

## License

This project incorporates code from the WiFi microscope reverse engineering project by Christian Zietz.

See `proto/proto.py` for original BSD-2-Clause license.

## Credits

- WiFi protocol reverse engineering: [Christian Zietz](https://www.chzsoft.de/site/hardware/reverse-engineering-a-wifi-microscope/)
- Web viewer implementation: Custom development

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## Known Issues

- macOS may show Continuity Camera warning (harmless)
- Some browsers may not support VP9 codec (falls back to VP8)
- WiFi detection takes ~2 seconds on startup
- **macOS USB Reconnection**: On macOS, when a USB microscope is disconnected and reconnected, the viewer will detect it as stale and require a manual restart. This is a limitation of OpenCV's AVFoundation backend. Linux and Windows do not have this issue.
