# Accessories Virtual Try-On Web App — Implementation Plan

## Overview

Build a real-time virtual try-on web app where users see their webcam feed with AR accessory overlays (glasses, hats, earrings). The app uses MediaPipe Face Mesh for landmark detection, OpenCV for frame processing, and Flask for serving. Users can select from a built-in catalogue, upload custom PNGs, take snapshots, and get face-shape-based recommendations.

---

## Proposed Changes

### 1. Project Setup & Dependencies

#### [NEW] requirements.txt
- `flask`, `opencv-python`, `mediapipe`, `numpy`, `Pillow`, `rembg`

#### [NEW] Accessory Assets
- Generate 3 glasses PNGs, 3 hats PNGs, and 3 earrings PNGs using the image generation tool
- All assets will be transparent-background PNGs placed under `static/accessories/{glasses,hats,earrings}/`

---

### 2. Overlay Engine — `overlay_engine.py`

#### [NEW] overlay_engine.py

Core class `OverlayEngine` with these responsibilities:

| Method | Purpose |
|---|---|
| `__init__()` | Initialize MediaPipe Face Mesh (with `refine_landmarks=True`, `max_num_faces=1`) |
| `load_accessory(path, category)` | Load a PNG with alpha channel (`cv2.IMREAD_UNCHANGED`) |
| `process_frame(frame)` | Run face mesh detection, compute landmarks, call overlay method |
| `_overlay_glasses(frame, landmarks)` | Place glasses between eye landmarks 33 & 263, scaled by inter-eye distance × 2.5 |
| `_overlay_hat(frame, landmarks)` | Place hat above landmark 10 (forehead), scaled by face width × 3.0 |
| `_overlay_earrings(frame, landmarks)` | Place earrings at landmarks 234 (left) & 454 (right), scaled × 0.8 |
| `_compute_rotation(landmarks)` | `arctan2(right_eye.y - left_eye.y, right_eye.x - left_eye.x)` in degrees |
| `_alpha_blend(frame, overlay, x, y)` | Per-pixel: `out = α·accessory + (1-α)·frame` |
| `_rotate_image(img, angle)` | Rotate PNG around center with `cv2.getRotationMatrix2D` |
| `analyze_face_shape(frame)` | Measure face width/height/jaw/forehead proportions → classify into Oval/Round/Square/Heart/Oblong |

**Key landmark indices:**
- Left eye outer: **33**, Right eye outer: **263**
- Left eye inner: **133**, Right eye inner: **362**
- Forehead top: **10**
- Left ear: **234**, Right ear: **454**
- Chin: **152**, Left cheek: **234**, Right cheek: **454**
- Forehead width: landmarks **71** and **301**

---

### 3. Flask Backend — `app.py`

#### [NEW] app.py

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Serve `index.html` |
| `/video_feed` | GET | MJPEG stream — each frame processed through overlay engine |
| `/select` | POST | Accept `{category, filename}`, load accessory from catalogue |
| `/upload` | POST | Accept PNG file, run `rembg` for background removal, save to `/uploads/`, apply |
| `/snapshot` | GET | Capture current frame → return as `image/png` download |
| `/face_shape` | POST | Analyze current frame → return JSON `{shape, suggestion}` |
| `/accessories` | GET | Return JSON list of available accessories per category |

**Streaming implementation:**
```python
def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame = engine.process_frame(frame)
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
```

---

### 4. Frontend — `templates/index.html` + `static/css/style.css` + `static/js/main.js`

#### [NEW] index.html
- Single-page layout with two-panel design
- Left: webcam feed (`<img src="/video_feed">`)
- Right: tabbed accessory catalogue + controls

#### [NEW] style.css
**Dark premium theme** with:
- Background: deep charcoal (`#0f0f1a`) with subtle gradient
- Cards: glassmorphism (`backdrop-filter: blur`) with `rgba(255,255,255,0.05)` backgrounds
- Accent color: vibrant purple-blue gradient (`#7c3aed` → `#3b82f6`)
- Google Font: **Inter** for clean typography
- Smooth hover animations on accessory thumbnails (scale + glow)
- Responsive grid layout for accessory catalogue
- Snapshot modal with blur backdrop
- Tab switches with animated underline indicator

#### [NEW] main.js
- Tab switching for Glasses / Hats / Earrings categories
- `fetch('/accessories')` on load to populate catalogue grid
- Click handler → `POST /select` to apply accessory
- Upload handler with file type validation (PNG only) + preview
- Snapshot button → `GET /snapshot` → display in modal with download link
- Face shape button → `POST /face_shape` → display result card
- "Remove accessory" button to clear current overlay

---

### 5. Upload Feature

Handled in `app.py` `/upload` route:
1. Validate file is PNG
2. Save to `uploads/` directory
3. Run `rembg.remove()` on the image to strip any solid background
4. Pass cleaned PNG path to overlay engine
5. Auto-detect category from user selection (radio buttons in upload form)

---

### 6. Face Shape Detection

In `overlay_engine.py` → `analyze_face_shape()`:

| Face Shape | Condition |
|---|---|
| Round | width/height ratio > 0.85 |
| Square | width/height ratio > 0.85 AND jawline ≈ forehead width |
| Oblong | width/height ratio < 0.65 |
| Heart | forehead width > jawline width significantly |
| Oval | Default / balanced proportions |

**Recommendations:**
| Face Shape | Suggestion |
|---|---|
| Round | Angular/rectangular frames to add definition |
| Square | Round/oval frames to soften angles |
| Oval | Most styles work — try aviators |
| Heart | Bottom-heavy frames to balance forehead |
| Oblong | Oversized frames to add width |

---

## Open Questions

> [!IMPORTANT]
> **Accessory Assets**: I'll generate sample accessory PNG images using the image generation tool for glasses, hats, and earrings. These will be stylized illustrations. Is that acceptable, or do you have specific asset files to use?

> [!NOTE]
> **`rembg` dependency**: The `rembg` library downloads a ~170MB ONNX model on first use. This is expected behavior. If you'd prefer to skip this dependency for a lighter install, I can use a simpler color-based background removal instead.

---

## Verification Plan

### Automated Tests
1. `pip install -r requirements.txt` — verify all dependencies install
2. `python app.py` — verify server starts without errors on port 5000

### Manual Verification (Browser)
1. Open `http://localhost:5000` — verify webcam feed displays
2. Click each accessory category tab — verify thumbnails load
3. Click an accessory — verify it overlays on face in real time
4. Move/tilt head — verify accessory follows and rotates
5. Upload a custom PNG — verify it applies correctly
6. Click Snapshot — verify image captures and downloads
7. Click Face Shape — verify analysis returns a result

### Browser Recording
- Use the browser subagent to navigate to the app and record the UI for visual verification
