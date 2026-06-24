"""
PlateVision v9 — Flask entry-point.

Root-cause fix: ALPR gives undersized bbox.
_grow_bbox_to_plate() now walks outward from ALPR box to find TRUE plate edges
BEFORE any corner detection runs — so sticker always covers the full plate.
"""

from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import re
import os
from dataclasses import replace

from plate_vision_core import (
    find_plate_corners,
    _contour_quad,
    _color_quad,
    _warp_sticker_onto_plate,
    apply_logo_perspective,
    detect_plate_color,
    draw_debug_overlay,
    _grow_bbox_to_plate,
    _compute_plate_dims,
    _is_axis_aligned_box,
    _bbox_corners,
    _order_corners,
    _quad_area,
    _expanded_bbox_quad,
    PLATE_STD_ASPECT,
    PLATE_HSRP_ASPECT,
    PLATE_MIN_ASPECT,
)

app = Flask(__name__)

LOGO_PATH = os.path.join(os.path.dirname(__file__), 'static', 'img', 'caryanams_logo.png')
_logo_cache = None


# ══════════════════════════════════════════════════════════════════════════════
# LOGO
# ══════════════════════════════════════════════════════════════════════════════

def _load_logo_bgr_on_white():
    raw = cv2.imread(LOGO_PATH, cv2.IMREAD_UNCHANGED)
    if raw is None:
        return np.full((60, 200, 3), 255, dtype=np.uint8)
    if raw.ndim == 2:
        return cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    if raw.shape[2] == 4:
        bgr   = raw[:, :, :3].astype(float)
        alpha = (raw[:, :, 3] / 255.0)[:, :, np.newaxis]
        return (bgr * alpha + 255 * (1 - alpha)).astype(np.uint8)
    return raw[:, :, :3]


def get_logo():
    global _logo_cache
    if _logo_cache is None:
        logo   = _load_logo_bgr_on_white()
        h, w   = logo.shape[:2]
        border = max(4, int(min(h, w) * 0.03))
        inner  = logo[border:h-border, border:w-border]
        gray   = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
        mask   = gray < 235
        ys, xs = np.where(mask)
        if len(xs) == 0:
            _logo_cache = inner
        else:
            m = 4
            _logo_cache = inner[
                max(0, ys.min()-m):min(inner.shape[0], ys.max()+m),
                max(0, xs.min()-m):min(inner.shape[1], xs.max()+m)
            ]
    return _logo_cache


# ══════════════════════════════════════════════════════════════════════════════
# ALPR
# ══════════════════════════════════════════════════════════════════════════════

_alpr = None

def get_alpr():
    global _alpr
    if _alpr is None:
        from fast_alpr import ALPR
        _alpr = ALPR(
            detector_model="yolo-v9-t-640-license-plate-end2end",
            detector_conf_thresh=0.25,
            ocr_model="cct-xs-v2-global-model",
        )
    return _alpr


# ══════════════════════════════════════════════════════════════════════════════
# PLATE TEXT VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

MIN_PLATE_ASPECT_RATIO = 0.4
VALID_STATE_CODES = {
    'AP','AR','AS','BR','CG','CH','DD','DL','DN','GA','GJ','HP','HR',
    'JH','JK','KA','KL','LA','LD','MH','ML','MN','MP','MZ','NL','OD',
    'OR','PB','PY','RJ','SK','TN','TR','TS','UK','UP','WB','AN'
}
INDIAN_PLATE_RE = re.compile(r'([A-Z]{2})(\d{2})([A-Z]{1,3})(\d{1,4})')
BH_PLATE_RE     = re.compile(r'(\d{2})(BH)(\d{4})([A-Z]{1,2})')

def clean(text):
    return re.sub(r'[^A-Z0-9]', '', text.upper())

def format_plate(raw):
    if not raw: return None
    txt = clean(raw)
    if len(txt) < 8: return None
    m = BH_PLATE_RE.search(txt)
    if m: return ''.join(m.groups())
    for m in INDIAN_PLATE_RE.finditer(txt):
        state = m.group(1)
        if state not in VALID_STATE_CODES: continue
        plate = state + m.group(2) + m.group(3) + m.group(4).zfill(3)
        if 9 <= len(plate) <= 11: return plate
    if 8 <= len(txt) <= 11 and not re.fullmatch(r'(.)\1+', txt): return txt
    return None


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def decode_image(data_url):
    _, data = data_url.split(',', 1)
    arr = np.frombuffer(base64.b64decode(data), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def encode_image(img):
    _, buf = cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return "data:image/png;base64," + base64.b64encode(buf).decode()

def enhance_for_detection(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)


# ══════════════════════════════════════════════════════════════════════════════
# ALPR MULTI-PASS
# ══════════════════════════════════════════════════════════════════════════════

def run_alpr_dual_pass(alpr, img):
    results = alpr.predict(img)
    if results: return results, 'plain'
    results = alpr.predict(enhance_for_detection(img))
    if results: return results, 'clahe'
    results = run_alpr_tiled(alpr, img)
    return results, 'tiled'

def _make_recovery_tiles(iw, ih):
    y0 = int(ih * 0.25); tiles = []
    col_w = max(1, int(iw * 0.55)); step = max(1, int(iw * 0.30))
    x = 0
    while True:
        x2 = min(iw, x + col_w); x1 = max(0, x2 - col_w)
        tiles.append((x1, y0, x2, ih))
        if x2 >= iw: break
        x += step
    tiles.append((0, y0, iw, ih))
    return tiles

def run_alpr_tiled(alpr, img, target_long_side=900):
    ih, iw = img.shape[:2]; recovered = []
    for (tx1, ty1, tx2, ty2) in _make_recovery_tiles(iw, ih):
        crop = img[ty1:ty2, tx1:tx2]
        ch, cw = crop.shape[:2]
        if ch < 10 or cw < 10: continue
        long_side = max(ch, cw)
        scale = target_long_side / long_side if long_side < target_long_side else 1.0
        crop_in = crop if scale == 1.0 else cv2.resize(
            crop,
            (max(1, int(round(cw*scale))), max(1, int(round(ch*scale)))),
            interpolation=cv2.INTER_CUBIC,
        )
        try: tile_results = alpr.predict(crop_in)
        except Exception: continue
        for r in tile_results:
            bb = r.detection.bounding_box
            new_bb = replace(bb,
                x1=int(bb.x1/scale)+tx1, y1=int(bb.y1/scale)+ty1,
                x2=int(bb.x2/scale)+tx1, y2=int(bb.y2/scale)+ty1)
            recovered.append(replace(r, detection=replace(r.detection, bounding_box=new_bb)))
    recovered.sort(key=lambda r: r.detection.bounding_box.area, reverse=True)
    return recovered


# ══════════════════════════════════════════════════════════════════════════════
# BORDER HELPER
# ══════════════════════════════════════════════════════════════════════════════

def apply_border_only(img, corners, color=(0,120,255), thickness=4):
    out = img.copy()
    pts = corners.astype(np.int32).reshape((-1,1,2))
    cv2.polylines(out,[pts],True,
                  (int(color[0]*0.4),int(color[1]*0.4),int(color[2]*0.4)),
                  thickness+4, cv2.LINE_AA)
    cv2.polylines(out,[pts],True, color, thickness, cv2.LINE_AA)
    for pt in corners.astype(int):
        cv2.circle(out,tuple(pt),thickness+4,color,-1,cv2.LINE_AA)
        cv2.circle(out,tuple(pt),thickness+1,(255,255,255),-1,cv2.LINE_AA)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# FLASK ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.get_json()
        img  = decode_image(data.get('image', ''))
        mode = data.get('mode', 'logo')   # "logo" | "border" | "debug"
        if img is None:
            return jsonify({'error': 'Could not decode image'}), 400

        alpr = get_alpr()
        results, detect_pass = run_alpr_dual_pass(alpr, img)
        used_enhancement = (detect_pass != 'plain')

        if not results:
            return jsonify({
                'result_image': encode_image(img),
                'plate_text':   '',
                'message':      'No number plate detected',
                'status':       'none',
            })

        results_sorted = sorted(
            results,
            key=lambda r: (r.detection.bounding_box.area, r.detection.confidence),
            reverse=True,
        )
        max_area = results_sorted[0].detection.bounding_box.area if results_sorted else 0

        def _passes(cbb):
            return cbb.area >= max_area * 0.35

        best = bb = None
        for c in results_sorted:
            cbb = c.detection.bounding_box
            if not _passes(cbb): break
            if format_plate(c.ocr.text if c.ocr else ''):
                best = c; bb = cbb; break

        if best is None:
            for c in results_sorted:
                cbb = c.detection.bounding_box
                if not _passes(cbb): break
                bw = cbb.x2 - cbb.x1; bh = cbb.y2 - cbb.y1
                if bh > 0 and bw / bh >= MIN_PLATE_ASPECT_RATIO:
                    best = c; bb = cbb; break

        if best is None:
            return jsonify({
                'result_image': encode_image(img),
                'plate_text':   '',
                'message':      f'No clear plate (tried {len(results)} detections)',
                'status':       'none',
            })

        raw_text   = best.ocr.text if best.ocr else ''
        plate_text = format_plate(raw_text)

        # Original bbox dimensions (used for aspect correction)
        bbox_w = float(bb.x2 - bb.x1)
        bbox_h = float(bb.y2 - bb.y1)

        # ── Detect color FIRST (needed by grow + contour hint) ────────────
        plate_color = detect_plate_color(img, bb.x1, bb.y1, bb.x2, bb.y2)

        # ── Grow bbox → TRUE plate boundary ──────────────────────────────
        gx1, gy1, gx2, gy2 = _grow_bbox_to_plate(
            img, bb.x1, bb.y1, bb.x2, bb.y2, plate_color)
        grown_bbox_w = float(gx2 - gx1)
        grown_bbox_h = float(gy2 - gy1)

        # ── Corner detection (on grown bbox) ──────────────────────────────
        corners = find_plate_corners(img, bb.x1, bb.y1, bb.x2, bb.y2,
                                     plate_hint=plate_color)

        pw, ph = _compute_plate_dims(corners)
        detected_aspect  = float(pw / ph) if ph > 0 else 0.0
        bbox_aspect      = float(bbox_w / bbox_h) if bbox_h > 0 else 0.0
        aspect_corrected = bbox_aspect < PLATE_MIN_ASPECT

        # ── Render ────────────────────────────────────────────────────────
        if mode == 'border':
            border_color = (0,165,255) if plate_color == 'yellow' else (0,120,255)
            result_img   = apply_border_only(img, corners, color=border_color)

        elif mode == 'debug':
            ih, iw = img.shape[:2]
            cx1 = max(0, gx1 - 8); cy1 = max(0, gy1 - 8)
            cx2 = min(iw, gx2 + 8); cy2 = min(ih, gy2 + 8)
            crop = img[cy1:cy2, cx1:cx2]
            bbox_area = grown_bbox_w * grown_bbox_h
            refined = (
                _contour_quad(crop, cx1, cy1, bbox_area)
                or _color_quad(crop, cx1, cy1, bbox_area, plate_color)
            )
            result_img = draw_debug_overlay(
                img,
                alpr_bbox=(bb.x1, bb.y1, bb.x2, bb.y2),
                refined_contour_pts=refined,
                final_quad=corners,
                grown_bbox=(gx1, gy1, gx2, gy2),
            )

        else:  # 'logo'
            # Use GROWN bbox dimensions for aspect correction (more accurate)
            result_img = apply_logo_perspective(
                img, corners, plate_color,
                grown_bbox_w, grown_bbox_h,
                get_logo_fn=get_logo,
                raw_x1=gx1, raw_y1=gy1, raw_x2=gx2, raw_y2=gy2,
            )

        is_perspective_quad = not _is_axis_aligned_box(corners)

        if mode == 'debug':
            message = (f'DEBUG: alpr=({bb.x1},{bb.y1},{bb.x2},{bb.y2}) '
                       f'grown=({gx1},{gy1},{gx2},{gy2}) '
                       f'color={plate_color}')
        elif mode == 'border':
            message = f'Plate: {plate_text}' if plate_text else 'Plate highlighted'
        else:
            message = 'Number plate hidden' if plate_text else 'Plate region covered'

        if plate_color == 'yellow':      message += ' (yellow plate)'
        if aspect_corrected:             message += ' (angle-corrected)'
        if detect_pass == 'clahe':       message += ' (CLAHE enhanced)'
        elif detect_pass == 'tiled':     message += ' (tiled re-scan)'

        return jsonify({
            'result_image':     encode_image(result_img),
            'plate_text':       plate_text or '',
            'message':          message,
            'status':           'success' if plate_text else 'partial',
            'corners':          corners.tolist(),
            'mode':             mode,
            'enhanced':         bool(used_enhancement),
            'detect_pass':      detect_pass,
            'conf':             round(float(best.detection.confidence), 3),
            'plate_color':      plate_color,
            'perspective_quad': bool(is_perspective_quad),
            'aspect_corrected': bool(aspect_corrected),
            'detected_aspect':  round(float(detected_aspect), 2),
            'bbox_aspect':      round(float(bbox_aspect), 2),
            'alpr_bbox':        [bb.x1, bb.y1, bb.x2, bb.y2],
            'grown_bbox':       [gx1, gy1, gx2, gy2],
        })

    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


if __name__ == '__main__':
    print("\n  PlateVision v9 → http://localhost:5000")
    print("  Root-cause fix: color-walk bbox growth before corner detection\n")
    app.run(debug=False, host='0.0.0.0', port=5000)
