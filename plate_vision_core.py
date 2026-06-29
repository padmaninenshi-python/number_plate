"""
PlateVision v10 — COMPLETE REWRITE of corner detection + bbox growth.

ROOT CAUSES FIXED (v9 failures):
  1. _grow_bbox_to_plate() was a STUB — returned ALPR bbox unchanged.
     Now: walks outward pixel-by-pixel using plate color mask to find TRUE edges.
  2. find_plate_corners() used OTSU on full expanded crop — white car body
     confused it. Now: uses plate-color-aware mask, not raw OTSU.
  3. Contour filtering too loose — accepted background blobs far from plate.
     Now: strict center proximity + aspect score.
  4. Yellow plate expansion was same path as white — different HSV range needed.
"""

import cv2
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

PLATE_STD_ASPECT  = 3.33
PLATE_HSRP_ASPECT = 4.17
PLATE_MIN_ASPECT  = 1.8

QUAD_MIN_ASPECT = 1.0
QUAD_MAX_ASPECT = 9.0

PAD_W_FRAC = 0.0
PAD_H_FRAC = 0.0

# Max outward expansion allowed when growing bbox (fraction of original bbox size)
GROW_MAX_FRAC = 0.40   # 40% max expansion per side


# ══════════════════════════════════════════════════════════════════════════════
# GEOMETRY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _order_corners(pts):
    pts = np.array(pts, dtype=np.float32).reshape(4, 2)
    s   = pts.sum(axis=1)
    d   = np.diff(pts, axis=1).squeeze()
    tl  = pts[np.argmin(s)]
    br  = pts[np.argmax(s)]
    tr  = pts[np.argmin(d)]
    bl  = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _bbox_corners(x1, y1, x2, y2):
    return np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], dtype=np.float32)


def _quad_area(pts):
    n = len(pts); area = 0.0
    for i in range(n):
        j = (i+1) % n
        area += pts[i][0]*pts[j][1]
        area -= pts[j][0]*pts[i][1]
    return abs(area)/2.0


def _is_convex(pts):
    if len(pts) != 4:
        return False
    hull = cv2.convexHull(pts.astype(np.float32), returnPoints=True)
    return len(hull) == 4


def _is_axis_aligned_box(corners, tol=3):
    tl, tr, br, bl = corners
    return (abs(tl[1]-tr[1]) < tol and abs(br[1]-bl[1]) < tol and
            abs(tl[0]-bl[0]) < tol and abs(tr[0]-br[0]) < tol)


def _compute_plate_dims(corners):
    tl, tr, br, bl = corners
    w = (float(np.linalg.norm(tr-tl)) + float(np.linalg.norm(br-bl))) / 2.0
    h = (float(np.linalg.norm(bl-tl)) + float(np.linalg.norm(br-tr))) / 2.0
    return w, h


# ══════════════════════════════════════════════════════════════════════════════
# STEP 0 — GROW THE ALPR BBOX TO TRUE PLATE BOUNDARIES  ← KEY FIX v10
# ══════════════════════════════════════════════════════════════════════════════

def _grow_bbox_to_plate(img, x1, y1, x2, y2, plate_color):
    """
    Walk outward from ALPR bbox in all 4 directions using plate color mask.
    Stops when the plate color density drops below threshold.
    This finds the TRUE plate boundary before corner detection runs.
    """
    ih, iw = img.shape[:2]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    bw = x2 - x1
    bh = y2 - y1

    max_exp_x = int(bw * GROW_MAX_FRAC)
    max_exp_y = int(bh * GROW_MAX_FRAC)

    # Build color mask for the whole image
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if plate_color == 'yellow':
        mask = cv2.inRange(hsv, np.array([12, 60, 80]), np.array([40, 255, 255]))
    else:
        white  = cv2.inRange(hsv, np.array([0,   0, 185]), np.array([180, 45, 255]))
        slight = cv2.inRange(hsv, np.array([14, 15, 170]), np.array([38, 75, 255]))
        mask   = cv2.bitwise_or(white, slight)

    # Morphological clean to remove noise
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    # Threshold: row/col must have at least this fraction of plate pixels to expand
    DENSITY_THRESH = 0.25

    # Grow LEFT
    new_x1 = x1
    for step in range(1, max_exp_x + 1):
        col = max(0, x1 - step)
        col_strip = mask[y1:y2, col:col+1]
        if col_strip.size == 0: break
        density = np.mean(col_strip) / 255.0
        if density < DENSITY_THRESH: break
        new_x1 = col

    # Grow RIGHT
    new_x2 = x2
    for step in range(1, max_exp_x + 1):
        col = min(iw - 1, x2 + step)
        col_strip = mask[y1:y2, col:col+1]
        if col_strip.size == 0: break
        density = np.mean(col_strip) / 255.0
        if density < DENSITY_THRESH: break
        new_x2 = col

    # Grow UP
    new_y1 = y1
    for step in range(1, max_exp_y + 1):
        row = max(0, y1 - step)
        row_strip = mask[row:row+1, x1:x2]
        if row_strip.size == 0: break
        density = np.mean(row_strip) / 255.0
        if density < DENSITY_THRESH: break
        new_y1 = row

    # Grow DOWN
    new_y2 = y2
    for step in range(1, max_exp_y + 1):
        row = min(ih - 1, y2 + step)
        row_strip = mask[row:row+1, x1:x2]
        if row_strip.size == 0: break
        density = np.mean(row_strip) / 255.0
        if density < DENSITY_THRESH: break
        new_y2 = row

    return (
        max(0,  new_x1),
        max(0,  new_y1),
        min(iw, new_x2),
        min(ih, new_y2),
    )


# ══════════════════════════════════════════════════════════════════════════════
# YELLOW / WHITE PLATE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_plate_color(img, x1, y1, x2, y2):
    """
    Two-pass yellow detection on the inner 70% of the ALPR crop.
    """
    ih, iw = img.shape[:2]
    pw, ph = x2 - x1, y2 - y1
    sx1 = int(max(0,  x1 + pw * 0.15))
    sy1 = int(max(0,  y1 + ph * 0.15))
    sx2 = int(min(iw, x2 - pw * 0.15))
    sy2 = int(min(ih, y2 - ph * 0.15))
    crop = img[sy1:sy2, sx1:sx2]
    if crop.size == 0:
        return 'white'

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Pass 1 — strict
    strict = cv2.inRange(hsv, np.array([18, 130, 130]), np.array([34, 255, 255]))
    if np.mean(strict) / 255.0 > 0.20:
        return 'yellow'

    # Pass 2 — relaxed (faded/dim yellow plates like old taxis)
    relaxed = cv2.inRange(hsv, np.array([12, 70, 90]), np.array([40, 255, 255]))
    if np.mean(relaxed) / 255.0 > 0.28:
        return 'yellow'

    return 'white'


# ══════════════════════════════════════════════════════════════════════════════
# COLOR MASK
# ══════════════════════════════════════════════════════════════════════════════

def _color_plate_mask(crop, plate_hint='white'):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    if plate_hint == 'yellow':
        mask = cv2.inRange(hsv, np.array([12, 60, 80]), np.array([40, 255, 255]))
    else:
        white  = cv2.inRange(hsv, np.array([0,   0, 185]), np.array([180, 45, 255]))
        slight = cv2.inRange(hsv, np.array([14, 15, 170]), np.array([38, 75, 255]))
        mask   = cv2.bitwise_or(white, slight)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    return mask


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def _is_valid_plate_quad(pts, bbox_area):
    area = _quad_area(pts)
    if area < bbox_area * 0.20 or area > bbox_area * 4.0:
        return False
    w, h = _compute_plate_dims(pts)
    if h < 4 or w < 4:
        return False
    aspect = w / h
    if not (QUAD_MIN_ASPECT <= aspect <= QUAD_MAX_ASPECT):
        return False
    if not _is_convex(pts):
        return False
    return True


def _clamp_to_region(quad, x1, y1, x2, y2):
    out = quad.copy()
    out[:, 0] = np.clip(out[:, 0], x1, x2)
    out[:, 1] = np.clip(out[:, 1], y1, y2)
    return out.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# CONTOUR QUAD
# ══════════════════════════════════════════════════════════════════════════════

def _contour_quad(crop, cx1, cy1, bbox_area):
    ch, cw = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    def _edges(g):
        blurred = cv2.GaussianBlur(g, (5, 5), 0)
        median  = float(np.median(blurred))
        lo      = max(15, int(0.50 * median))
        hi      = min(220, int(1.40 * median))
        e       = cv2.Canny(blurred, lo, hi)
        k       = np.ones((3, 3), np.uint8)
        return cv2.dilate(e, k, iterations=1)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    passes = [gray, clahe.apply(gray)]

    for g in passes:
        edges = _edges(g)
        cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for cnt in cnts[:12]:
            if cv2.contourArea(cnt) < ch * cw * 0.06:
                break
            peri = cv2.arcLength(cnt, True)
            for eps in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08]:
                approx = cv2.approxPolyDP(cnt, eps * peri, True)
                if len(approx) == 4:
                    pts = approx.reshape(4, 2).astype(np.float32)
                    pts[:, 0] += cx1
                    pts[:, 1] += cy1
                    pts = _order_corners(pts)
                    if _is_valid_plate_quad(pts, bbox_area):
                        return pts
    return None


# ══════════════════════════════════════════════════════════════════════════════
# COLOR QUAD
# ══════════════════════════════════════════════════════════════════════════════

def _color_quad(crop, cx1, cy1, bbox_area, plate_hint='white'):
    mask = _color_plate_mask(crop, plate_hint)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < bbox_area * 0.15:
        return None

    peri = cv2.arcLength(cnt, True)
    for eps in [0.02, 0.03, 0.05, 0.07, 0.10]:
        approx = cv2.approxPolyDP(cnt, eps * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype(np.float32)
            pts[:, 0] += cx1
            pts[:, 1] += cy1
            pts = _order_corners(pts)
            if _is_valid_plate_quad(pts, bbox_area):
                return pts

    # minAreaRect fallback
    rect = cv2.minAreaRect(cnt)
    box  = cv2.boxPoints(rect).astype(np.float32)
    box[:, 0] += cx1
    box[:, 1] += cy1
    pts = _order_corners(box)
    if _is_valid_plate_quad(pts, bbox_area):
        return pts
    return None


# ══════════════════════════════════════════════════════════════════════════════
# minAreaRect ON EDGE MAP
# ══════════════════════════════════════════════════════════════════════════════

def _minAreaRect_quad(crop, cx1, cy1, bbox_area):
    gray    = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe   = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    blurred = cv2.GaussianBlur(clahe.apply(gray), (5, 5), 0)
    edges   = cv2.Canny(blurred, 15, 80)
    k       = np.ones((3, 3), np.uint8)
    edges   = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=3)

    pts = cv2.findNonZero(edges)
    if pts is None or len(pts) < 20:
        return None

    rect = cv2.minAreaRect(pts)
    box  = cv2.boxPoints(rect).astype(np.float32)
    box[:, 0] += cx1
    box[:, 1] += cy1
    ordered = _order_corners(box)
    if _is_valid_plate_quad(ordered, bbox_area):
        return ordered

    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        all_pts = np.vstack([c.reshape(-1,2) for c in cnts if cv2.contourArea(c) > 20])
        if len(all_pts) >= 4:
            rect2 = cv2.minAreaRect(all_pts)
            box2  = cv2.boxPoints(rect2).astype(np.float32)
            box2[:, 0] += cx1
            box2[:, 1] += cy1
            ordered2 = _order_corners(box2)
            if _is_valid_plate_quad(ordered2, bbox_area):
                return ordered2
    return None


# ══════════════════════════════════════════════════════════════════════════════
# EXPANDED BBOX FALLBACK
# ══════════════════════════════════════════════════════════════════════════════

def _expanded_bbox_quad(x1, y1, x2, y2, img_w, img_h):
    """Last resort: padded ALPR/grown bbox rect."""
    pw = x2 - x1; ph = y2 - y1
    pad_w = int(pw * PAD_W_FRAC)
    pad_h = int(ph * PAD_H_FRAC)
    nx1 = max(0,     x1 - pad_w)
    ny1 = max(0,     y1 - pad_h)
    nx2 = min(img_w, x2 + pad_w)
    ny2 = min(img_h, y2 + pad_h)
    return _bbox_corners(nx1, ny1, nx2, ny2)


# ══════════════════════════════════════════════════════════════════════════════
# MASTER CORNER FINDER  — v10: uses GROWN bbox + plate-color mask  ← KEY FIX
# ══════════════════════════════════════════════════════════════════════════════

def find_plate_corners(img, x1, y1, x2, y2, plate_hint='white'):
    """
    Find exact 4 corners of the plate.

    v10 changes vs v9:
    - Uses plate-COLOR mask (not raw OTSU) to segment the plate region.
      OTSU fails when car body is also white — color mask isolates plate.
    - Operates on GROWN bbox (grown before this call by _grow_bbox_to_plate).
    - Expand by 25%x / 35%y to see surrounding dark bumper frame.
    - Multi-candidate scoring: proximity to ALPR center + aspect closeness to 3.33.
    - Strict aspect window [1.5, 7.0] to reject non-plate shapes.
    - Falls back to grown ALPR bbox (not just ALPR bbox) if all fails.
    """
    ih, iw = img.shape[:2]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    bw = x2 - x1
    bh = y2 - y1

    # Expand to see dark bumper frame surrounding the plate
    ex = int(bw * 0.25)
    ey = int(bh * 0.35)
    ex1 = max(0,  x1 - ex);  ey1 = max(0,  y1 - ey)
    ex2 = min(iw, x2 + ex);  ey2 = min(ih, y2 + ey)
    crop = img[ey1:ey2, ex1:ex2]
    if crop.size == 0:
        return _bbox_corners(x1, y1, x2, y2)

    # ── Step 1: plate-color mask (avoids white-car-body confusion) ──────────
    plate_mask = _color_plate_mask(crop, plate_hint)

    # Fill internal text holes so plate is one solid blob
    k_fill = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 7))
    plate_mask = cv2.morphologyEx(plate_mask, cv2.MORPH_CLOSE, k_fill, iterations=4)
    plate_mask = cv2.morphologyEx(plate_mask, cv2.MORPH_OPEN,  k_fill, iterations=1)

    cnts, _ = cv2.findContours(plate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ── Step 2: if color mask gives no contours, fall back to OTSU ──────────
    if not cnts:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k2, iterations=3)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  k2, iterations=1)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not cnts:
        return _bbox_corners(x1, y1, x2, y2)

    # ALPR center in crop coords (used for proximity scoring)
    alpr_cx = (x1 + x2) / 2.0 - ex1
    alpr_cy = (y1 + y2) / 2.0 - ey1
    alpr_area = float(bw * bh)

    best = None
    best_score = -1.0

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        # Must be meaningfully sized relative to ALPR bbox
        if area < alpr_area * 0.25 or area > alpr_area * 5.0:
            continue
        # Center must be close to ALPR center
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']
        dist = ((cx - alpr_cx)**2 + (cy - alpr_cy)**2) ** 0.5
        # Allow max distance = 60% of bbox diagonal
        max_dist = ((bw**2 + bh**2) ** 0.5) * 0.60
        if dist > max_dist:
            continue

        rect = cv2.minAreaRect(cnt)
        rw, rh = rect[1]
        if rh == 0 or rw == 0:
            continue
        aspect = max(rw, rh) / min(rw, rh)
        # Strict aspect window for plate
        if not (1.5 <= aspect <= 7.0):
            continue

        # Score: weighted combination of aspect closeness + proximity
        aspect_score = 1.0 - min(abs(aspect - PLATE_STD_ASPECT), abs(aspect - PLATE_HSRP_ASPECT)) / 5.0
        prox_score   = 1.0 - dist / (max_dist + 1.0)
        size_score   = min(area / alpr_area, alpr_area / (area + 1)) 
        score = aspect_score * 0.5 + prox_score * 0.35 + size_score * 0.15

        if score > best_score:
            best_score = score
            best = (cnt, rect)

    if best is not None:
        cnt, rect = best
        box = cv2.boxPoints(rect).astype(np.float32)
        # Translate back to full image coords
        box[:, 0] += ex1
        box[:, 1] += ey1
        ordered = _order_corners(box)
        w, h = _compute_plate_dims(ordered)
        # Accept if size is reasonable (45%–220% of ALPR bbox)
        if (h >= 4 and w >= 4 and
            bw * 0.45 <= w <= bw * 2.2 and
            bh * 0.45 <= h <= bh * 2.2):
            return ordered

    # Fallback: use the grown ALPR bbox directly as rectangle
    return _bbox_corners(x1, y1, x2, y2)


# ══════════════════════════════════════════════════════════════════════════════
# ASPECT CORRECTION (side-angle shots)
# ══════════════════════════════════════════════════════════════════════════════

def _aspect_correct_corners(corners, bbox_w, bbox_h):
    plate_w, plate_h = _compute_plate_dims(corners)
    if plate_h < 1 or plate_w < 1:
        return corners

    detected_aspect = plate_w / plate_h
    if detected_aspect >= 2.0:
        return corners

    target_h  = plate_w / PLATE_STD_ASPECT
    scale     = target_h / plate_h
    center    = corners.mean(axis=0)
    width_dir = corners[1] - corners[0]
    wn = np.linalg.norm(width_dir)
    if wn < 1:
        return corners
    width_dir  = width_dir / wn
    height_dir = np.array([width_dir[1], -width_dir[0]], dtype=np.float32)

    corrected = []
    for pt in corners:
        v      = pt - center
        w_comp = float(np.dot(v, width_dir))
        h_comp = float(np.dot(v, height_dir))
        corrected.append(center + w_comp * width_dir + (h_comp * scale) * height_dir)

    return np.array(corrected, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# STICKER CANVAS
# ══════════════════════════════════════════════════════════════════════════════

def _sharp_resize(img, target_w, target_h):
    target_w, target_h = max(1, target_w), max(1, target_h)
    src_h, src_w = img.shape[:2]
    interp  = cv2.INTER_AREA if (target_w <= src_w and target_h <= src_h) else cv2.INTER_LANCZOS4
    resized = cv2.resize(img, (target_w, target_h), interpolation=interp)
    blurred = cv2.GaussianBlur(resized, (0, 0), 1.0)
    sharp   = cv2.addWeighted(resized, 1.6, blurred, -0.6, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def _build_sticker_canvas(w, h, get_logo_fn):
    S  = 3
    cw, ch = w * S, h * S
    canvas = np.full((ch, cw, 3), 255, dtype=np.uint8)

    logo = get_logo_fn()
    px   = max(int(cw * 0.05), 4)
    py   = max(int(ch * 0.12), 3)
    lw   = max(1, cw - 2 * px)
    lh   = max(1, ch - 2 * py)
    canvas[py:py + lh, px:px + lw] = _sharp_resize(logo, lw, lh)

    b = max(2, S * 2)
    canvas[:b,  :]   = (150, 150, 150)
    canvas[-b:, :]   = (150, 150, 150)
    canvas[:,  :b]   = (150, 150, 150)
    canvas[:, -b:]   = (150, 150, 150)

    return _sharp_resize(canvas, w, h)


# ══════════════════════════════════════════════════════════════════════════════
# PERSPECTIVE WARP  — feathered composite
# ══════════════════════════════════════════════════════════════════════════════

def _warp_sticker_onto_plate(img, corners, get_logo_fn):
    out = img.copy()
    ih, iw = out.shape[:2]
    tl, tr, br, bl = corners.astype(np.float32)

    w_top   = float(np.linalg.norm(tr - tl))
    w_bot   = float(np.linalg.norm(br - bl))
    h_left  = float(np.linalg.norm(bl - tl))
    h_right = float(np.linalg.norm(br - tr))
    plate_w = max(1, int(round((w_top + w_bot) / 2.0)))
    plate_h = max(1, int(round((h_left + h_right) / 2.0)))

    sticker = _build_sticker_canvas(plate_w, plate_h, get_logo_fn)

    src_pts = np.array([
        [0,           0          ],
        [plate_w - 1, 0          ],
        [plate_w - 1, plate_h - 1],
        [0,           plate_h - 1],
    ], dtype=np.float32)
    dst_pts = np.array([tl, tr, br, bl], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    warped = cv2.warpPerspective(
        sticker, M, (iw, ih),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    # Hard mask
    hard_mask = np.zeros((ih, iw), dtype=np.uint8)
    poly = dst_pts.astype(np.int32).reshape((-1, 1, 2))
    cv2.fillConvexPoly(hard_mask, poly, 255)

    # Feathered mask
    k_size = max(3, int(min(plate_w, plate_h) * 0.04) | 1)
    feather = cv2.erode(hard_mask, np.ones((3, 3), np.uint8), iterations=2)
    feather = cv2.GaussianBlur(feather.astype(np.float32),
                               (k_size, k_size), sigmaX=3.0)
    feather = np.clip(feather / 255.0, 0.0, 1.0)[:, :, np.newaxis]

    out = (warped.astype(np.float32) * feather +
           out.astype(np.float32)    * (1.0 - feather)).astype(np.uint8)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def apply_logo_perspective(img, corners, plate_color, bbox_w, bbox_h,
                           get_logo_fn,
                           raw_x1=None, raw_y1=None, raw_x2=None, raw_y2=None):
    ih, iw = img.shape[:2]

    corners_c = _aspect_correct_corners(corners, bbox_w, bbox_h)
    corners_c[:, 0] = np.clip(corners_c[:, 0], 0, iw - 1)
    corners_c[:, 1] = np.clip(corners_c[:, 1], 0, ih - 1)

    w, h = _compute_plate_dims(corners_c)
    if w >= 4 and h >= 4:
        return _warp_sticker_onto_plate(img, corners_c, get_logo_fn)

    # Degenerate fallback
    out = img.copy()
    if raw_x1 is not None:
        x1, y1, x2, y2 = int(raw_x1), int(raw_y1), int(raw_x2), int(raw_y2)
    else:
        tl, tr, br, bl = corners_c
        x1 = int(min(tl[0], bl[0])); y1 = int(min(tl[1], tr[1]))
        x2 = int(max(tr[0], br[0])); y2 = int(max(bl[1], br[1]))

    pad_w = int((x2 - x1) * PAD_W_FRAC)
    pad_h = int((y2 - y1) * PAD_H_FRAC)
    x1 = max(0,  x1 - pad_w); y1 = max(0,  y1 - pad_h)
    x2 = min(iw, x2 + pad_w); y2 = min(ih, y2 + pad_h)
    pw = max(10, x2 - x1);    ph = max(5,  y2 - y1)
    sticker = _build_sticker_canvas(pw, ph, get_logo_fn)
    out[y1:y1 + ph, x1:x1 + pw] = sticker
    return out


# ══════════════════════════════════════════════════════════════════════════════
# DEBUG OVERLAY
# ══════════════════════════════════════════════════════════════════════════════

def draw_debug_overlay(img, alpr_bbox, refined_contour_pts, final_quad,
                       grown_bbox=None):
    out = img.copy()

    x1, y1, x2, y2 = [int(v) for v in alpr_bbox]
    cv2.rectangle(out, (x1,y1),(x2,y2),(255,80,0),2,cv2.LINE_AA)
    cv2.putText(out,'ALPR',(x1,max(y1-6,10)),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,80,0),2,cv2.LINE_AA)

    if grown_bbox is not None:
        gx1,gy1,gx2,gy2 = [int(v) for v in grown_bbox]
        cv2.rectangle(out,(gx1,gy1),(gx2,gy2),(255,220,0),2,cv2.LINE_AA)
        cv2.putText(out,'GROWN',(gx1,max(gy1-6,10)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,220,0),1,cv2.LINE_AA)

    if refined_contour_pts is not None:
        pts = refined_contour_pts.astype(np.int32).reshape((-1,1,2))
        cv2.polylines(out,[pts],True,(0,210,0),2,cv2.LINE_AA)

    if final_quad is not None:
        pts = final_quad.astype(np.int32).reshape((-1,1,2))
        cv2.polylines(out,[pts],True,(0,0,230),2,cv2.LINE_AA)
        labels = ['TL','TR','BR','BL']
        for i,(pt,lbl) in enumerate(zip(final_quad.astype(int),labels)):
            cv2.circle(out,tuple(pt),6,(0,0,230),-1,cv2.LINE_AA)
            cv2.putText(out,lbl,(pt[0]+7,pt[1]-4),
                        cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,230),1,cv2.LINE_AA)
    return out
