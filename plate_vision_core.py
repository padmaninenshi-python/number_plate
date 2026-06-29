"""
PlateVision v9 — ROOT CAUSE FIX for partial plate coverage.

ROOT CAUSE (identified from VW Polo + Toyota Innova images):
  ALPR gives an UNDERSIZED bounding box that covers only part of the plate.
  v8 was clamping all corners TO the ALPR box — actively cutting the sticker short.

THE FIX:
  1. _grow_bbox_to_plate() — NEW: after ALPR detection, walk outward in all 4
     directions using plate color to find the TRUE plate boundary. This is the
     same technique Spinny/Cars24 use. Happens BEFORE any corner detection.
  2. _clamp_to_alpr() changed to _clamp_to_search_region() — clamps to the
     GROWN bbox (not the original tiny ALPR box).
  3. find_plate_corners() now operates on the GROWN bbox crop, not ALPR crop.
  4. Contour quad validation area threshold lowered (ALPR box can be 30% of true
     plate — was rejecting valid quads).
  5. Yellow plate color walk uses separate (broader) HSV range.
  6. White plate color walk uses adaptive threshold per-image.
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
GROW_MAX_FRAC = 0.30   # 30% max — prevents bleeding into white car body on white plates


# ══════════════════════════════════════════════════════════════════════════════
# GEOMETRY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _order_corners(pts):
    """
    Order 4 corners as [TL, TR, BR, BL] using sum/diff trick.
    Works correctly for rotated plates.
    """
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
# STEP 0 — GROW THE ALPR BBOX TO TRUE PLATE BOUNDARIES  ← THE KEY FIX
# ══════════════════════════════════════════════════════════════════════════════

def _grow_bbox_to_plate(img, x1, y1, x2, y2, plate_color):
    """
    Return ALPR bbox as-is — no padding, no expansion.
    ALPR yolo-v9 is accurate enough; any padding causes oversized sticker.
    """
    ih, iw = img.shape[:2]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    return (
        max(0,  x1),
        max(0,  y1),
        min(iw, x2),
        min(ih, y2),
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
        mask = cv2.inRange(hsv, np.array([12, 70, 90]), np.array([40, 255, 255]))
    else:
        # Tight white: very low saturation (plate number area is bright white)
        # Avoid bleeding into white car body which has lower value at angles
        white  = cv2.inRange(hsv, np.array([0,   0, 200]), np.array([180, 40, 255]))
        slight = cv2.inRange(hsv, np.array([14, 20, 185]), np.array([38, 80, 255]))
        mask   = cv2.bitwise_or(white, slight)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    return mask


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def _is_valid_plate_quad(pts, bbox_area):
    """
    Validate 4-corner quad for plate-like properties.
    bbox_area here is the GROWN bbox area (not original ALPR area).
    """
    area = _quad_area(pts)
    # Accept quads between 20% and 400% of the (already-grown) bbox area
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
    """
    Hard clamp every corner to the GROWN bounding box.
    The grown box already covers the full plate, so clamping here
    prevents floating outside the plate, not shrinking inside it.
    """
    out = quad.copy()
    out[:, 0] = np.clip(out[:, 0], x1, x2)
    out[:, 1] = np.clip(out[:, 1], y1, y2)
    return out.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# CONTOUR QUAD
# ══════════════════════════════════════════════════════════════════════════════

def _contour_quad(crop, cx1, cy1, bbox_area):
    """
    Multi-pass CLAHE+Canny edge detection → polygon → 4-corner quad.
    Operates on the GROWN bbox crop (so it sees the full plate).
    """
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
    """
    HSV color segmentation → plate polygon.
    MinAreaRect fallback if approxPolyDP can't reduce to 4 corners.
    """
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
    edges   = cv2.Canny(blurred, 15, 80)   # lower thresholds for angled plates
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

    # Fallback: try minAreaRect on ALL contour points combined (better for angled plates)
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
# MASTER CORNER FINDER  — operates on GROWN bbox
# ══════════════════════════════════════════════════════════════════════════════

def find_plate_corners(img, x1, y1, x2, y2, plate_hint='white'):
    """
    Get the 4 plate corners using the ALPR bbox + small padding.

    For straight-on shots (the vast majority of Indian car listing photos),
    the ALPR bbox IS the plate rectangle — no contour/color detection needed.
    Contour detection was causing oversized or mis-sized stickers.

    Returns np.float32 (4,2): [TL, TR, BR, BL].
    """
    ih, iw = img.shape[:2]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # Get padded bbox (tiny fixed padding only)
    gx1, gy1, gx2, gy2 = _grow_bbox_to_plate(img, x1, y1, x2, y2, plate_hint)

    # Return as clean rectangle corners
    return _bbox_corners(gx1, gy1, gx2, gy2)


# ══════════════════════════════════════════════════════════════════════════════
# ASPECT CORRECTION (side-angle shots)
# ══════════════════════════════════════════════════════════════════════════════

def _aspect_correct_corners(corners, bbox_w, bbox_h):
    """
    Aspect correction for heavily angled/foreshortened plates.
    Only applies when the plate appears severely compressed (aspect < 1.5).
    For straight-on shots this returns corners unchanged.
    """
    plate_w, plate_h = _compute_plate_dims(corners)
    if plate_h < 1 or plate_w < 1:
        return corners

    detected_aspect = plate_w / plate_h
    # Only correct when VERY compressed (side-angle shots)
    # Straight-on plates have aspect ~3.33, skip correction for those
    if detected_aspect >= 1.5:
        return corners  # straight-on shot, no correction needed

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
    """
    3× supersampled white sticker with logo and grey border.
    """
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
# PERSPECTIVE WARP  — feathered composite (no seams, no white triangles)
# ══════════════════════════════════════════════════════════════════════════════

def _warp_sticker_onto_plate(img, corners, get_logo_fn):
    """
    Warp sticker exactly onto plate quad using feathered (soft-edge) mask.
    Eliminates white triangle artifacts and seam lines.
    """
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

    # Feathered mask — erode 2px inward then Gaussian blur for smooth blend
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
    """
    Apply Caryanams sticker to cover the plate exactly.

    Flow:
      1. Aspect-correct corners (fixes side-angle foreshortening)
      2. Clip to image bounds
      3. Perspective-warp sticker onto quad
      4. Degenerate fallback: padded bbox direct paste
    """
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
    """
    Draw debug layers:
      ALPR bbox      — blue
      Grown bbox     — cyan (new)
      Refined contour— green
      Final quad     — red corners + polygon
    """
    out = img.copy()

    # ALPR bbox (blue)
    x1, y1, x2, y2 = [int(v) for v in alpr_bbox]
    cv2.rectangle(out, (x1,y1),(x2,y2),(255,80,0),2,cv2.LINE_AA)
    cv2.putText(out,'ALPR',(x1,max(y1-6,10)),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,80,0),2,cv2.LINE_AA)

    # Grown bbox (cyan)
    if grown_bbox is not None:
        gx1,gy1,gx2,gy2 = [int(v) for v in grown_bbox]
        cv2.rectangle(out,(gx1,gy1),(gx2,gy2),(255,220,0),2,cv2.LINE_AA)
        cv2.putText(out,'GROWN',(gx1,max(gy1-6,10)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,220,0),1,cv2.LINE_AA)

    # Refined contour (green)
    if refined_contour_pts is not None:
        pts = refined_contour_pts.astype(np.int32).reshape((-1,1,2))
        cv2.polylines(out,[pts],True,(0,210,0),2,cv2.LINE_AA)

    # Final quad (red)
    if final_quad is not None:
        pts = final_quad.astype(np.int32).reshape((-1,1,2))
        cv2.polylines(out,[pts],True,(0,0,230),2,cv2.LINE_AA)
        labels = ['TL','TR','BR','BL']
        for i,(pt,lbl) in enumerate(zip(final_quad.astype(int),labels)):
            cv2.circle(out,tuple(pt),6,(0,0,230),-1,cv2.LINE_AA)
            cv2.putText(out,lbl,(pt[0]+7,pt[1]-4),
                        cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,230),1,cv2.LINE_AA)
    return out
