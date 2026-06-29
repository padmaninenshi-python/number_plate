"""
PlateVision v11 — Spinny/Cars24-style exact plate corner detection.

COMPLETE REWRITE of find_plate_corners():
  Old approach (v9/v10): OTSU or color mask → contours → minAreaRect
  Problem: White car body bleeds into white plate mask. Contours are noisy.

  New approach (v11): Hough line-based plate edge detection
  1. Expand ALPR bbox by 40% on all sides
  2. Edge map (CLAHE + Canny) on the expanded crop
  3. Hough lines → cluster into 4 dominant line groups (top/bottom/left/right)
  4. Intersect lines → 4 corner points
  5. Validate aspect ratio + size
  6. If Hough fails → color-contour fallback → ALPR bbox fallback

  Also fixed _grow_bbox_to_plate() which was a stub in v9.
"""

import cv2
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

PLATE_STD_ASPECT  = 3.33
PLATE_HSRP_ASPECT = 4.17
PLATE_MIN_ASPECT  = 1.8

QUAD_MIN_ASPECT = 1.2
QUAD_MAX_ASPECT = 8.0

PAD_W_FRAC = 0.0
PAD_H_FRAC = 0.0

GROW_MAX_FRAC = 0.40


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


def _line_intersection(l1, l2):
    """Intersect two lines given as (rho, theta). Returns (x, y) or None."""
    rho1, theta1 = l1
    rho2, theta2 = l2
    a = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)],
    ])
    b = np.array([rho1, rho2])
    det = a[0,0]*a[1,1] - a[0,1]*a[1,0]
    if abs(det) < 1e-6:
        return None
    x = (b[0]*a[1,1] - b[1]*a[0,1]) / det
    y = (b[1]*a[0,0] - b[0]*a[1,0]) / det
    return (x, y)


# ══════════════════════════════════════════════════════════════════════════════
# GROW BBOX
# ══════════════════════════════════════════════════════════════════════════════

def _grow_bbox_to_plate(img, x1, y1, x2, y2, plate_color):
    """
    Walk outward from ALPR bbox using plate color density to find true edges.
    """
    ih, iw = img.shape[:2]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    bw = x2 - x1
    bh = y2 - y1

    max_exp_x = int(bw * GROW_MAX_FRAC)
    max_exp_y = int(bh * GROW_MAX_FRAC)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if plate_color == 'yellow':
        mask = cv2.inRange(hsv, np.array([12, 60, 80]), np.array([40, 255, 255]))
    else:
        white  = cv2.inRange(hsv, np.array([0,   0, 180]), np.array([180, 50, 255]))
        slight = cv2.inRange(hsv, np.array([14, 15, 165]), np.array([38, 80, 255]))
        mask   = cv2.bitwise_or(white, slight)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    DENSITY_THRESH = 0.22

    new_x1 = x1
    for step in range(1, max_exp_x + 1):
        col = max(0, x1 - step)
        strip = mask[y1:y2, col:col+1]
        if strip.size == 0 or np.mean(strip)/255.0 < DENSITY_THRESH: break
        new_x1 = col

    new_x2 = x2
    for step in range(1, max_exp_x + 1):
        col = min(iw-1, x2+step)
        strip = mask[y1:y2, col:col+1]
        if strip.size == 0 or np.mean(strip)/255.0 < DENSITY_THRESH: break
        new_x2 = col

    new_y1 = y1
    for step in range(1, max_exp_y + 1):
        row = max(0, y1-step)
        strip = mask[row:row+1, x1:x2]
        if strip.size == 0 or np.mean(strip)/255.0 < DENSITY_THRESH: break
        new_y1 = row

    new_y2 = y2
    for step in range(1, max_exp_y + 1):
        row = min(ih-1, y2+step)
        strip = mask[row:row+1, x1:x2]
        if strip.size == 0 or np.mean(strip)/255.0 < DENSITY_THRESH: break
        new_y2 = row

    return (max(0, new_x1), max(0, new_y1), min(iw, new_x2), min(ih, new_y2))


# ══════════════════════════════════════════════════════════════════════════════
# PLATE COLOR
# ══════════════════════════════════════════════════════════════════════════════

def detect_plate_color(img, x1, y1, x2, y2):
    ih, iw = img.shape[:2]
    pw, ph = x2-x1, y2-y1
    sx1 = int(max(0,  x1+pw*0.15)); sy1 = int(max(0,  y1+ph*0.15))
    sx2 = int(min(iw, x2-pw*0.15)); sy2 = int(min(ih, y2-ph*0.15))
    crop = img[sy1:sy2, sx1:sx2]
    if crop.size == 0: return 'white'
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    strict  = cv2.inRange(hsv, np.array([18,130,130]), np.array([34,255,255]))
    if np.mean(strict)/255.0 > 0.20: return 'yellow'
    relaxed = cv2.inRange(hsv, np.array([12, 70, 90]), np.array([40,255,255]))
    if np.mean(relaxed)/255.0 > 0.28: return 'yellow'
    return 'white'


# ══════════════════════════════════════════════════════════════════════════════
# COLOR MASK
# ══════════════════════════════════════════════════════════════════════════════

def _color_plate_mask(crop, plate_hint='white'):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    if plate_hint == 'yellow':
        mask = cv2.inRange(hsv, np.array([12,60,80]), np.array([40,255,255]))
    else:
        white  = cv2.inRange(hsv, np.array([0,  0,180]), np.array([180,50,255]))
        slight = cv2.inRange(hsv, np.array([14,15,165]), np.array([38,80,255]))
        mask   = cv2.bitwise_or(white, slight)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    return mask


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def _is_valid_plate_quad(pts, bbox_area):
    area = _quad_area(pts)
    if area < bbox_area * 0.18 or area > bbox_area * 5.0: return False
    w, h = _compute_plate_dims(pts)
    if h < 4 or w < 4: return False
    aspect = w / h
    if not (QUAD_MIN_ASPECT <= aspect <= QUAD_MAX_ASPECT): return False
    if not _is_convex(pts): return False
    return True


def _clamp_to_region(quad, x1, y1, x2, y2):
    out = quad.copy()
    out[:,0] = np.clip(out[:,0], x1, x2)
    out[:,1] = np.clip(out[:,1], y1, y2)
    return out.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# CONTOUR QUAD (fallback)
# ══════════════════════════════════════════════════════════════════════════════

def _contour_quad(crop, cx1, cy1, bbox_area):
    ch, cw = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    def _edges(g):
        blurred = cv2.GaussianBlur(g, (5,5), 0)
        median  = float(np.median(blurred))
        lo = max(15, int(0.50*median)); hi = min(220, int(1.40*median))
        e  = cv2.Canny(blurred, lo, hi)
        return cv2.dilate(e, np.ones((3,3),np.uint8), iterations=1)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    for g in [gray, clahe.apply(gray)]:
        edges = _edges(g)
        cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: continue
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for cnt in cnts[:12]:
            if cv2.contourArea(cnt) < ch*cw*0.06: break
            peri = cv2.arcLength(cnt, True)
            for eps in [0.01,0.02,0.03,0.04,0.05,0.06,0.08]:
                approx = cv2.approxPolyDP(cnt, eps*peri, True)
                if len(approx) == 4:
                    pts = approx.reshape(4,2).astype(np.float32)
                    pts[:,0] += cx1; pts[:,1] += cy1
                    pts = _order_corners(pts)
                    if _is_valid_plate_quad(pts, bbox_area): return pts
    return None


# ══════════════════════════════════════════════════════════════════════════════
# COLOR QUAD (fallback)
# ══════════════════════════════════════════════════════════════════════════════

def _color_quad(crop, cx1, cy1, bbox_area, plate_hint='white'):
    mask = _color_plate_mask(crop, plate_hint)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < bbox_area * 0.15: return None
    peri = cv2.arcLength(cnt, True)
    for eps in [0.02,0.03,0.05,0.07,0.10]:
        approx = cv2.approxPolyDP(cnt, eps*peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4,2).astype(np.float32)
            pts[:,0] += cx1; pts[:,1] += cy1
            pts = _order_corners(pts)
            if _is_valid_plate_quad(pts, bbox_area): return pts
    rect = cv2.minAreaRect(cnt)
    box  = cv2.boxPoints(rect).astype(np.float32)
    box[:,0] += cx1; box[:,1] += cy1
    pts = _order_corners(box)
    if _is_valid_plate_quad(pts, bbox_area): return pts
    return None


def _minAreaRect_quad(crop, cx1, cy1, bbox_area):
    gray    = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe   = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    blurred = cv2.GaussianBlur(clahe.apply(gray), (5,5), 0)
    edges   = cv2.Canny(blurred, 15, 80)
    k = np.ones((3,3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=3)
    pts = cv2.findNonZero(edges)
    if pts is None or len(pts) < 20: return None
    rect = cv2.minAreaRect(pts)
    box  = cv2.boxPoints(rect).astype(np.float32)
    box[:,0] += cx1; box[:,1] += cy1
    ordered = _order_corners(box)
    if _is_valid_plate_quad(ordered, bbox_area): return ordered
    return None


def _expanded_bbox_quad(x1, y1, x2, y2, img_w, img_h):
    pw = x2-x1; ph = y2-y1
    pad_w = int(pw*PAD_W_FRAC); pad_h = int(ph*PAD_H_FRAC)
    nx1=max(0,x1-pad_w); ny1=max(0,y1-pad_h)
    nx2=min(img_w,x2+pad_w); ny2=min(img_h,y2+pad_h)
    return _bbox_corners(nx1,ny1,nx2,ny2)


# ══════════════════════════════════════════════════════════════════════════════
# HOUGH LINE CORNER DETECTOR  ← v11 NEW PRIMARY METHOD
# ══════════════════════════════════════════════════════════════════════════════

def _hough_plate_corners(img, x1, y1, x2, y2, plate_hint='white'):
    """
    Spinny/Cars24 style: find plate corners using Hough line detection.

    Strategy:
    1. Expand ALPR bbox 40%x / 50%y to see dark border frame around plate
    2. Build edge map: CLAHE grayscale + Canny (tight thresholds)
    3. Probabilistic Hough → line segments
    4. Separate into near-horizontal and near-vertical lines
    5. From horizontal lines, pick the best top-edge and bottom-edge lines
       (closest to ALPR top/bottom boundaries)
    6. From vertical lines, pick the best left-edge and right-edge lines
    7. Intersect the 4 lines → 4 corners
    8. Validate: aspect, size, inside expanded region
    """
    ih, iw = img.shape[:2]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    bw = x2 - x1
    bh = y2 - y1

    # Expand to see the dark bumper frame bordering the plate
    ex = int(bw * 0.45)
    ey = int(bh * 0.55)
    ex1 = max(0,  x1-ex);  ey1 = max(0,  y1-ey)
    ex2 = min(iw, x2+ex);  ey2 = min(ih, y2+ey)
    crop = img[ey1:ey2, ex1:ex2]
    if crop.size == 0: return None

    ch, cw = crop.shape[:2]

    # ALPR boundaries in crop coords (expected plate edges are near these)
    alpr_top    = y1 - ey1
    alpr_bottom = y2 - ey1
    alpr_left   = x1 - ex1
    alpr_right  = x2 - ex1

    # ── Edge map ─────────────────────────────────────────────────────────────
    gray  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    gray  = clahe.apply(gray)
    blur  = cv2.GaussianBlur(gray, (3,3), 0)

    # Adaptive thresholds based on image stats
    med = float(np.median(blur))
    lo  = max(20, int(med * 0.4))
    hi  = min(200, int(med * 1.2))
    edges = cv2.Canny(blur, lo, hi)

    # ── Hough lines ──────────────────────────────────────────────────────────
    min_line_len = int(min(bw, bh) * 0.4)
    lines = cv2.HoughLinesP(
        edges,
        rho=1, theta=np.pi/180,
        threshold=max(20, min_line_len//2),
        minLineLength=min_line_len,
        maxLineGap=int(min_line_len * 0.3),
    )
    if lines is None or len(lines) < 4:
        return None

    # ── Separate horizontal vs vertical ──────────────────────────────────────
    h_lines = []  # (y_mean, x1, y1, x2, y2, angle_deg)
    v_lines = []

    for seg in lines:
        x1s, y1s, x2s, y2s = seg[0]
        dx = x2s - x1s; dy = y2s - y1s
        length = (dx**2 + dy**2) ** 0.5
        if length < 1: continue
        angle = abs(np.degrees(np.arctan2(dy, dx)))

        if angle < 25 or angle > 155:  # near horizontal
            y_mean = (y1s + y2s) / 2.0
            h_lines.append((y_mean, x1s, y1s, x2s, y2s))
        elif 65 < angle < 115:          # near vertical
            x_mean = (x1s + x2s) / 2.0
            v_lines.append((x_mean, x1s, y1s, x2s, y2s))

    if len(h_lines) < 2 or len(v_lines) < 2:
        return None

    # ── Pick best top/bottom/left/right edge lines ───────────────────────────
    # "Best" = line segment closest to expected ALPR boundary,
    #           AND has significant horizontal/vertical extent

    def _fit_infinite_line(x1s, y1s, x2s, y2s):
        """Return (rho, theta) for infinite Hough line form."""
        dx = x2s - x1s; dy = y2s - y1s
        if dx == 0 and dy == 0: return None
        theta = np.arctan2(-dx, dy)  # perpendicular angle
        rho   = x1s * np.cos(theta) + y1s * np.sin(theta)
        return (rho, theta)

    # Top edge: horizontal line closest to alpr_top from above/at
    h_lines_sorted_top = sorted(h_lines, key=lambda l: abs(l[0] - alpr_top))
    # Bottom edge: closest to alpr_bottom
    h_lines_sorted_bot = sorted(h_lines, key=lambda l: abs(l[0] - alpr_bottom))
    # Left edge: vertical closest to alpr_left
    v_lines_sorted_lft = sorted(v_lines, key=lambda l: abs(l[0] - alpr_left))
    # Right edge: vertical closest to alpr_right
    v_lines_sorted_rgt = sorted(v_lines, key=lambda l: abs(l[0] - alpr_right))

    # Try combinations of top/bottom/left/right candidates
    best_corners = None
    best_score   = -1.0

    for ti in range(min(4, len(h_lines_sorted_top))):
        lt = h_lines_sorted_top[ti]
        line_top = _fit_infinite_line(lt[1], lt[2], lt[3], lt[4])
        if line_top is None: continue

        for bi in range(min(4, len(h_lines_sorted_bot))):
            if bi == ti: continue
            lb = h_lines_sorted_bot[bi]
            # Top must be above bottom
            if lb[0] <= lt[0]: continue
            line_bot = _fit_infinite_line(lb[1], lb[2], lb[3], lb[4])
            if line_bot is None: continue

            for li in range(min(4, len(v_lines_sorted_lft))):
                ll = v_lines_sorted_lft[li]
                line_lft = _fit_infinite_line(ll[1], ll[2], ll[3], ll[4])
                if line_lft is None: continue

                for ri in range(min(4, len(v_lines_sorted_rgt))):
                    if ri == li: continue
                    lr = v_lines_sorted_rgt[ri]
                    # Left must be left of right
                    if lr[0] <= ll[0]: continue
                    line_rgt = _fit_infinite_line(lr[1], lr[2], lr[3], lr[4])
                    if line_rgt is None: continue

                    # Intersect all pairs
                    tl_pt = _line_intersection(line_top, line_lft)
                    tr_pt = _line_intersection(line_top, line_rgt)
                    br_pt = _line_intersection(line_bot, line_rgt)
                    bl_pt = _line_intersection(line_bot, line_lft)
                    if None in (tl_pt, tr_pt, br_pt, bl_pt): continue

                    # Translate back to full image coords
                    corners_crop = np.array([tl_pt, tr_pt, br_pt, bl_pt], dtype=np.float32)
                    corners_img  = corners_crop.copy()
                    corners_img[:,0] += ex1
                    corners_img[:,1] += ey1
                    ordered = _order_corners(corners_img)

                    # Validate
                    w, h_dim = _compute_plate_dims(ordered)
                    if h_dim < 4 or w < 4: continue
                    aspect = w / h_dim
                    if not (QUAD_MIN_ASPECT <= aspect <= QUAD_MAX_ASPECT): continue
                    if not _is_convex(ordered): continue

                    # Must cover ALPR bbox reasonably
                    tl2, tr2, br2, bl2 = ordered
                    det_x1 = min(tl2[0], bl2[0])
                    det_y1 = min(tl2[1], tr2[1])
                    det_x2 = max(tr2[0], br2[0])
                    det_y2 = max(bl2[1], br2[1])
                    alpr_w = float(x2 - x1); alpr_h = float(y2 - y1)
                    if not (alpr_w*0.4 <= w <= alpr_w*2.5): continue
                    if not (alpr_h*0.4 <= h_dim <= alpr_h*2.5): continue

                    # Score: aspect closeness + size match
                    asp_score  = 1.0 - min(abs(aspect-PLATE_STD_ASPECT),
                                           abs(aspect-PLATE_HSRP_ASPECT)) / 5.0
                    size_score = 1.0 - abs(w - alpr_w) / (alpr_w + 1.0)
                    score = asp_score*0.6 + size_score*0.4

                    if score > best_score:
                        best_score   = score
                        best_corners = ordered

    return best_corners


# ══════════════════════════════════════════════════════════════════════════════
# MASTER CORNER FINDER  v11 — Hough first, then fallbacks
# ══════════════════════════════════════════════════════════════════════════════

def find_plate_corners(img, x1, y1, x2, y2, plate_hint='white'):
    """
    v11 detection cascade:
    1. Hough line intersection (best for angled plates, Spinny style)
    2. Color-contour minAreaRect (good for front-facing plates)
    3. Edge-contour minAreaRect (fallback)
    4. ALPR bbox rectangle (last resort)
    """
    ih, iw = img.shape[:2]
    x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
    bw = x2i - x1i
    bh = y2i - y1i
    bbox_area = float(bw * bh)

    # ── Pass 1: Hough lines ───────────────────────────────────────────────────
    hough = _hough_plate_corners(img, x1i, y1i, x2i, y2i, plate_hint)
    if hough is not None:
        return hough

    # ── Pass 2: color-contour on expanded crop ────────────────────────────────
    ex = int(bw * 0.30);  ey = int(bh * 0.40)
    cx1 = max(0,  x1i-ex); cy1 = max(0,  y1i-ey)
    cx2 = min(iw, x2i+ex); cy2 = min(ih, y2i+ey)
    crop = img[cy1:cy2, cx1:cx2]

    if crop.size > 0:
        # Color mask → minAreaRect on largest contour
        mask = _color_plate_mask(crop, plate_hint)
        kf = cv2.getStructuringElement(cv2.MORPH_RECT, (9,7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kf, iterations=4)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kf, iterations=1)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if cnts:
            alpr_cx_crop = (x1i+x2i)/2.0 - cx1
            alpr_cy_crop = (y1i+y2i)/2.0 - cy1
            best_cnt  = None
            best_sc   = -1.0
            for cnt in cnts:
                area = cv2.contourArea(cnt)
                if area < bbox_area*0.20 or area > bbox_area*5.0: continue
                M = cv2.moments(cnt)
                if M['m00'] == 0: continue
                ccx = M['m10']/M['m00']; ccy = M['m01']/M['m00']
                dist = ((ccx-alpr_cx_crop)**2 + (ccy-alpr_cy_crop)**2)**0.5
                max_dist = ((bw**2+bh**2)**0.5)*0.6
                if dist > max_dist: continue
                rect = cv2.minAreaRect(cnt)
                rw, rh = rect[1]
                if rh == 0 or rw == 0: continue
                aspect = max(rw,rh)/min(rw,rh)
                if not (1.5 <= aspect <= 7.0): continue
                asp_sc = 1.0 - min(abs(aspect-PLATE_STD_ASPECT),
                                   abs(aspect-PLATE_HSRP_ASPECT))/5.0
                prx_sc = 1.0 - dist/(max_dist+1.0)
                sc = asp_sc*0.55 + prx_sc*0.45
                if sc > best_sc:
                    best_sc = sc; best_cnt = (cnt, rect)

            if best_cnt:
                cnt, rect = best_cnt
                box = cv2.boxPoints(rect).astype(np.float32)
                box[:,0] += cx1; box[:,1] += cy1
                ordered = _order_corners(box)
                w, h = _compute_plate_dims(ordered)
                if (h >= 4 and w >= 4 and
                    bw*0.40 <= w <= bw*2.4 and
                    bh*0.40 <= h <= bh*2.4):
                    return ordered

    # ── Pass 3: edge contour minAreaRect ──────────────────────────────────────
    if crop.size > 0:
        mar = _minAreaRect_quad(crop, cx1, cy1, bbox_area)
        if mar is not None:
            w, h = _compute_plate_dims(mar)
            if (h >= 4 and w >= 4 and
                bw*0.40 <= w <= bw*2.4 and
                bh*0.40 <= h <= bh*2.4):
                return mar

    # ── Pass 4: ALPR bbox ─────────────────────────────────────────────────────
    return _bbox_corners(x1i, y1i, x2i, y2i)


# ══════════════════════════════════════════════════════════════════════════════
# ASPECT CORRECTION
# ══════════════════════════════════════════════════════════════════════════════

def _aspect_correct_corners(corners, bbox_w, bbox_h):
    plate_w, plate_h = _compute_plate_dims(corners)
    if plate_h < 1 or plate_w < 1: return corners
    detected_aspect = plate_w / plate_h
    if detected_aspect >= 2.0: return corners
    target_h  = plate_w / PLATE_STD_ASPECT
    scale     = target_h / plate_h
    center    = corners.mean(axis=0)
    width_dir = corners[1] - corners[0]
    wn = np.linalg.norm(width_dir)
    if wn < 1: return corners
    width_dir  = width_dir / wn
    height_dir = np.array([width_dir[1], -width_dir[0]], dtype=np.float32)
    corrected  = []
    for pt in corners:
        v      = pt - center
        w_comp = float(np.dot(v, width_dir))
        h_comp = float(np.dot(v, height_dir))
        corrected.append(center + w_comp*width_dir + (h_comp*scale)*height_dir)
    return np.array(corrected, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# STICKER CANVAS
# ══════════════════════════════════════════════════════════════════════════════

def _sharp_resize(img, target_w, target_h):
    target_w, target_h = max(1,target_w), max(1,target_h)
    src_h, src_w = img.shape[:2]
    interp  = cv2.INTER_AREA if (target_w<=src_w and target_h<=src_h) else cv2.INTER_LANCZOS4
    resized = cv2.resize(img, (target_w,target_h), interpolation=interp)
    blurred = cv2.GaussianBlur(resized, (0,0), 1.0)
    sharp   = cv2.addWeighted(resized, 1.6, blurred, -0.6, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def _build_sticker_canvas(w, h, get_logo_fn):
    S  = 3
    cw, ch = w*S, h*S
    canvas = np.full((ch,cw,3), 255, dtype=np.uint8)
    logo   = get_logo_fn()
    px = max(int(cw*0.05),4); py = max(int(ch*0.12),3)
    lw = max(1,cw-2*px);      lh = max(1,ch-2*py)
    canvas[py:py+lh, px:px+lw] = _sharp_resize(logo, lw, lh)
    b = max(2, S*2)
    canvas[:b,:]   = (150,150,150)
    canvas[-b:,:]  = (150,150,150)
    canvas[:,:b]   = (150,150,150)
    canvas[:,-b:]  = (150,150,150)
    return _sharp_resize(canvas, w, h)


# ══════════════════════════════════════════════════════════════════════════════
# PERSPECTIVE WARP — feathered
# ══════════════════════════════════════════════════════════════════════════════

def _warp_sticker_onto_plate(img, corners, get_logo_fn):
    out = img.copy()
    ih, iw = out.shape[:2]
    tl, tr, br, bl = corners.astype(np.float32)
    w_top   = float(np.linalg.norm(tr-tl))
    w_bot   = float(np.linalg.norm(br-bl))
    h_left  = float(np.linalg.norm(bl-tl))
    h_right = float(np.linalg.norm(br-tr))
    plate_w = max(1, int(round((w_top+w_bot)/2.0)))
    plate_h = max(1, int(round((h_left+h_right)/2.0)))

    sticker = _build_sticker_canvas(plate_w, plate_h, get_logo_fn)
    src_pts = np.array([[0,0],[plate_w-1,0],[plate_w-1,plate_h-1],[0,plate_h-1]], dtype=np.float32)
    dst_pts = np.array([tl,tr,br,bl], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped  = cv2.warpPerspective(sticker, M, (iw,ih),
                                  flags=cv2.INTER_LANCZOS4,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(255,255,255))

    hard_mask = np.zeros((ih,iw), dtype=np.uint8)
    poly = dst_pts.astype(np.int32).reshape((-1,1,2))
    cv2.fillConvexPoly(hard_mask, poly, 255)

    k_size  = max(3, int(min(plate_w,plate_h)*0.04)|1)
    feather = cv2.erode(hard_mask, np.ones((3,3),np.uint8), iterations=2)
    feather = cv2.GaussianBlur(feather.astype(np.float32), (k_size,k_size), sigmaX=3.0)
    feather = np.clip(feather/255.0, 0.0, 1.0)[:,:,np.newaxis]

    out = (warped.astype(np.float32)*feather +
           out.astype(np.float32)*(1.0-feather)).astype(np.uint8)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def apply_logo_perspective(img, corners, plate_color, bbox_w, bbox_h,
                           get_logo_fn,
                           raw_x1=None, raw_y1=None, raw_x2=None, raw_y2=None):
    ih, iw = img.shape[:2]
    corners_c = _aspect_correct_corners(corners, bbox_w, bbox_h)
    corners_c[:,0] = np.clip(corners_c[:,0], 0, iw-1)
    corners_c[:,1] = np.clip(corners_c[:,1], 0, ih-1)
    w, h = _compute_plate_dims(corners_c)
    if w >= 4 and h >= 4:
        return _warp_sticker_onto_plate(img, corners_c, get_logo_fn)

    # Degenerate fallback
    out = img.copy()
    if raw_x1 is not None:
        x1,y1,x2,y2 = int(raw_x1),int(raw_y1),int(raw_x2),int(raw_y2)
    else:
        tl,tr,br,bl = corners_c
        x1=int(min(tl[0],bl[0])); y1=int(min(tl[1],tr[1]))
        x2=int(max(tr[0],br[0])); y2=int(max(bl[1],br[1]))
    pw = max(10,x2-x1); ph = max(5,y2-y1)
    sticker = _build_sticker_canvas(pw, ph, get_logo_fn)
    out[y1:y1+ph, x1:x1+pw] = sticker
    return out


# ══════════════════════════════════════════════════════════════════════════════
# DEBUG OVERLAY
# ══════════════════════════════════════════════════════════════════════════════

def draw_debug_overlay(img, alpr_bbox, refined_contour_pts, final_quad,
                       grown_bbox=None):
    out = img.copy()
    x1,y1,x2,y2 = [int(v) for v in alpr_bbox]
    cv2.rectangle(out,(x1,y1),(x2,y2),(255,80,0),2,cv2.LINE_AA)
    cv2.putText(out,'ALPR',(x1,max(y1-6,10)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,80,0),2,cv2.LINE_AA)
    if grown_bbox is not None:
        gx1,gy1,gx2,gy2 = [int(v) for v in grown_bbox]
        cv2.rectangle(out,(gx1,gy1),(gx2,gy2),(255,220,0),2,cv2.LINE_AA)
        cv2.putText(out,'GROWN',(gx1,max(gy1-6,10)),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,220,0),1,cv2.LINE_AA)
    if refined_contour_pts is not None:
        pts = refined_contour_pts.astype(np.int32).reshape((-1,1,2))
        cv2.polylines(out,[pts],True,(0,210,0),2,cv2.LINE_AA)
    if final_quad is not None:
        pts = final_quad.astype(np.int32).reshape((-1,1,2))
        cv2.polylines(out,[pts],True,(0,0,230),2,cv2.LINE_AA)
        labels = ['TL','TR','BR','BL']
        for pt,lbl in zip(final_quad.astype(int),labels):
            cv2.circle(out,tuple(pt),6,(0,0,230),-1,cv2.LINE_AA)
            cv2.putText(out,lbl,(pt[0]+7,pt[1]-4),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,230),1,cv2.LINE_AA)
    return out
