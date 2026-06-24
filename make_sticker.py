"""
make_sticker.py — Numberplate-size logo sticker generator
==========================================================

Kya karta hai:
  - Diye gaye logo (caryanams_logo.png) ko Indian numberplate ki exact
    pixel size mein fit karke sticker-ready PNG banata hai.
  - Logo plate ke andar centered + padded set hota hai, bilkul waisa
    jaise app.py ka perspective warp karta hai — so sticker aur
    detected plate dono consistent dikhte hain.

Indian numberplate standard sizes:
  Type                  W × H (mm)    Common pixel (96 dpi)
  ──────────────────────────────────  ─────────────────────
  Standard car/bike     340 × 200 *   1338 × 787
  High-security (HSRP)  500 × 120     1969 × 472
  Two-wheeler           200 × 100     787 × 394

  * 340×200 mm is the most common visible size on a passenger car;
    340×150 mm is also widely used.

Usage:
  python make_sticker.py                          # default: all three
  python make_sticker.py --type car               # 340×200 car plate
  python make_sticker.py --type hsrp              # 500×120 HSRP plate
  python make_sticker.py --type bike              # 200×100 two-wheeler
  python make_sticker.py --dpi 300                # higher DPI output
  python make_sticker.py --logo path/to/logo.png  # custom logo
  python make_sticker.py --outdir ./stickers      # custom output folder
"""

import argparse
import os
import sys
import cv2
import numpy as np

# ─── Plate specifications ─────────────────────────────────────────────────────

PLATE_SPECS = {
    "car": {
        "label"       : "Standard Car (340×200 mm)",
        "width_mm"    : 340,
        "height_mm"   : 200,
        "bg_color"    : (255, 255, 255),   # white background
        "border_color": (0, 0, 0),         # black border
        "border_px"   : 6,
        "filename"    : "sticker_car_340x200.png",
    },
    "hsrp": {
        "label"       : "HSRP / Long plate (500×120 mm)",
        "width_mm"    : 500,
        "height_mm"   : 120,
        "bg_color"    : (255, 255, 255),
        "border_color": (0, 0, 0),
        "border_px"   : 6,
        "filename"    : "sticker_hsrp_500x120.png",
    },
    "bike": {
        "label"       : "Two-Wheeler (200×100 mm)",
        "width_mm"    : 200,
        "height_mm"   : 100,
        "bg_color"    : (255, 255, 255),
        "border_color": (0, 0, 0),
        "border_px"   : 4,
        "filename"    : "sticker_bike_200x100.png",
    },
}

# Default DPI for pixel output (96 = screen; 300 = print quality)
DEFAULT_DPI = 150

# ─── Logo loading (same logic as app.py get_logo()) ──────────────────────────

def load_logo_tight(logo_path: str) -> np.ndarray:
    """
    Load logo and return tightly cropped content on white background.
    Handles RGBA, RGB, and grayscale inputs.
    """
    raw = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Cannot read logo: {logo_path}")

    # Flatten alpha onto white background
    if raw.ndim == 2:
        bgr = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    elif raw.shape[2] == 4:
        bgr = raw[:, :, :3].astype(float)
        alpha = (raw[:, :, 3] / 255.0)[:, :, np.newaxis]
        bgr = (bgr * alpha + 255.0 * (1 - alpha)).astype(np.uint8)
    else:
        bgr = raw[:, :, :3]

    # Tight crop around non-white content
    gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask  = gray < 235
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return bgr   # all white — return as-is

    margin = max(4, int(min(bgr.shape[:2]) * 0.02))
    y1 = max(0, ys.min() - margin)
    y2 = min(bgr.shape[0], ys.max() + margin + 1)
    x1 = max(0, xs.min() - margin)
    x2 = min(bgr.shape[1], xs.max() + margin + 1)
    return bgr[y1:y2, x1:x2]


# ─── Sharp resize (same as app.py) ───────────────────────────────────────────

def sharp_resize(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """
    INTER_AREA for shrinking (better for text), INTER_LANCZOS4 for upscaling,
    followed by unsharp mask to recover perceived crispness.
    """
    target_w = max(1, target_w)
    target_h = max(1, target_h)
    src_h, src_w = img.shape[:2]
    shrinking = (target_w <= src_w) and (target_h <= src_h)
    interp    = cv2.INTER_AREA if shrinking else cv2.INTER_LANCZOS4
    resized   = cv2.resize(img, (target_w, target_h), interpolation=interp)

    blurred   = cv2.GaussianBlur(resized, (0, 0), 1.0)
    sharpened = cv2.addWeighted(resized, 1.6, blurred, -0.6, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


# ─── Core sticker generator ───────────────────────────────────────────────────

def make_sticker(
    logo_path : str,
    spec      : dict,
    dpi       : int,
    padding_pct: float = 0.07,   # padding inside plate as fraction of plate height
) -> np.ndarray:
    """
    Create a sticker image the exact pixel size of the given plate spec.

    Parameters
    ----------
    logo_path   : path to the logo PNG/JPG
    spec        : one entry from PLATE_SPECS
    dpi         : output resolution (dots per inch)
    padding_pct : fraction of plate height used as padding on all sides

    Returns
    -------
    np.ndarray  BGR image ready to save with cv2.imwrite
    """
    mm_per_inch = 25.4
    plate_w_px  = int(round(spec["width_mm"]  / mm_per_inch * dpi))
    plate_h_px  = int(round(spec["height_mm"] / mm_per_inch * dpi))

    # ── Canvas: white plate background ────────────────────────────────
    canvas = np.full((plate_h_px, plate_w_px, 3),
                     spec["bg_color"][::-1],   # RGB→BGR
                     dtype=np.uint8)

    # ── Border ────────────────────────────────────────────────────────
    bpx = spec["border_px"]
    if bpx > 0:
        cv2.rectangle(canvas, (0, 0), (plate_w_px - 1, plate_h_px - 1),
                      spec["border_color"][::-1], bpx)

    # ── Logo placement ─────────────────────────────────────────────────
    logo = load_logo_tight(logo_path)

    # Usable inner area (after border + padding)
    pad_y = max(bpx + 2, int(plate_h_px * padding_pct))
    pad_x = max(bpx + 2, int(plate_w_px * padding_pct * 0.5))

    inner_w = plate_w_px - 2 * pad_x
    inner_h = plate_h_px - 2 * pad_y

    if inner_w < 1 or inner_h < 1:
        return canvas   # plate too tiny for any logo

    # Scale logo to fit inside inner area, preserving aspect ratio
    logo_h, logo_w = logo.shape[:2]
    scale = min(inner_w / logo_w, inner_h / logo_h)
    fit_w = max(1, int(logo_w * scale))
    fit_h = max(1, int(logo_h * scale))

    resized_logo = sharp_resize(logo, fit_w, fit_h)

    # Center in inner area
    offset_x = pad_x + (inner_w - fit_w) // 2
    offset_y = pad_y + (inner_h - fit_h) // 2

    canvas[offset_y:offset_y + fit_h,
           offset_x:offset_x + fit_w] = resized_logo

    return canvas


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_logo = os.path.join(here, "static", "img", "caryanams_logo.png")

    parser = argparse.ArgumentParser(
        description="Generate numberplate-sized logo stickers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--logo",   default=default_logo,
                        help="Path to logo image (default: caryanams_logo.png)")
    parser.add_argument("--type",   choices=list(PLATE_SPECS.keys()) + ["all"],
                        default="all",
                        help="Plate type to generate (default: all)")
    parser.add_argument("--dpi",    type=int, default=DEFAULT_DPI,
                        help=f"Output DPI (default: {DEFAULT_DPI})")
    parser.add_argument("--outdir", default=here,
                        help="Output directory (default: same as script)")
    parser.add_argument("--padding", type=float, default=0.07,
                        help="Padding fraction (default: 0.07 = 7%% of plate height)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    types_to_make = list(PLATE_SPECS.keys()) if args.type == "all" else [args.type]

    print(f"\n  Logo      : {args.logo}")
    print(f"  DPI       : {args.dpi}")
    print(f"  Output dir: {args.outdir}")
    print()

    for ptype in types_to_make:
        spec = PLATE_SPECS[ptype]
        outpath = os.path.join(args.outdir, spec["filename"])

        try:
            img = make_sticker(
                logo_path   = args.logo,
                spec        = spec,
                dpi         = args.dpi,
                padding_pct = args.padding,
            )
            cv2.imwrite(outpath, img)
            h, w = img.shape[:2]
            print(f"  ✅  {spec['label']}")
            print(f"      → {spec['width_mm']}×{spec['height_mm']} mm  "
                  f"= {w}×{h} px @ {args.dpi} dpi")
            print(f"      → {outpath}")
        except FileNotFoundError as e:
            print(f"  ❌  {e}", file=sys.stderr)
        except Exception as e:
            import traceback
            print(f"  ❌  {spec['label']}: {e}", file=sys.stderr)
            traceback.print_exc()
        print()

    print("  Done! Stickers are ready to print / sticky-apply.\n")


if __name__ == "__main__":
    main()
