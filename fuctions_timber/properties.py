# ec5/sections/properties.py
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Tuple, List, Literal, Optional

ShapeName = Literal[
    "rectangle", "circle", "ellipse",
    "rhs", "chs",
    "i_section", "t_section",
    "trapezoid", "triangle"
]

# --- NEW: optional sectionproperties dependency -----------------------------
try:
    from sectionproperties.pre.library import primitive_sections, steel_sections
    from sectionproperties.analysis.section import Section as SPSection
    _HAS_SECPROPS = True
except ImportError:   # library not installed → fine, we fall back
    _HAS_SECPROPS = False

@dataclass
class SectionProps:
    # All about centroidal axes x (horizontal) & y (vertical), origin at centroid.
    A: float
    cx: float
    cy: float
    Ixx: float
    Iyy: float
    Ixy: float
    # Extreme fiber distances (from centroid) for section moduli
    y_top: float
    y_bot: float
    x_right: float
    x_left: float
    # Elastic section moduli (about centroidal axes)
    Sx_top: float
    Sx_bot: float
    Sy_right: float
    Sy_left: float
    # Radii of gyration
    rx: float
    ry: float
    # Polar quantities
    Jp: float          # Ixx + Iyy (polar second moment at centroid)
    Jt: Optional[float] = None  # Saint-Venant torsion constant (approx/exact where available)
    # Plastic section moduli (when simple/known; else None)
    Zx: Optional[float] = None
    Zy: Optional[float] = None
        # --- NEW: shear areas for Vx, Vy (used in τ = V / Av) -------------------
    Av_x: Optional[float] = None
    Av_y: Optional[float] = None

# -------------------------------
# Utility: polygon property engine
# -------------------------------
def _poly_props_xy(points: List[Tuple[float, float]]) -> Tuple[float, float, float, float, float, float]:
    """
    Shoelace formulas for a simple polygon (non-self-intersecting).
    Returns: (A, cx, cy, Ixx_c, Iyy_c, Ixy_c) about centroidal axes.
    Points must be ordered either CW or CCW (closed or open).
    """
    if points[0] != points[-1]:
        pts = points + [points[0]]
    else:
        pts = points
    # Area & centroid (global)
    A2 = 0.0
    Cx = 0.0
    Cy = 0.0
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        c = x0 * y1 - x1 * y0
        A2 += c
        Cx += (x0 + x1) * c
        Cy += (y0 + y1) * c
    A = 0.5 * A2
    if abs(A) < 1e-16:
        raise ValueError("Polygon area is zero or degenerate.")
    cx = Cx / (3.0 * A2)
    cy = Cy / (3.0 * A2)

    # Second moments about origin (0,0)
    Ixx_o = 0.0
    Iyy_o = 0.0
    Ixy_o = 0.0
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        c = x0 * y1 - x1 * y0
        Ixx_o += (y0**2 + y0*y1 + y1**2) * c
        Iyy_o += (x0**2 + x0*x1 + x1**2) * c
        Ixy_o += (x0*y1 + 2*x0*y0 + 2*x1*y1 + x1*y0) * c
    Ixx_o *= (1.0/12.0)
    Iyy_o *= (1.0/12.0)
    Ixy_o *= (1.0/24.0)

    # Shift to centroid
    Ixx_c = Ixx_o - A * cy**2
    Iyy_c = Iyy_o - A * cx**2
    Ixy_c = Ixy_o - A * cx * cy
    # Make area positive & keep sign consistency
    A = abs(A)
    return A, cx, cy, Ixx_c, Iyy_c, Ixy_c

def _extents(points: List[Tuple[float, float]], cx: float, cy: float) -> Tuple[float, float, float, float]:
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_left = cx - x_min
    x_right = x_max - cx
    y_bot = cy - y_min
    y_top = y_max - cy
    return x_right, x_left, y_top, y_bot

# -------------------------------
# Closed-form helper formulas
# -------------------------------
def _rect_props(b: float, h: float) -> SectionProps:
    A = b * h
    cx = 0.0; cy = 0.0
    Ixx = b * h**3 / 12.0
    Iyy = h * b**3 / 12.0
    Ixy = 0.0
    y_top = h/2; y_bot = h/2; x_right = b/2; x_left = b/2
    Sx = Ixx / y_top
    Sy = Iyy / x_right
    rx = math.sqrt(Ixx / A); ry = math.sqrt(Iyy / A)
    Jp = Ixx + Iyy
    # Roark approx torsion constant for solid rectangle (a >= b)
    a, t = (b, h) if b >= h else (h, b)
    Jt = a * t**3 * (1/3 - 0.21*(t/a)*(1 - (t**4)/(12*a**4)))
    # Plastic moduli
    Zx = b * h**2 / 4.0
    Zy = h * b**2 / 4.0
    return SectionProps(A, cx, cy, Ixx, Iyy, Ixy,
                        y_top, y_bot, x_right, x_left,
                        Sx, Sx, Sy, Sy, rx, ry, Jp, Jt, Zx, Zy)

def _circle_props(d: float) -> SectionProps:
    r = d/2
    A = math.pi * r**2
    cx = 0.0; cy = 0.0
    I = math.pi * d**4 / 64.0
    Ixx = Iyy = I
    Ixy = 0.0
    y_top = r; y_bot = r; x_right = r; x_left = r
    S = I / r
    rx = ry = math.sqrt(I/A)
    Jp = Ixx + Iyy
    Jt = math.pi * d**4 / 32.0  # exact for solid circle
    # Plastic modulus (solid circle)
    Z = 4.0 * r**3 / 3.0
    return SectionProps(A, cx, cy, Ixx, Iyy, Ixy,
                        y_top, y_bot, x_right, x_left,
                        S, S, S, S, rx, ry, Jp, Jt, Z, Z)

def _ellipse_props(a_diam: float, b_diam: float) -> SectionProps:
    # a_diam = major diameter (x), b_diam = minor diameter (y)
    a = a_diam / 2.0
    b = b_diam / 2.0
    A = math.pi * a * b
    cx = 0.0; cy = 0.0
    Ixx = math.pi * a * b**3 / 4.0
    Iyy = math.pi * a**3 * b / 4.0
    Ixy = 0.0
    y_top = b; y_bot = b; x_right = a; x_left = a
    Sx = Ixx / y_top
    Sy = Iyy / x_right
    rx = math.sqrt(Ixx/A); ry = math.sqrt(Iyy/A)
    Jp = Ixx + Iyy
    # Exact Saint-Venant torsion constant for ellipse
    Jt = math.pi * a * b * (a**2 + b**2) / 4.0
    return SectionProps(A, cx, cy, Ixx, Iyy, Ixy,
                        y_top, y_bot, x_right, x_left,
                        Sx, Sx, Sy, Sy, rx, ry, Jp, Jt, None, None)

def _rhs_props(b: float, h: float, t: float) -> SectionProps:
    # Uniform wall rectangular tube (no corner radii)
    b_i = b - 2*t
    h_i = h - 2*t
    if b_i <= 0 or h_i <= 0:
        raise ValueError("Wall thickness too large for RHS.")
    A = b*h - b_i*h_i
    cx = 0.0; cy = 0.0
    Ixx = (b*h**3 - b_i*h_i**3) / 12.0
    Iyy = (h*b**3 - h_i*b_i**3) / 12.0
    Ixy = 0.0
    y_top = h/2; y_bot = h/2; x_right = b/2; x_left = b/2
    Sx = Ixx / y_top
    Sy = Iyy / x_right
    rx = math.sqrt(Ixx / A); ry = math.sqrt(Iyy / A)
    Jp = Ixx + Iyy
    # Thin-wall torsion constant
    Jt = 2.0 * t * (b*h) * ( (b*h)/(b + h) ) / ( (b/h) + (h/b) )
    # Simpler engineering approx:
    Jt = 2.0 * t * (b*h) * ( (b*h) / (b + h) ) / 3.0
    return SectionProps(A, cx, cy, Ixx, Iyy, Ixy,
                        y_top, y_bot, x_right, x_left,
                        Sx, Sx, Sy, Sy, rx, ry, Jp, Jt, None, None)

def _chs_props(d: float, t: float) -> SectionProps:
    do = d
    di = d - 2*t
    if di <= 0:
        raise ValueError("Wall thickness too large for CHS.")
    A = math.pi/4 * (do**2 - di**2)
    cx = 0.0; cy = 0.0
    I = (math.pi/64) * (do**4 - di**4)
    Ixx = Iyy = I
    Ixy = 0.0
    r_o = do/2
    y_top = r_o; y_bot = r_o; x_right = r_o; x_left = r_o
    S = I / r_o
    rx = ry = math.sqrt(I / A)
    Jp = Ixx + Iyy
    # Thin-wall torsion constant for circular tube
    Jt = math.pi/32 * (do**4 - di**4)
    return SectionProps(A, cx, cy, Ixx, Iyy, Ixy,
                        y_top, y_bot, x_right, x_left,
                        S, S, S, S, rx, ry, Jp, Jt, None, None)

def _i_section_props(h: float, b: float, tw: float, tf: float) -> SectionProps:
    # Simple I/H without fillets; centroid at geometric center
    if 2*tf >= h or tw >= b:
        raise ValueError("Unrealistic I-section proportions.")
    A = 2*b*tf + (h - 2*tf)*tw
    cx = 0.0; cy = 0.0
    Ixx = (b*h**3)/12.0 - ((b - tw)*(h - 2*tf)**3)/12.0
    Iyy = 2*(tf*b**3)/12.0 + ((h - 2*tf)*tw**3)/12.0
    Ixy = 0.0
    y_top = h/2; y_bot = h/2; x_right = b/2; x_left = b/2
    Sx = Ixx / y_top
    Sy = Iyy / x_right
    rx = math.sqrt(Ixx/A); ry = math.sqrt(Iyy/A)
    Jp = Ixx + Iyy
    # Thin-wall torsion constant
    Jt = 2.0*b*tf**3/3.0 + (h - 2*tf)*tw**3/3.0
    # Plastic section modulus (approx, symmetric I)
    Zx = 2.0 * ( b*tf*(h/2 - tf/2) + (tw*(h/2 - tf))*( (h/4 - tf/2) ) )
    Zy = (tf * b**2 / 2.0) + (h - 2*tf) * (tw**2 / 4.0)
    return SectionProps(A, cx, cy, Ixx, Iyy, Ixy,
                        y_top, y_bot, x_right, x_left,
                        Sx, Sx, Sy, Sy, rx, ry, Jp, Jt, Zx, Zy)

def _t_section_props(h: float, b: float, tw: float, tf: float) -> SectionProps:
    # T with flange on top, web below; centroid NOT at mid-depth in general.
    if tf >= h or tw > b:
        raise ValueError("Unrealistic T-section proportions.")
    A_f = b*tf
    A_w = tw*(h - tf)
    A = A_f + A_w
    # Centroid from mid-depth? Work from bottom for clarity:
    # y=0 at bottom, y up. Place flange at top: bottom web height = h - tf
    y_f_c = h - tf/2
    y_w_c = (h - tf)/2
    cy = (A_f*y_f_c + A_w*y_w_c)/A
    cx = 0.0  # symmetric about y-axis
    # Ixx about centroid via parallel axes
    Ixx_f_c = (b*tf**3)/12.0 + A_f*(y_f_c - cy)**2
    Ixx_w_c = (tw*(h - tf)**3)/12.0 + A_w*(y_w_c - cy)**2
    Ixx = Ixx_f_c + Ixx_w_c
    Iyy = (b**3*tf)/12.0 + (tw**3*(h - tf))/12.0
    Ixy = 0.0
    # Extents:
    y_top = h - cy
    y_bot = cy
    x_right = b/2
    x_left = b/2
    Sx_top = Ixx / y_top
    Sx_bot = Ixx / y_bot if y_bot > 0 else float("inf")
    Sy = Iyy / x_right
    rx = math.sqrt(Ixx/A); ry = math.sqrt(Iyy/A)
    Jp = Ixx + Iyy
    # Very rough thin-wall torsion constant:
    Jt = b*tf**3/3.0 + (h - tf)*tw**3/3.0
    return SectionProps(A, cx, cy, Ixx, Iyy, Ixy,
                        y_top, y_bot, x_right, x_left,
                        Sx_top, Sx_bot, Sy, Sy, rx, ry, Jp, Jt, None, None)

def _trapezoid_props(b1: float, b2: float, h: float) -> SectionProps:
    """
    Isosceles trapezoid with bases b1 (bottom) and b2 (top), height h.
    Centered on y-axis (x=0 at centerline), bottom at y=0.
    """
    x0 = -b1/2; x1 = b1/2
    x2 = b2/2;  x3 = -b2/2
    pts = [(x0, 0.0), (x1, 0.0), (x2, h), (x3, h)]  # CCW
    A, cxg, cyg, Ixx, Iyy, Ixy = _poly_props_xy(pts)
    # Shift centroid to origin:
    cx = cxg - cxg  # 0
    cy = 0.0        # we’ll re-center by translating points
    # Translate points to centroid to get extents
    pts_c = [(x - cxg, y - cyg) for (x, y) in pts]
    xr, xl, yt, yb = _extents(pts_c, 0.0, 0.0)
    Sx_top = Ixx / yt
    Sx_bot = Ixx / yb
    Sy_right = Iyy / xr
    Sy_left = Iyy / xl
    rx = math.sqrt(Ixx/A); ry = math.sqrt(Iyy/A)
    Jp = Ixx + Iyy
    return SectionProps(A, 0.0, 0.0, Ixx, Iyy, Ixy, yt, yb, xr, xl,
                        Sx_top, Sx_bot, Sy_right, Sy_left, rx, ry, Jp, None, None, None)

def _triangle_props(b: float, h: float) -> SectionProps:
    """
    Isosceles triangle with base b on y=0 and apex at (0, h).
    """
    pts = [(-b/2, 0.0), (b/2, 0.0), (0.0, h)]  # CCW
    A, cxg, cyg, Ixx, Iyy, Ixy = _poly_props_xy(pts)
    # Translate to centroid
    pts_c = [(x - cxg, y - cyg) for (x, y) in pts]
    xr, xl, yt, yb = _extents(pts_c, 0.0, 0.0)
    Sx_top = Ixx / yt
    Sx_bot = Ixx / yb
    Sy_right = Iyy / xr
    Sy_left = Iyy / xl
    rx = math.sqrt(Ixx/A); ry = math.sqrt(Iyy/A)
    Jp = Ixx + Iyy
    return SectionProps(A, 0.0, 0.0, Ixx, Iyy, Ixy, yt, yb, xr, xl,
                        Sx_top, Sx_bot, Sy_right, Sy_left, rx, ry, Jp, None, None, None)
# -------------------------------
# Shear area helpers (Option A)
# -------------------------------

def _compute_shear_areas_sp(shape: str, dims: Dict[str, float], A: float) -> Tuple[Optional[float], Optional[float]]:
    """
    Try to compute shear areas (Av_x, Av_y) using the sectionproperties library.
    If not available or shape not mapped, returns (None, None).
    """
    if not _HAS_SECPROPS:
        return None, None

    s = shape.lower()

    # Map our shapes → sectionproperties geometries
    try:
        geom = None

        if s == "rectangle":
            b = dims["b"]; h = dims["h"]
            geom = primitive_sections.rectangular_section(b=b, d=h)

        elif s == "circle":
            d = dims["d"]
            geom = primitive_sections.circular_section(d=d)

        elif s == "ellipse":
            a = dims["a"]; b = dims["b"]
            geom = primitive_sections.elliptical_section(b=a, d=b)

        elif s == "rhs":
            b = dims["b"]; h = dims["h"]; t = dims["t"]
            geom = steel_sections.rectangular_hollow_section(
                d=h, b=b, t_f=t, t_w=t, r_out=0.0, n_r=8
            )

        elif s == "chs":
            d = dims["d"]; t = dims["t"]
            geom = steel_sections.circular_hollow_section(d=d, t=t)

        elif s == "i_section":
            h = dims["h"]; b = dims["b"]; tw = dims["tw"]; tf = dims["tf"]
            geom = steel_sections.i_section(
                d=h, b=b, t_f=tf, t_w=tw, r=0.0, n_r=8
            )

        elif s == "t_section":
            h = dims["h"]; b = dims["b"]; tw = dims["tw"]; tf = dims["tf"]
            geom = steel_sections.tee_section(
                d=h, b=b, t_f=tf, t_w=tw, r=0.0, n_r=8
            )

        # More shapes can be mapped later...

        if geom is None:
            return None, None

        # crude mesh size ~ 1/10 of max dimension
        max_dim = max(dims.values()) if dims else 1.0
        geom.create_mesh(mesh_sizes=[max_dim / 10.0])

        sec = SPSection(geometry=geom)
        sec.calculate_geometric_properties()
        sec.calculate_warping_properties()

        # sectionproperties uses (Asx, Asy) in the section x-y system
        Asx, Asy = sec.get_as()
        return float(abs(Asx)), float(abs(Asy))

    except Exception:
        # any failure → just skip and fall back to simple k·A
        return None, None


def _approx_shear_areas(shape: str, A: float) -> Tuple[float, float]:
    """
    Simple engineering approximations for shear area.
    Used when sectionproperties is not available or fails.
    """
    s = shape.lower()

    # classic values (approximate, EC5-compatible level)
    if s in ("rectangle", "trapezoid", "triangle"):
        k = 5.0 / 6.0   # solid rectangular-ish
        return k * A, k * A

    if s in ("circle", "chs"):
        k = 0.9         # solid/hollow circular, rough
        return k * A, k * A

    if s in ("i_section", "t_section", "rhs"):
        # very rough: most shear in web → ~0.8A
        k = 0.8
        return k * A, k * A

    # default conservative fallback
    return A, A


def _attach_shear_areas(shape: str, dims: Dict[str, float], P: SectionProps) -> SectionProps:
    """
    Fill P.Av_x and P.Av_y using:
        1) sectionproperties (if available, mapped)
        2) otherwise approximate k·A
    """
    Avx_sp, Avy_sp = _compute_shear_areas_sp(shape, dims, P.A)

    if Avx_sp is not None and Avy_sp is not None:
        P.Av_x = Avx_sp
        P.Av_y = Avy_sp
        return P

    # fallback
    Avx, Avy = _approx_shear_areas(shape, P.A)
    P.Av_x = Avx
    P.Av_y = Avy
    return P

# -------------------------------
# Public API
# -------------------------------
# -------------------------------
# Public API
# -------------------------------
def get_section_properties(shape: ShapeName, **dims) -> Dict[str, float]:
    """
    Returns a dict with standard centroidal properties:
    A, cx, cy, Ixx, Iyy, Ixy, y_top, y_bot, x_right, x_left,
    Sx_top, Sx_bot, Sy_right, Sy_left, rx, ry, Jp,
    optional: Jt, Zx, Zy, Av_x, Av_y

    Dimensions are in consistent units (e.g. mm). Outputs are in those units:
    - Area -> units^2
    - Inertias -> units^4
    - Section moduli -> units^3
    - Radii of gyration -> units
    """
    s = shape.lower()
    # cast dims to float once
    dims_f = {k: float(v) for k, v in dims.items()}

    if s == "rectangle":
        b = dims_f["b"]; h = dims_f["h"]
        P = _rect_props(b, h)
    elif s == "circle":
        d = dims_f["d"]
        P = _circle_props(d)
    elif s == "ellipse":
        a = dims_f["a"]; b = dims_f["b"]
        P = _ellipse_props(a, b)
    elif s == "rhs":
        b = dims_f["b"]; h = dims_f["h"]; t = dims_f["t"]
        P = _rhs_props(b, h, t)
    elif s == "chs":
        d = dims_f["d"]; t = dims_f["t"]
        P = _chs_props(d, t)
    elif s == "i_section":
        h = dims_f["h"]; b = dims_f["b"]; tw = dims_f["tw"]; tf = dims_f["tf"]
        P = _i_section_props(h, b, tw, tf)
    elif s == "t_section":
        h = dims_f["h"]; b = dims_f["b"]; tw = dims_f["tw"]; tf = dims_f["tf"]
        P = _t_section_props(h, b, tw, tf)
    elif s == "trapezoid":
        b1 = dims_f["b1"]; b2 = dims_f["b2"]; h = dims_f["h"]
        P = _trapezoid_props(b1, b2, h)
    elif s == "triangle":
        b = dims_f["b"]; h = dims_f["h"]
        P = _triangle_props(b, h)
    else:
        raise ValueError(f"Unsupported shape: {shape}")

    # --- NEW: attach shear areas (Av_x, Av_y) -------------------------------
    P = _attach_shear_areas(s, dims_f, P)

    # Pack to dict
    out = {
        "A": P.A, "cx": P.cx, "cy": P.cy,
        "Ixx": P.Ixx, "Iyy": P.Iyy, "Ixy": P.Ixy,
        "y_top": P.y_top, "y_bot": P.y_bot, "x_right": P.x_right, "x_left": P.x_left,
        "Sx_top": P.Sx_top, "Sx_bot": P.Sx_bot, "Sy_right": P.Sy_right, "Sy_left": P.Sy_left,
        "rx": P.rx, "ry": P.ry, "Jp": P.Jp,
    }
    if P.Jt is not None:
        out["Jt"] = P.Jt
    if P.Zx is not None:
        out["Zx"] = P.Zx
    if P.Zy is not None:
        out["Zy"] = P.Zy
    if P.Av_x is not None:
        out["Av_x"] = P.Av_x
    if P.Av_y is not None:
        out["Av_y"] = P.Av_y

    return out


