# fuctions_timber/checks.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from .properties import get_section_properties  # <-- CHANGE HERE



# ---------------------------------
# Data containers
# ---------------------------------

@dataclass
class InternalForces:
    """
    Design internal forces for one member & one combination.

    Sign convention:
        N  > 0 : tension,  N < 0 : compression
        Mx : bending about x-axis → tension/compression at top/bottom (±y)
        My : bending about y-axis → tension/compression at left/right (±x)
        Vx : shear force in x-direction
        Vy : shear force in y-direction
        T  : torsion
    Units must be consistent with geometry & strengths (e.g. N, N·mm).
    """
    N: float = 0.0
    Mx: float = 0.0
    My: float = 0.0
    Vx: float = 0.0
    Vy: float = 0.0
    T: float = 0.0


@dataclass
class DesignStrengths:
    """
    EC5 DESIGN strengths already including kmod / gamma_M.
    All in consistent stress units (e.g. MPa).

    fm_x_d, fm_y_d:
        bending strength about y- and x-axis respectively
        (Mx → σ varying along y → compare with fm_y_d, etc.)

    fv_x_d, fv_y_d:
        shear strengths in x- and y-directions.

    fv_d:
        generic shear strength for torsion check (if not None).
    """
    ft0_d: float           # tension parallel to grain
    fc0_d: float           # compression parallel to grain
    fm_x_d: float          # bending strength for My (stress along x)
    fm_y_d: float          # bending strength for Mx (stress along y)
    fv_x_d: float          # shear strength in x-direction
    fv_y_d: float          # shear strength in y-direction

    # Optional extras (for future use / torsion)
    ft90_d: Optional[float] = None
    fc90_d: Optional[float] = None
    fv_d: Optional[float] = None

    # EC5 bending interaction factor k_m (6.1.6, 6.2.3, 6.2.4)
    km: float = 1.0


@dataclass
class CheckResult:
    name: str
    utilization: float
    limit: float = 1.0
    ok: bool = True
    details: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemberCheckSummary:
    """
    Summary for one member & one combination (or envelope later).
    """
    shape: str
    dims: Dict[str, float]
    section: Dict[str, float]
    forces: InternalForces
    strengths: DesignStrengths
    checks: List[CheckResult]
    governing_check: CheckResult


# ---------------------------------
# Helper: torsion shape factor (EC5-ish)
# ---------------------------------

def _torsion_shape_factor(shape: str, section: Dict[str, float]) -> float:
    """
    EC5-ish k_shape for torsion:
      - circle/CHS: k_shape ≈ 1.2
      - rectangles/RHS/trapezoid/triangle: 1 + 0.15 * h/b  ≤ 2.0
      - I/T sections & others: ≈ 1.0 (neutral)

    Used in torsion check:
        τ_tor,d ≤ k_shape * f_v,d
    """
    s = shape.lower()
    b = 2.0 * section.get("x_right", 0.0)
    h = 2.0 * section.get("y_top", 0.0)

    if s in ("circle", "chs"):
        return 1.2

    if s in ("rectangle", "rhs", "trapezoid", "triangle"):
        if b > 0.0 and h > 0.0:
            k = 1.0 + 0.15 * (h / b)
            return min(k, 2.0)
        return 1.0

    # I / T sections & default
    return 1.0


# ---------------------------------
# Helper: compute basic stresses
# ---------------------------------

def _compute_basic_stresses(section: Dict[str, float],
                            forces: InternalForces) -> Dict[str, float]:
    """
    Compute simple stress measures from geometry + forces.
    Uses centroidal section moduli, area and shear areas.

    NOTE: Units must be consistent so that:
        - N / A → same stress units as strengths (e.g. MPa if N, mm²)
        - M / S → same stress units as strengths (e.g. MPa if N·mm, mm³)
    """
    A = section["A"]

    # Axial stress (positive = tension)
    sigma_N = forces.N / A if A != 0.0 else 0.0

    # Bending about x-axis (Mx → tension at top/bottom = ±y)
    Sx_top = section["Sx_top"]
    Sx_bot = section["Sx_bot"]
    sigma_mx_top = forces.Mx / Sx_top if Sx_top != 0.0 else 0.0
    sigma_mx_bot = -forces.Mx / Sx_bot if Sx_bot != 0.0 else 0.0  # opposite sign fibre

    # Bending about y-axis (My → tension at left/right = ±x)
    Sy_right = section["Sy_right"]
    Sy_left = section["Sy_left"]
    sigma_my_right = forces.My / Sy_right if Sy_right != 0.0 else 0.0
    sigma_my_left = -forces.My / Sy_left if Sy_left != 0.0 else 0.0

    # Shear stresses using shear areas Av_x, Av_y if present
    Av_x = section.get("Av_x", A)  # fallback: A
    Av_y = section.get("Av_y", A)
    tau_x = forces.Vx / Av_x if Av_x != 0.0 else 0.0
    tau_y = forces.Vy / Av_y if Av_y != 0.0 else 0.0

    # Extreme fibre total stresses (top/bottom)
    sigma_top = sigma_N + sigma_mx_top
    sigma_bot = sigma_N + sigma_mx_bot
    # left/right including axial:
    sigma_left = sigma_N + sigma_my_left
    sigma_right = sigma_N + sigma_my_right

    return {
        "sigma_N": sigma_N,
        "sigma_mx_top": sigma_mx_top,
        "sigma_mx_bot": sigma_mx_bot,
        "sigma_my_right": sigma_my_right,
        "sigma_my_left": sigma_my_left,
        "sigma_top": sigma_top,
        "sigma_bot": sigma_bot,
        "sigma_left": sigma_left,
        "sigma_right": sigma_right,
        "tau_x": tau_x,
        "tau_y": tau_y,
    }


# ---------------------------------
# Mother of checks
# ---------------------------------

def run_ec5_basic_checks(
    shape: str,
    dims: Dict[str, float],
    forces: InternalForces,
    strengths: DesignStrengths,
) -> MemberCheckSummary:
    """
    Central 'mother of checks' for timber members according to EC5 (simplified).

    - Accepts ANY section shape supported by get_section_properties().
    - Computes basic stresses.
    - Performs:
        * axial tension & compression (no buckling)
        * bending about x and y
        * biaxial bending interaction (6.1.6) even if N = 0
        * shear in x and y
        * torsion (6.1.8 style)
        * combined bending + axial tension (6.2.3)
        * combined bending + axial compression (6.2.4, simplified)
    - Returns all CheckResult plus the governing one.
    """

    # 1) Geometry
    section = get_section_properties(shape, **dims)  # dict with A, Sx_top, etc.

    # 2) Stresses
    s = _compute_basic_stresses(section, forces)

    checks: List[CheckResult] = []

    # ==============
    # Axial checks
    # ==============

    # Axial tension parallel to grain (6.1.2)
    if forces.N > 0.0 and strengths.ft0_d > 0.0:
        sigma_t0 = s["sigma_N"]
        eta_t0 = sigma_t0 / strengths.ft0_d
        checks.append(CheckResult(
            name="axial_tension_parallel",
            utilization=eta_t0,
            ok=(eta_t0 <= 1.0),
            details=f"σ_t,0,d / f_t,0,d = {eta_t0:.3f}",
            extra={"sigma_t0": sigma_t0}
        ))

    # Axial compression parallel to grain (no buckling) (6.1.4)
    if forces.N < 0.0 and strengths.fc0_d > 0.0:
        sigma_c0 = abs(s["sigma_N"])
        eta_c0 = sigma_c0 / strengths.fc0_d
        checks.append(CheckResult(
            name="axial_compression_parallel",
            utilization=eta_c0,
            ok=(eta_c0 <= 1.0),
            details=f"σ_c,0,d / f_c,0,d = {eta_c0:.3f}",
            extra={"sigma_c0": sigma_c0}
        ))

    # ==============
    # Bending checks (single-axis)
    # ==============

    sigma_mx_max = max(abs(s["sigma_mx_top"]), abs(s["sigma_mx_bot"]))
    sigma_my_max = max(abs(s["sigma_my_right"]), abs(s["sigma_my_left"]))

    # Bending about x-axis (Mx) → compare with fm_y_d
    if strengths.fm_y_d > 0.0:
        eta_mx = sigma_mx_max / strengths.fm_y_d
        checks.append(CheckResult(
            name="bending_x",
            utilization=eta_mx,
            ok=(eta_mx <= strengths.km),
            limit=strengths.km,
            details=f"max|σ_mx,d| / f_m,y,d = {eta_mx:.3f} (limit k_m={strengths.km})",
            extra={"sigma_mx_max": sigma_mx_max}
        ))

    # Bending about y-axis (My) → compare with fm_x_d
    if strengths.fm_x_d > 0.0:
        eta_my = sigma_my_max / strengths.fm_x_d
        checks.append(CheckResult(
            name="bending_y",
            utilization=eta_my,
            ok=(eta_my <= strengths.km),
            limit=strengths.km,
            details=f"max|σ_my,d| / f_m,x,d = {eta_my:.3f} (limit k_m={strengths.km})",
            extra={"sigma_my_max": sigma_my_max}
        ))

    # ==============
    # Biaxial bending interaction (6.1.6)
    #   σ_mx/fm_y + σ_my/fm_x ≤ k_m
    # ==============
    term_mx_bi = sigma_mx_max / strengths.fm_y_d if strengths.fm_y_d > 0.0 else 0.0
    term_my_bi = sigma_my_max / strengths.fm_x_d if strengths.fm_x_d > 0.0 else 0.0
    eta_bi = term_mx_bi + term_my_bi

    if term_mx_bi > 0.0 or term_my_bi > 0.0:
        checks.append(CheckResult(
            name="bending_biaxial",
            utilization=eta_bi,
            ok=(eta_bi <= strengths.km),
            limit=strengths.km,
            details=(
                "EC5 6.1.6: σ_mx/f_m,y + σ_my/f_m,x = "
                f"{eta_bi:.3f} (limit k_m={strengths.km})"
            ),
            extra={
                "term_mx": term_mx_bi,
                "term_my": term_my_bi,
            }
        ))

    # ==============
    # Shear checks (6.1.7)
    # ==============
    if strengths.fv_x_d > 0.0:
        tau_x = abs(s["tau_x"])
        eta_vx = tau_x / strengths.fv_x_d
        checks.append(CheckResult(
            name="shear_x",
            utilization=eta_vx,
            ok=(eta_vx <= 1.0),
            details=f"τ_x,d / f_v,x,d = {eta_vx:.3f}",
            extra={"tau_x": tau_x}
        ))

    if strengths.fv_y_d > 0.0:
        tau_y = abs(s["tau_y"])
        eta_vy = tau_y / strengths.fv_y_d
        checks.append(CheckResult(
            name="shear_y",
            utilization=eta_vy,
            ok=(eta_vy <= 1.0),
            details=f"τ_y,d / f_v,y,d = {eta_vy:.3f}",
            extra={"tau_y": tau_y}
        ))

    # ==============
    # Torsion check (6.1.8 style)
    # τ_tor,d ≤ k_shape * f_v,d
    # τ_tor ≈ T * r_max / Jt
    # ==============
    Jt = section.get("Jt", None)
    if Jt is not None and Jt != 0.0 and abs(forces.T) > 0.0:
        # distance of furthest fibre from centroid
        x_right = section.get("x_right", 0.0)
        y_top = section.get("y_top", 0.0)
        r_max = (x_right**2 + y_top**2) ** 0.5

        tau_tor = abs(forces.T) * r_max / Jt  # consistent units: T[N·len], Jt[len^4]

        # choose a shear strength to compare with:
        fv_design = strengths.fv_d
        if fv_design is None:
            fv_design = max(strengths.fv_x_d, strengths.fv_y_d, 0.0)

        if fv_design > 0.0:
            k_shape = _torsion_shape_factor(shape, section)
            eta_tor = tau_tor / (k_shape * fv_design)
            checks.append(CheckResult(
                name="torsion",
                utilization=eta_tor,
                ok=(eta_tor <= 1.0),
                details=(
                    f"EC5 6.1.8: τ_tor,d / (k_shape f_v,d) = {eta_tor:.3f} "
                    f"(k_shape={k_shape:.2f})"
                ),
                extra={
                    "tau_tor": tau_tor,
                    "fv_design": fv_design,
                    "k_shape": k_shape,
                    "r_max": r_max,
                }
            ))

    # ==============
    # Combined bending + axial tension (6.2.3)
    #
    # σ_mx/fm_y + σ_my/fm_x + σ_t0/ft0 ≤ k_m
    # ==============
    if forces.N > 0.0 and strengths.ft0_d > 0.0:
        sigma_t0 = s["sigma_N"]

        term_mx = sigma_mx_max / strengths.fm_y_d if strengths.fm_y_d > 0.0 else 0.0
        term_my = sigma_my_max / strengths.fm_x_d if strengths.fm_x_d > 0.0 else 0.0
        term_Nt = sigma_t0 / strengths.ft0_d

        eta_comb_t = term_mx + term_my + term_Nt

        checks.append(CheckResult(
            name="combined_bending_tension",
            utilization=eta_comb_t,
            ok=(eta_comb_t <= strengths.km),
            limit=strengths.km,
            details=(
                "EC5 6.2.3: σ_mx/f_m,y + σ_my/f_m,x + σ_t0/f_t0 "
                f"= {eta_comb_t:.3f} (limit k_m={strengths.km})"
            ),
            extra={
                "term_mx": term_mx,
                "term_my": term_my,
                "term_Nt": term_Nt,
            }
        ))

    # ==============
    # Combined bending + axial compression (6.2.4, simplified)
    #
    # sqrt( (σ_mx/fm_y + σ_my/fm_x)^2 + (σ_c0/fc0)^2 ) ≤ k_m
    # ==============
    if forces.N < 0.0 and strengths.fc0_d > 0.0:
        sigma_c0 = abs(s["sigma_N"])

        term_mx = sigma_mx_max / strengths.fm_y_d if strengths.fm_y_d > 0.0 else 0.0
        term_my = sigma_my_max / strengths.fm_x_d if strengths.fm_x_d > 0.0 else 0.0
        term_c = sigma_c0 / strengths.fc0_d

        Rm = term_mx + term_my
        Rc = term_c
        eta_comb_c = (Rm**2 + Rc**2) ** 0.5

        checks.append(CheckResult(
            name="combined_bending_compression",
            utilization=eta_comb_c,
            ok=(eta_comb_c <= strengths.km),
            limit=strengths.km,
            details=(
                "EC5 6.2.4 (simplified): sqrt( (σ_mx/f_m,y + σ_my/f_m,x)^2 "
                f"+ (σ_c0/f_c0)^2 ) = {eta_comb_c:.3f} (limit k_m={strengths.km})"
            ),
            extra={
                "Rm": Rm,
                "Rc": Rc,
                "term_mx": term_mx,
                "term_my": term_my,
                "term_c": term_c,
            }
        ))

    # ==============
    # Governing check
    # ==============
    if checks:
        governing = max(checks, key=lambda c: c.utilization)
    else:
        governing = CheckResult(
            name="no_checks",
            utilization=0.0,
            ok=True,
            details="No checks performed."
        )

    return MemberCheckSummary(
        shape=shape,
        dims=dims,
        section=section,
        forces=forces,
        strengths=strengths,
        checks=checks,
        governing_check=governing,
    )
