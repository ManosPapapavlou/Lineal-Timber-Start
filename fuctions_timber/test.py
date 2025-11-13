"""
Quick test script for:
- properties.get_section_properties (including Av_x, Av_y)
- checks.run_ec5_basic_checks (mother of checks)

Run with:
    python test_sections.py
"""

from pprint import pprint

from properties import get_section_properties
from checks import (
    InternalForces,
    DesignStrengths,
    run_ec5_basic_checks,
)


def test_section_geometry():
    print("=" * 80)
    print("GEOMETRY TESTS")
    print("=" * 80)

    # A few sample shapes (dimensions in mm)
    test_cases = [
        ("rectangle", {"b": 100.0, "h": 200.0}),
        ("circle",    {"d": 200.0}),
        ("i_section", {"h": 300.0, "b": 150.0, "tw": 8.0, "tf": 12.0}),
        ("rhs",       {"b": 200.0, "h": 100.0, "t": 8.0}),
    ]

    for shape, dims in test_cases:
        print(f"\n--- Shape: {shape}  dims={dims} ---")
        sec = get_section_properties(shape, **dims)

        # Basic key properties
        print(f"A        = {sec['A']:.3f}")
        print(f"Ixx      = {sec['Ixx']:.3e}")
        print(f"Iyy      = {sec['Iyy']:.3e}")
        print(f"Sx_top   = {sec['Sx_top']:.3e}, Sx_bot   = {sec['Sx_bot']:.3e}")
        print(f"Sy_right = {sec['Sy_right']:.3e}, Sy_left = {sec['Sy_left']:.3e}")

        # Shear areas if present
        Av_x = sec.get("Av_x", None)
        Av_y = sec.get("Av_y", None)
        if Av_x is not None and Av_y is not None:
            print(f"Av_x     = {Av_x:.3f}, Av_y = {Av_y:.3f}")
            print(f"Av_y / A = {Av_y / sec['A']:.3f}")
        else:
            print("Av_x / Av_y not present in section dict (fallback to A in checks).")


def test_ec5_checks_rectangular():
    print("\n" + "=" * 80)
    print("EC5 CHECKS TEST – RECTANGULAR BEAM (biaxial + shear + torsion)")
    print("=" * 80)

    # Example: rectangular C24 beam 100x200 mm
    shape = "rectangle"
    dims = {"b": 100.0, "h": 200.0}  # mm

    # Internal forces (example design values, in N, N·mm)
    # Think: 20 kN·m bending about x, 10 kN·m about y, 60 kN shear, 5 kN·m torsion
    forces = InternalForces(
        N=0.0,           # pure bending + shear + torsion
        Mx=20e6,         # 20 kN·m → 20e6 N·mm (bending about x)
        My=10e6,         # 10 kN·m → 10e6 N·mm (bending about y)
        Vx=0.0,
        Vy=60e3,         # 60 kN shear → 60e3 N
        T=5e6,           # 5 kN·m torsion → 5e6 N·mm
    )

    # Design strengths (example for C24, already kmod/gammaM reduced)
    # Replace with your real ft0,d, fc0,d, fm,d, fv,d when you link to your EC5 tables.
    strengths = DesignStrengths(
        ft0_d=14.0,   # MPa
        fc0_d=21.0,   # MPa
        fm_x_d=24.0,  # MPa (bending about y-axis → stress along x)
        fm_y_d=24.0,  # MPa (bending about x-axis → stress along y)
        fv_x_d=3.5,   # MPa
        fv_y_d=3.5,   # MPa
        fv_d=3.5,     # generic shear for torsion
        km=0.7,       # EC5 k_m for rectangular solid timber in bending
    )

    summary = run_ec5_basic_checks(
        shape=shape,
        dims=dims,
        forces=forces,
        strengths=strengths,
    )

    print(f"\nShape: {summary.shape}, dims={summary.dims}")
    print("\nSection properties (subset):")
    for k in ["A", "Ixx", "Iyy", "Sx_top", "Sy_right", "Av_x", "Av_y", "Jt"]:
        v = summary.section.get(k, None)
        if v is not None:
            print(f"  {k:8s} = {v}")

    print("\nInternal forces:")
    print(summary.forces)

    print("\nChecks:")
    for chk in summary.checks:
        status = "OK" if chk.ok else "NOT OK"
        print(
            f"  - {chk.name:30s} "
            f"η = {chk.utilization:.3f} (limit {chk.limit:.3f}) → {status}"
        )
        if chk.details:
            print(f"      {chk.details}")

    print("\nGoverning check:")
    g = summary.governing_check
    print(
        f"  {g.name:30s} η = {g.utilization:.3f} "
        f"(limit {g.limit:.3f}) → {'OK' if g.ok else 'NOT OK'}"
    )
    print("  details:", g.details)


def test_ec5_checks_i_section():
    print("\n" + "=" * 80)
    print("EC5 CHECKS TEST – I-SECTION (bending + axial + shear)")
    print("=" * 80)

    shape = "i_section"
    dims = {"h": 300.0, "b": 150.0, "tw": 8.0, "tf": 12.0}  # mm

    # Internal forces: compression + major-axis bending + shear
    forces = InternalForces(
        N=-150e3,       # 150 kN compression
        Mx=40e6,        # 40 kN·m about x
        My=0.0,
        Vx=0.0,
        Vy=80e3,        # 80 kN shear
        T=5.0,
    )

    strengths = DesignStrengths(
        ft0_d=14.0,
        fc0_d=21.0,
        fm_x_d=24.0,
        fm_y_d=24.0,
        fv_x_d=3.5,
        fv_y_d=3.5,
        fv_d=3.5,
        km=0.7,
    )

    summary = run_ec5_basic_checks(
        shape=shape,
        dims=dims,
        forces=forces,
        strengths=strengths,
    )

    print(f"\nShape: {summary.shape}, dims={summary.dims}")
    print("\nSection properties (subset):")
    for k in ["A", "Ixx", "Iyy", "Sx_top", "Sy_right", "Av_x", "Av_y", "Jt"]:
        v = summary.section.get(k, None)
        if v is not None:
            print(f"  {k:8s} = {v}")

    print("\nInternal forces:")
    print(summary.forces)

    print("\nChecks:")
    for chk in summary.checks:
        status = "OK" if chk.ok else "NOT OK"
        print(
            f"  - {chk.name:30s} "
            f"η = {chk.utilization:.3f} (limit {chk.limit:.3f}) → {status}"
        )
        if chk.details:
            print(f"      {chk.details}")

    print("\nGoverning check:")
    g = summary.governing_check
    print(
        f"  {g.name:30s} η = {g.utilization:.3f} "
        f"(limit {g.limit:.3f}) → {'OK' if g.ok else 'NOT OK'}"
    )
    print("  details:", g.details)


if __name__ == "__main__":
    test_section_geometry()
    test_ec5_checks_rectangular()
    test_ec5_checks_i_section()
