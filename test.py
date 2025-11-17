# test.py
"""
Quick test for EC5 timber cross-section checks (including torsion).

Example:
    Rectangular section b = 60 mm, h = 140 mm
    Timber class: C24
    kmod = 0.60 (service class 1, permanent)
    gamma_M = 1.30
    fvk = 4.0 MPa

    ⇒ f_v,d = kmod * fvk / gamma_M ≈ 1.846 MPa

    We apply only torsion:
        M_tor,d = 1.0 kNm = 1.0e6 Nmm

Run:
    python test.py
"""

from fuctions_timber.checksv2 import (
    run_ec5_basic_checks,
    InternalForces,
    DesignStrengths,
)

def main():
    # ----------------------------------------------------------
    # 1) Geometry: 60 x 140 mm rectangular section
    # ----------------------------------------------------------
    shape = "rectangle"
    dims = {
        "b": 60.0,   # width in mm
        "h": 140.0,  # depth in mm
    }

    # ----------------------------------------------------------
    # 2) Material strengths for C24 (DESIGN values, already kmod/gammaM)
    #    Using your JSON:
    #      fmk  = 24.0 MPa
    #      ft0k = 14.5
    #      fc0k = 21.0
    #      fvk  = 4.0
    #
    #    kmod  = 0.60
    #    gammaM = 1.30
    #
    #    ft0d = kmod*ft0k/gammaM
    #    fc0d = kmod*fc0k/gammaM
    #    fmd  = kmod*fmk /gammaM
    #    fvd  = kmod*fvk /gammaM
    # ----------------------------------------------------------
    kmod = 0.60
    gamma_M = 1.30

    ft0k = 14.5
    fc0k = 21.0
    fmk  = 24.0
    fvk  = 4.0

    ft0_d = kmod * ft0k / gamma_M
    fc0_d = kmod * fc0k / gamma_M
    fm_d  = kmod * fmk  / gamma_M
    fv_d  = kmod * fvk  / gamma_M

    # For this simple test we take:
    #   fm_x_d = fm_y_d = fm_d
    #   fv_x_d = fv_y_d = fv_d
    strengths = DesignStrengths(
        ft0_d=ft0_d,
        fc0_d=fc0_d,
        fm_x_d=fm_d,
        fm_y_d=fm_d,
        fv_x_d=fv_d,
        fv_y_d=fv_d,
        # No perpendicular strengths provided in this test
        ft90_d=None,
        fc90_d=None,
        fv_d=fv_d,   # generic shear for torsion
        km=1.0,      # interaction factor for bending – set as needed
    )

    # ----------------------------------------------------------
    # 3) Internal forces: Only torsion, to mimic WoodExpress report
    # ----------------------------------------------------------
    M_tor_kNm = 1.0       # kNm
    M_tor_Nmm = M_tor_kNm * 1e6  # convert to Nmm

    forces = InternalForces(
        N=1000.0,
        Mx=1000000.0,
        My=1000000.0,
        Vx=1000000.0,
        Vy=10000.0,
        T=M_tor_Nmm,
    )

    # ----------------------------------------------------------
    # 4) Run checks
    # ----------------------------------------------------------
    summary = run_ec5_basic_checks(
        shape=shape,
        dims=dims,
        forces=forces,
        strengths=strengths,
    )

    # ----------------------------------------------------------
    # 5) Print results
    # ----------------------------------------------------------
    print("=" * 60)
    print(" EC5 Timber Cross-Section Checks – Test Example")
    print("=" * 60)
    print(f"Shape        : {summary.shape}")
    print(f"Dims (mm)    : {summary.dims}")
    print(f"Internal N,M,V,T (N, Nmm, N):")
    print(f"  N  = {forces.N:.3f}")
    print(f"  Mx = {forces.Mx:.3f}")
    print(f"  My = {forces.My:.3f}")
    print(f"  Vx = {forces.Vx:.3f}")
    print(f"  Vy = {forces.Vy:.3f}")
    print(f"  T  = {forces.T:.3f}")
    print("-" * 60)

    for chk in summary.checks:
        flag = "OK " if chk.ok else "FAIL"
        print(f"[{flag}] {chk.name:28s} η = {chk.utilization:7.3f}  (limit {chk.limit})")
        if chk.name == "torsion":
            # Print some extra torsion info if available
            tau_tor = chk.extra.get("tau_tor", None)
            fv_design = chk.extra.get("fv_design", None)
            k_shape = chk.extra.get("k_shape", None)
            if tau_tor is not None and fv_design is not None and k_shape is not None:
                print(
                    f"      τ_tor = {tau_tor:.3f} MPa, "
                    f"k_shape = {k_shape:.3f}, "
                    f"f_v,d = {fv_design:.3f} MPa"
                )
        # Uncomment if you want full details:
        # print("     ", chk.details)

    print("-" * 60)
    gov = summary.governing_check
    status = "OK" if gov.ok else "FAIL"
    print(f"GOVERNING CHECK: {gov.name}  η = {gov.utilization:.3f}  → {status}")
    print(f"Details: {gov.details}")
    print("=" * 60)


if __name__ == "__main__":
    main()
