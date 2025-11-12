# ec5/design/checks_beam.py
from dataclasses import dataclass
from typing import Literal, Dict
from fuctions_timber.properties import get_section_properties
from data.timber_strength_classes import get_timber_class
from data.kmod_table import KMOD_TABLE, LOAD_CLASSES
from data.kdef_table import KDEF_TABLE

MaterialFamily = Literal["Natural timber","Glued laminated timber","Laminated veneer lumber (LVL)",
                         "Plywood","Oriented strand board (OSB)","Particleboard","Hardboard",
                         "Medium board","MDF board"]

GAMMA_M = 1.30  # EC5 Table 2.3 for timber-like products

def get_kmod(material_family: MaterialFamily, service_class: int, load_class: str) -> float:
    return KMOD_TABLE[material_family]["values"][service_class][LOAD_CLASSES.index(load_class)]

def get_kdef(material_family: MaterialFamily, service_class: int) -> float:
    vals = KDEF_TABLE[material_family]["values"]
    if isinstance(vals, dict) and service_class in vals:
        return vals[service_class]
    # plywood/boards use “part” labels; choose a conservative max
    return max(v for v in vals.values())

@dataclass
class BeamULSResult:
    # geometry
    A: float; Wy: float; Wz: float
    # stresses
    sigma_t: float = 0.0
    sigma_c: float = 0.0
    sigma_my: float = 0.0
    sigma_mz: float = 0.0
    tau_v: float = 0.0
    # design strengths
    ft0d: float = 0.0; fc0d: float = 0.0; fmd: float = 0.0; fvd: float = 0.0
    # utilization ratios
    util_tension: float = 0.0
    util_shear: float = 0.0
    util_bending: float = 0.0
    util_NM_tension: float = 0.0
    util_NM_compression: float = 0.0

def check_beam_ULS_rectangular(
    b_mm: float, h_mm: float,
    Ned_kN: float, Myd_kNm: float, Mzd_kNm: float, Ved_kN: float,
    timber_class: str = "C24",
    material_family: MaterialFamily = "Natural timber",
    service_class: int = 2, load_class: str = "Medium-term",
) -> BeamULSResult:
    """Replicates your live-scripts logic for a rectangular beam (mm, kN, kNm)."""
    sec = get_section_properties("rectangle", b=b_mm, h=h_mm)
    mat = get_timber_class(timber_class)
    kmod = get_kmod(material_family, service_class, load_class)

    # design strengths (MPa)
    ft0d = kmod * mat["ft0k"] / GAMMA_M
    fc0d = kmod * mat["fc0k"] / GAMMA_M
    fmd  = kmod * mat["fmk"]  / GAMMA_M
    fvd  = kmod * mat["fvk"]  / GAMMA_M

    # stresses (N/mm² = MPa), inputs are kN, kNm
    A  = sec["A"]                    # mm²
    Wy = sec["Sx_top"] * (h_mm/2)    # since Sx = I/y, Sx_top* y_top = I
    Wz = sec["Sy_right"] * (b_mm/2)

    sigma_t = max(Ned_kN,0.0)*1e3 / A
    sigma_c = max(-Ned_kN,0.0)*1e3 / A
    sigma_my = (Myd_kNm*1e6) / Wy if Wy>0 else 0.0
    sigma_mz = (Mzd_kNm*1e6) / Wz if Wz>0 else 0.0

    # shear – rectangular effective area & 1.5·V/Anetto like your script
    bef = 0.67 * b_mm
    Anetto = bef * h_mm
    tau_v = 1.5 * Ved_kN * 1e3 / Anetto if Anetto>0 else 0.0

    # basic utilizations
    util_tension  = sigma_t/ft0d if ft0d>0 else 0.0
    util_shear    = tau_v/fvd    if fvd>0 else 0.0
    util_bending  = (sigma_my/fmd if fmd>0 else 0.0) + 0.7*(sigma_mz/fmd if fmd>0 else 0.0)

    # N–M interactions (mirrors your two expressions)
    Km = 0.7
    util_NM_tension     = (sigma_t/ft0d if ft0d>0 else 0.0) + (sigma_my/fmd if fmd>0 else 0.0) + Km*(sigma_mz/fmd if fmd>0 else 0.0)
    util_NM_compression = (sigma_c/fc0d if fc0d>0 else 0.0)**2 + (sigma_my/fmd if fmd>0 else 0.0) + Km*(sigma_mz/fmd if fmd>0 else 0.0)

    return BeamULSResult(
        A=A, Wy=Wy, Wz=Wz,
        sigma_t=sigma_t, sigma_c=sigma_c, sigma_my=sigma_my, sigma_mz=sigma_mz, tau_v=tau_v,
        ft0d=ft0d, fc0d=fc0d, fmd=fmd, fvd=fvd,
        util_tension=util_tension, util_shear=util_shear,
        util_bending=util_bending, util_NM_tension=util_NM_tension, util_NM_compression=util_NM_compression
    )
