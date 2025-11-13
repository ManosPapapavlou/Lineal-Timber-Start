from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Any, Optional, Tuple

import json
from pathlib import Path

from flask import Flask, render_template, request

# ---- our packages (under data/ and fuctions_timber/) ----------------------
from data import timber_strength_classes as tsc
from data.kmod_table import KMOD_TABLE, LOAD_CLASSES
from data.kdef_table import KDEF_TABLE  # not used yet, but imported for future
from fuctions_timber.properties import get_section_properties
from fuctions_timber.checks import (
    InternalForces,
    DesignStrengths,
    run_ec5_basic_checks,
)

app = Flask(__name__)

# -------------------------------------------------------------------------
# CONFIG / CHOICES
# -------------------------------------------------------------------------

SHAPES = [
    "rectangle",
    "circle",
    "ellipse",
    "rhs",
    "chs",
    "i_section",
    "t_section",
    "trapezoid",
    "triangle",
]


def _get_all_timber_classes() -> list[str]:
    """Read all timber class names from timber_strength_classes.json."""
    data = json.loads(tsc.TIMBER_PATH.read_text(encoding="utf-8"))
    return sorted(list(data["data"].keys()))


TIMBER_CLASSES = _get_all_timber_classes()

MATERIAL_TYPES = [
    "Natural timber",
    "Glued laminated timber",
    "LVL",
    "Plywood",
    "OSB",
]

SERVICE_CLASSES = [1, 2, 3]
LOAD_DURATION_CLASSES = LOAD_CLASSES  # ["Permanent", "Long-term", ...]


# -------------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------------

def compute_kmod(
    material_type: str,
    service_class: int,
    load_duration: str,
) -> float:
    """
    Get kmod from KMOD_TABLE given:
      - material type (e.g. "Natural timber")
      - service class (1, 2, 3)
      - load duration ("Permanent", "Long-term", ...)
    """
    mat = KMOD_TABLE.get(material_type)
    if mat is None:
        raise ValueError(f"Unknown material type for kmod: {material_type}")

    if service_class not in mat["values"]:
        raise ValueError(f"Service class {service_class} not in KMOD table for {material_type}")

    values = mat["values"][service_class]
    try:
        idx = LOAD_DURATION_CLASSES.index(load_duration)
    except ValueError:
        raise ValueError(f"Unknown load duration: {load_duration}")

    return float(values[idx])


def default_gamma_M(material_type: str) -> float:
    """Choose a default γM based on material type."""
    s = material_type.lower()
    if "glued" in s or "laminated" in s:
        return 1.25  # typical for glulam
    return 1.30      # typical for solid timber


def build_design_strengths(
    timber_class: str,
    material_type: str,
    service_class: int,
    load_duration: str,
    gamma_M: Optional[float] = None,
    km: float = 0.7,
) -> Tuple[DesignStrengths, float, float]:
    """
    Build DesignStrengths from:
      - timber strength class (C24, GL28h, etc.)
      - Kmod (from material, service class, duration)
      - gamma_M (partial factor)

    Returns:
        strengths, kmod, gamma_M_used
    """
    props = tsc.get_timber_class(timber_class)  # dict with fmk, ft0k, fc0k, fvk,...

    fm_k = float(props.get("fmk", 0.0))    # bending
    ft0_k = float(props.get("ft0k", 0.0))  # tension parallel
    fc0_k = float(props.get("fc0k", 0.0))  # compression parallel
    fv_k = float(props.get("fvk", 0.0))    # shear

    kmod = compute_kmod(material_type, service_class, load_duration)

    if gamma_M is None or gamma_M <= 0.0:
        gamma_M_used = default_gamma_M(material_type)
    else:
        gamma_M_used = gamma_M

    factor = kmod / gamma_M_used

    fm_d = fm_k * factor
    ft0_d = ft0_k * factor
    fc0_d = fc0_k * factor
    fv_d = fv_k * factor

    # assume same bending/shear capacities in x/y for now
    strengths = DesignStrengths(
        ft0_d=ft0_d,
        fc0_d=fc0_d,
        fm_x_d=fm_d,
        fm_y_d=fm_d,
        fv_x_d=fv_d,
        fv_y_d=fv_d,
        fv_d=fv_d,  # for torsion
        km=km,
    )

    return strengths, kmod, gamma_M_used


def parse_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def collect_dims_from_form(shape: str, form) -> Dict[str, float]:
    """
    Collect section dimensions from the form depending on shape.
    Units must be consistent with properties.py (e.g. mm).
    """
    s = shape.lower()
    dims: Dict[str, float] = {}

    def f(name: str) -> float:
        return parse_float(form.get(name, "") or 0.0, 0.0)

    if s == "rectangle":
        dims["b"] = f("b")
        dims["h"] = f("h")
    elif s == "circle":
        dims["d"] = f("d")
    elif s == "ellipse":
        dims["a"] = f("a")
        dims["b"] = f("b")
    elif s == "rhs":
        dims["b"] = f("b")
        dims["h"] = f("h")
        dims["t"] = f("t")
    elif s == "chs":
        dims["d"] = f("d")
        dims["t"] = f("t")
    elif s == "i_section":
        dims["h"] = f("h")
        dims["b"] = f("b")
        dims["tw"] = f("tw")
        dims["tf"] = f("tf")
    elif s == "t_section":
        dims["h"] = f("h")
        dims["b"] = f("b")
        dims["tw"] = f("tw")
        dims["tf"] = f("tf")
    elif s == "trapezoid":
        dims["b1"] = f("b1")
        dims["b2"] = f("b2")
        dims["h"] = f("h")
    elif s == "triangle":
        dims["b"] = f("b")
        dims["h"] = f("h")
    else:
        raise ValueError(f"Unsupported shape in form: {shape}")

    return dims


# -------------------------------------------------------------------------
# ROUTES
# -------------------------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    section_props = None
    strengths_dict = None
    errors: list[str] = []

    # defaults
    selected_shape = "rectangle"
    selected_class = "C24" if "C24" in TIMBER_CLASSES else (TIMBER_CLASSES[0] if TIMBER_CLASSES else "")
    selected_material = "Natural timber"
    selected_service = 1
    selected_load_class = "Medium-term"
    gamma_default = default_gamma_M(selected_material)

    form_values: Dict[str, Any] = {}

    if request.method == "POST":
        # selections
        selected_shape = request.form.get("shape", selected_shape)
        selected_class = request.form.get("timber_class", selected_class)
        selected_material = request.form.get("material_type", selected_material)
        selected_service = int(request.form.get("service_class", selected_service))
        selected_load_class = request.form.get("load_class", selected_load_class)

        # γM from form (0 ή κενό → default)
        gamma_M_in = parse_float(request.form.get("gamma_M", ""), 0.0)

        form_values.update(
            shape=selected_shape,
            timber_class=selected_class,
            material_type=selected_material,
            service_class=selected_service,
            load_class=selected_load_class,
            gamma_M=gamma_M_in if gamma_M_in > 0 else "",
        )

        # dimensions
        try:
            dims = collect_dims_from_form(selected_shape, request.form)
        except Exception as e:
            errors.append(f"Error reading dimensions: {e}")
            dims = {}

        # store dims so they persist in the form
        form_values.update(dims)

        # forces
        forces = InternalForces(
            N=parse_float(request.form.get("N", "0")),
            Mx=parse_float(request.form.get("Mx", "0")),
            My=parse_float(request.form.get("My", "0")),
            Vx=parse_float(request.form.get("Vx", "0")),
            Vy=parse_float(request.form.get("Vy", "0")),
            T=parse_float(request.form.get("T", "0")),
        )
        form_values.update(
            N=forces.N,
            Mx=forces.Mx,
            My=forces.My,
            Vx=forces.Vx,
            Vy=forces.Vy,
            T=forces.T,
        )

        try:
            strengths, kmod_val, gamma_used = build_design_strengths(
                timber_class=selected_class,
                material_type=selected_material,
                service_class=selected_service,
                load_duration=selected_load_class,
                gamma_M=gamma_M_in if gamma_M_in > 0 else None,
                km=0.7,
            )
            strengths_dict = asdict(strengths)
            # βάζουμε και kmod, γM για να τα δείχνουμε στον πίνακα
            strengths_dict["kmod"] = kmod_val
            strengths_dict["gamma_M"] = gamma_used

            result = run_ec5_basic_checks(
                shape=selected_shape,
                dims=dims,
                forces=forces,
                strengths=strengths,
            )

            section_props = get_section_properties(selected_shape, **dims)

            gamma_default = gamma_used  # για να φαίνεται στο form μετά το POST

        except Exception as e:
            errors.append(str(e))

    return render_template(
        "index.html",
        shapes=SHAPES,
        timber_classes=TIMBER_CLASSES,
        material_types=MATERIAL_TYPES,
        service_classes=SERVICE_CLASSES,
        load_classes=LOAD_DURATION_CLASSES,
        selected_shape=selected_shape,
        selected_class=selected_class,
        selected_material=selected_material,
        selected_service=selected_service,
        selected_load_class=selected_load_class,
        gamma_default=gamma_default,
        form_values=form_values,
        result=result,
        section_props=section_props,
        strengths=strengths_dict,
        errors=errors,
    )


if __name__ == "__main__":
    app.run(debug=True)
