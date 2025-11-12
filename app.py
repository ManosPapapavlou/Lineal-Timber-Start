from flask import Flask, render_template, request, jsonify
from fuctions_timber.checks_beam import check_beam_ULS_rectangular, BeamULSResult
from data.timber_strength_classes import get_timber_class
from data.timber_strength_classes import TIMBER_CLASSES

...

@app.route("/")
def index():
    timber_classes = list(TIMBER_CLASSES.keys())
    return render_template(
        "index.html",
        section_inputs=SECTION_INPUTS,
        section_types=list(SECTION_INPUTS.keys()),
        timber_classes=timber_classes,
        materials=[
            "Natural timber",
            "Glued laminated timber",
            "Laminated veneer lumber (LVL)",
        ],
    )

app = Flask(__name__)

# -------------------------------------------------------------------
# Section input templates (for the front-end dynamic UI)
# -------------------------------------------------------------------
SECTION_INPUTS = {
    "Rectangular": ["b_mm", "h_mm"],
    "Circular": ["d_mm"],
    "RHS/SHS": ["b_mm", "h_mm", "t_mm"],
    "CHS": ["d_mm", "t_mm"],
    "I/H": ["h_mm", "b_mm", "tw_mm", "tf_mm"],
    "T-section": ["h_mm", "b_mm", "tw_mm", "tf_mm"],
    "Trapezoid": ["b1_mm", "b2_mm", "h_mm"],
    "Triangle": ["b_mm", "h_mm"]
}

# Only rectangular implemented for now
IMPLEMENTED_CHECKS = {"Rectangular": check_beam_ULS_rectangular}


# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------
@app.route("/")
def index():
    timber_classes = list(TIMBER_CLASSES.keys())
    return render_template(
        "index.html",
        section_inputs=SECTION_INPUTS,
        section_types=list(SECTION_INPUTS.keys()),
        timber_classes=timber_classes,
        materials=[
            "Natural timber",
            "Glued laminated timber",
            "Laminated veneer lumber (LVL)",
        ],
    )


@app.route("/api/run_check", methods=["POST"])
def run_check():
    data = request.get_json()
    section_type = data.get("section_type")
    timber_class = data.get("timber_class", "C24")
    material_family = data.get("material", "Natural timber")
    service_class = int(data.get("service_class", 2))
    load_class = data.get("load_class", "Medium-term")

    Ned = float(data.get("Ned_kN", 0))
    Myd = float(data.get("Myd_kNm", 0))
    Mzd = float(data.get("Mzd_kNm", 0))
    Ved = float(data.get("Ved_kN", 0))

    # geometry from request
    geom = {k: float(data[k]) for k in SECTION_INPUTS.get(section_type, []) if k in data}

    # check if function exists
    func = IMPLEMENTED_CHECKS.get(section_type)
    if not func:
        return jsonify(
            {"error": f"The {section_type} section check is not implemented yet."}
        )

    try:
        res = func(
            **geom,
            Ned_kN=Ned,
            Myd_kNm=Myd,
            Mzd_kNm=Mzd,
            Ved_kN=Ved,
            timber_class=timber_class,
            material_family=material_family,
            service_class=service_class,
            load_class=load_class,
        )

        # Convert dataclass → dict
        if isinstance(res, BeamULSResult):
            res = res.__dict__

        return jsonify(res)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
