KDEF_TABLE = {
    "Natural timber": {
        "standard": "EN 14081-1",
        "values": {1: 0.60, 2: 0.80, 3: 2.00},
    },
    "Glued laminated timber": {
        "standard": "EN 14080",
        "values": {1: 0.60, 2: 0.80, 3: 2.00},
    },
    "Laminated veneer lumber (LVL)": {
        "standard": "EN 14374, EN 14279",
        "values": {1: 0.60, 2: 0.80, 3: 2.00},
    },
    "Plywood": {
        "standard": "EN 636",
        "values": {"Part 1": 0.80, "Part 2": 1.00, "Part 3": 2.50},
    },
    "Oriented strand board (OSB)": {
        "standard": "EN 300",
        "values": {"OSB/2": 2.25, "OSB/3": 1.50, "OSB/4": 2.25},
    },
    "Particleboard": {
        "standard": "EN 312",
        "values": {"Part 4": 2.25, "Part 5": 3.00, "Part 6": 1.50, "Part 7": 1.25},
    },
    "Hardboard": {
        "standard": "EN 622-2",
        "values": {
            "HB.LA": 2.25,
            "HB.HLA1": 3.00,
            "HB.HLA2": 3.00,
        },
    },
    "Medium board": {
        "standard": "EN 622-3",
        "values": {
            "MBH.LA1": 3.00,
            "MBH.LA2": 3.00,
            "MBH.HLS1": 4.00,
            "MBH.HLS2": 4.00,
        },
    },
    "MDF board": {
        "standard": "EN 622-5",
        "values": {"MDF.LA": 2.25, "MDF.HLS": 3.00},
    },
}
def get_kdef(material: str, service_class: int = 1) -> float:
    """Return kdef value for given material and service class."""
    entry = KDEF_TABLE[material]["values"]
    # Choose numeric key if available, else fallback (e.g. for plywood parts)
    if service_class in entry:
        return entry[service_class]
    # fallback: pick highest matching numeric
    for k, v in entry.items():
        if isinstance(k, (int, float)) and k == service_class:
            return v
    # or average if no match
    return sum(entry.values()) / len(entry)

# Example:
if __name__ == "__main__":
    print(get_kdef("Natural timber", 2))  # → 0.80
