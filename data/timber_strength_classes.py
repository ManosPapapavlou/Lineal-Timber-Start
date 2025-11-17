import json
from pathlib import Path

TIMBER_PATH = Path(__file__).with_name("timber_strength_classes.json")

def get_timber_class(cls: str) -> dict:
    data = json.loads(TIMBER_PATH.read_text(encoding="utf-8"))
    cols = data["columns"]          # includes "class"
    vals = [cls] + data["data"][cls]  # prepend the class string
    return dict(zip(cols, vals))


# Example
if __name__ == "__main__":
    c24 = get_timber_class("C24")
    print("C24 timber properties:")
    for k, v in c24.items():
        print(f"{k:10s} = {v}")
