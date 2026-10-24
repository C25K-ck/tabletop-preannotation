"""Fixed three-class taxonomy. Order is the YOLO class id."""

CLASSES = [
    {
        "id": 0,
        "name": "item",
        "label": "物品",
        "color": "#22c55e",
        "prompts": [
            "A4 paper",
            "white sheet of paper",
            "cardboard box",
            "parcel",
            "small object",
        ],
    },
    {
        "id": 1,
        "name": "robot",
        "label": "机械臂",
        "color": "#3b82f6",
        "prompts": [
            "machine with lights",
            "robot vacuum",
            "mobile robot with an arm",
            "robot arm",
        ],
    },
    {
        "id": 2,
        "name": "hand",
        "label": "人手",
        "color": "#ef4444",
        "prompts": [
            "human hand",
        ],
    },
]

CLASS_NAMES = [item["name"] for item in CLASSES]
CLASS_BY_ID = {item["id"]: item for item in CLASSES}


def prompt_catalog():
    """Flat prompt list. Several prompts may share one class id."""
    catalog = []
    for item in CLASSES:
        for prompt in item["prompts"]:
            catalog.append({"class_id": item["id"], "prompt": prompt})
    return catalog
