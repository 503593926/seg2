from seg2.ingest.utils.annotation2mask import ClassInfo

CLASSES: list[ClassInfo] = [
    {"id": 1, "label": "invasive cancer", "rgb": (236, 28, 36)},
    {"id": 2, "label": "cancer in situ", "rgb": (255, 127, 39)},
    {"id": 3, "label": "immune infiltrate", "rgb": (255, 242, 0)},
    {"id": 4, "label": "breast glands", "rgb": (14, 209, 69)},
    {"id": 5, "label": "connective tissue", "rgb": (63, 72, 204)},
    {"id": 6, "label": "adipose tissue", "rgb": (112, 227, 223)},
]
