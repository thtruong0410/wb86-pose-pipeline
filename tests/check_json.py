#!/usr/bin/env python3
import json, math
from pathlib import Path

SLICES = {"body": (0,25), "lhand": (25,46), "rhand": (46,67), "face": (67,86)}

def count_file(p):
    with open(p, "r", encoding="utf-8") as f:
        r = json.load(f)
    kps = r["keypoints"]  # 86 x [x,y,conf]
    def valid(k): 
        x,y,c = k
        return c > 0 and math.isfinite(x) and math.isfinite(y)
    total = sum(valid(k) for k in kps)
    per = {name: sum(valid(k) for k in kps[s:e]) for name,(s,e) in SLICES.items()}
    return total, per

# Single file
p = Path(r"C:\Users\pc\Downloads\CSLR_Data\wb86-pose-pipeline\out\frame0000_person0.json")
print(p, *count_file(p))

# All outputs
for p in Path("out").glob("*_person*.json"):
    total, per = count_file(p)
    print(f"{p}: {total}/86 -> {per}")