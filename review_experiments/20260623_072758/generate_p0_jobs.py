#!/usr/bin/env python3
import csv
import os
import sys

out = sys.argv[1]
rows = []
counter = 0
datasets = ["ETTm2", "ECL", "Traffic"]
horizons = [96, 336, 720]
variants = ["Full", "NoSDE", "GlobalOnly", "FixedGate", "NoDynFilt", "WrongPhase", "DynamicOnly"]
multi_seed = {"Full", "NoSDE", "NoDynFilt"}

for dataset in datasets:
    for horizon in horizons:
        for variant in variants:
            seeds = [2021, 2022, 2023] if variant in multi_seed else [2021]
            for seed in seeds:
                counter += 1
                job_id = f"{counter:03d}_{dataset}_pl{horizon}_{variant}_s{seed}"
                rows.append([job_id, dataset, horizon, variant, seed, "pending"])

os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["job_id", "dataset", "horizon", "variant", "seed", "status"])
    w.writerows(rows)
print(len(rows))
