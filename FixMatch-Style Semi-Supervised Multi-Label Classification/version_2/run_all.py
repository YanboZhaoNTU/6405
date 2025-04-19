import json
import os
import pandas as pd
from train_and_evaluate import train_and_evaluate

# Load dataset config
with open("config/dataset_config.json", "r") as f:
    dataset_config = json.load(f)

results = []

for name, cfg in dataset_config.items():
    print("=" * 60)
    print(f"🚀 Running: {name}")

    try:
        label_count = cfg["label_count"]
        split_type = cfg["split"]

        if label_count <= 0:
            print(f"⚠️ Skipping {name}: invalid label_count = {label_count}")
            continue

        if split_type == "random":
            path = cfg["file"]
        else:
            path = cfg  # train_file and test_file inside dict

        metrics = train_and_evaluate(name, path, label_count, split_type=split_type)
        results.append({"Dataset": name, **metrics})

    except Exception as e:
        print(f"❌ Failed on {name}: {e}")

# Save results as CSV
os.makedirs("results", exist_ok=True)
df = pd.DataFrame(results)
df.to_csv("results/metrics.csv", index=False)
print("\n✅ All datasets completed. Results saved to results/metrics.csv")
