import os, json, pandas as pd
from glob import glob

DATA_DIR = "../data"
OUTPUT_FILE = os.path.join(DATA_DIR, "combined_dataset.csv")

rows = []
for repo_folder in glob(os.path.join(DATA_DIR, "*_*")):
    issues_path = os.path.join(repo_folder, "issues.json")
    if not os.path.exists(issues_path): continue
    with open(issues_path, "r", encoding="utf-8") as f:
        issues = json.load(f)
    for i in issues:
        if i.get("assignee"):  # only use labeled issues
            rows.append({
                "repo": os.path.basename(repo_folder),
                "title": i.get("title", ""),
                "body": i.get("body", ""),
                "text": f"{i.get('title', '')} {i.get('body', '')}",
                "assignee": i.get("assignee"),
                "created_at": i.get("created_at"),
                "labels": ", ".join(i.get("labels", []))
            })

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Combined dataset saved to: {OUTPUT_FILE}, rows: {len(df)}")
