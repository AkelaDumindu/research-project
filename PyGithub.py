"""
collect_github_data.py
--------------------------------
Collects repository metadata, issues, commits, and contributors from GitHub.
Safely handles rate limits, pagination, and data formatting for ML-based issue assignment projects.
"""

from github import Github, GithubException, RateLimitExceededException
import os
import json
import time
from dotenv import load_dotenv
from tqdm import tqdm

# ------------------------------------------------------
# Load GitHub token securely
# ------------------------------------------------------
load_dotenv()
token = os.getenv("GITHUB_TOKEN")

if not token:
    raise ValueError("❌ GitHub token not found! Please add it to your .env file as GITHUB_TOKEN=YOUR_TOKEN")

# Authenticate
g = Github(token)

# ------------------------------------------------------
# Configuration
# ------------------------------------------------------
REPO_NAME = "tensorflow/tensorflow"  # <--- change to your target repo
OUTPUT_DIR = "data"
ISSUE_LIMIT = 200        # Adjust limits if you want to collect more (use 0 for unlimited)
COMMIT_LIMIT = 200
CONTRIBUTOR_LIMIT = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------
# Helper functions
# ------------------------------------------------------
def save_json(data, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"💾 Saved {filename} ({len(data) if isinstance(data, list) else '1'} records)")

def handle_rate_limit():
    rate_limit = g.get_rate_limit()
    core = rate_limit.resources.core
    if core.remaining == 0:
        reset_time = core.reset.timestamp()
        sleep_for = reset_time - time.time() + 5
        if sleep_for > 0:
            print(f"⚠️ Rate limit reached. Sleeping for {sleep_for/60:.1f} minutes...")
            time.sleep(sleep_for)


# ------------------------------------------------------
# Step 1: Connect to Repository
# ------------------------------------------------------
try:
    repo = g.get_repo(REPO_NAME)
    print(f"✅ Connected to repository: {repo.full_name}")
except GithubException as e:
    raise SystemExit(f"❌ Error: Cannot access repo '{REPO_NAME}' - {e}")

# ------------------------------------------------------
# Step 2: Repository Metadata
# ------------------------------------------------------
repo_data = {
    "name": repo.full_name,
    "description": repo.description,
    "language": repo.language,
    "created_at": repo.created_at.isoformat(),
    "updated_at": repo.updated_at.isoformat(),
    "open_issues": repo.open_issues_count,
    "stargazers": repo.stargazers_count,
    "forks": repo.forks_count,
    "watchers": repo.watchers_count,
    "url": repo.html_url
}
save_json(repo_data, "repo_meta.json")

# ------------------------------------------------------
# Step 3: Collect Issues (excluding Pull Requests)
# ------------------------------------------------------
print("\n📥 Fetching issues...")
issues_data = []
issues = repo.get_issues(state="all")

for i, issue in enumerate(tqdm(issues, desc="Issues")):
    if ISSUE_LIMIT and i >= ISSUE_LIMIT:
        break
    if issue.pull_request is not None:
        continue  # Skip pull requests
    handle_rate_limit()
    try:
        issues_data.append({
            "number": issue.number,
            "title": issue.title or "",
            "body": issue.body or "",
            "text": (issue.title or "") + " " + (issue.body or ""),
            "labels": [label.name for label in issue.labels],
            "state": issue.state,
            "created_at": issue.created_at.isoformat() if issue.created_at else None,
            "closed_at": issue.closed_at.isoformat() if issue.closed_at else None,
            "assignee": issue.assignee.login if issue.assignee else None,
            "assignees": [a.login for a in issue.assignees] if issue.assignees else [],
            "comments": issue.comments,
            "url": issue.html_url
        })
    except Exception as e:
        print(f"⚠️ Skipped issue {issue.number} due to: {e}")

save_json(issues_data, "issues.json")

# ------------------------------------------------------
# Step 4: Collect Commits
# ------------------------------------------------------
print("\n📥 Fetching commits...")
commits_data = []
commits = repo.get_commits()

for i, commit in enumerate(tqdm(commits, desc="Commits")):
    if COMMIT_LIMIT and i >= COMMIT_LIMIT:
        break
    handle_rate_limit()
    try:
        commits_data.append({
            "sha": commit.sha,
            "message": commit.commit.message,
            "author": commit.author.login if commit.author else "Unknown",
            "date": commit.commit.author.date.isoformat(),
            "url": commit.html_url
        })
    except Exception as e:
        print(f"⚠️ Skipped commit due to: {e}")

save_json(commits_data, "commits.json")

# ------------------------------------------------------
# Step 5: Collect Contributors
# ------------------------------------------------------
print("\n📥 Fetching contributors...")
contributors_data = []
contributors = repo.get_contributors()

for i, user in enumerate(tqdm(contributors, desc="Contributors")):
    if CONTRIBUTOR_LIMIT and i >= CONTRIBUTOR_LIMIT:
        break
    handle_rate_limit()
    try:
        contributors_data.append({
            "login": user.login,
            "contributions": user.contributions,
            "profile_url": user.html_url,
            "type": user.type
        })
    except Exception as e:
        print(f"⚠️ Skipped contributor due to: {e}")

save_json(contributors_data, "contributors.json")

# ------------------------------------------------------
# Step 6: Combine All Data
# ------------------------------------------------------
final_output = {
    "repository": repo_data,
    "issues": issues_data,
    "commits": commits_data,
    "contributors": contributors_data
}

save_json(final_output, "repo_full_data.json")

print("\n✅ All data collection completed successfully!")
print("Files saved in the 'data/' folder.")
