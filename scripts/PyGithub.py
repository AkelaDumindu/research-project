"""
collect_multiple_repos_fixed_v2.py
----------------------------------
Handles NamedUser serialization correctly.
Converts PyGithub user objects into string logins before JSON export.
"""

import os
import json
import time
import random
from dotenv import load_dotenv
from tqdm import tqdm
from github import Github, GithubException

# Optional Auth handling (suppresses warnings)
try:
    from github import Auth
    _HAS_AUTH = True
except Exception:
    _HAS_AUTH = False

# Configuration 
OUTPUT_DIR = "../data"
ISSUE_LIMIT = 500
COMMIT_LIMIT = 500
CONTRIBUTOR_LIMIT = 500
MAX_RETRY = 3
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load token
load_dotenv()
TOKEN = os.getenv("GITHUB_TOKEN")
if not TOKEN:
    raise SystemExit("Please create a .env file with GITHUB_TOKEN=your_token")

# Create GitHub client
if _HAS_AUTH:
    g = Github(auth=Auth.Token(TOKEN))
else:
    g = Github(TOKEN)

def safe_login(user):
    """Safely get a username string from a NamedUser or None."""
    try:
        return user.login if user else None
    except Exception:
        return None

def safe_iso(dt):
    if not dt:
        return None
    try:
        return dt.isoformat()
    except Exception:
        return str(dt)

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def handle_rate_limit():
    """Pause if API rate limit reached."""
    try:
        remaining, _ = g.rate_limiting
        if remaining < 1:
            reset_time = g.rate_limiting_resettime
            sleep_for = reset_time - time.time() + 5
            if sleep_for > 0:
                print(f"Rate limit hit. Sleeping {sleep_for/60:.1f} minutes...")
                time.sleep(sleep_for)
    except Exception:
        time.sleep(5)

def collect_repo(repo_name):
    print(f"\n📂 Collecting from: {repo_name}")
    repo_folder = os.path.join(OUTPUT_DIR, repo_name.replace("/", "_"))
    os.makedirs(repo_folder, exist_ok=True)

    try:
        repo = g.get_repo(repo_name)
    except GithubException as e:
        print(f"Cannot access {repo_name}: {e}")
        return

    # ---- Metadata ----
    repo_meta = {
        "full_name": repo.full_name,
        "description": repo.description,
        "language": repo.language,
        "created_at": safe_iso(repo.created_at),
        "updated_at": safe_iso(repo.updated_at),
        "stargazers": repo.stargazers_count,
        "forks": repo.forks_count,
        "url": repo.html_url,
        "organization": safe_login(repo.organization)
    }
    save_json(repo_meta, os.path.join(repo_folder, "repo_meta.json"))

    # ---- Issues ----
    issues_data = []
    for i, issue in enumerate(tqdm(repo.get_issues(state="all"), desc=f"Issues - {repo_name}")):
        if ISSUE_LIMIT and i >= ISSUE_LIMIT:
            break
        if issue.pull_request is not None:
            continue
        handle_rate_limit()
        try:
            issues_data.append({
                "number": issue.number,
                "title": issue.title or "",
                "body": issue.body or "",
                "text": f"{issue.title or ''} {issue.body or ''}",
                "labels": [l.name for l in issue.labels],
                "state": issue.state,
                "created_at": safe_iso(issue.created_at),
                "closed_at": safe_iso(issue.closed_at),
                "assignee": safe_login(issue.assignee),
                "assignees": [safe_login(a) for a in issue.assignees],
                "comments": issue.comments,
                "url": issue.html_url
            })
        except Exception as e:
            print(f"Skipped issue #{issue.number}: {e}")
    save_json(issues_data, os.path.join(repo_folder, "issues.json"))

    # ---- Commits ----
    commits_data = []
    for i, commit in enumerate(tqdm(repo.get_commits(), desc=f"Commits - {repo_name}")):
        if COMMIT_LIMIT and i >= COMMIT_LIMIT:
            break
        handle_rate_limit()
        try:
            commits_data.append({
                "sha": commit.sha,
                "message": commit.commit.message,
                "author": safe_login(commit.author),
                "committer": safe_login(commit.committer),
                "date": safe_iso(commit.commit.author.date if commit.commit.author else None),
                "url": commit.html_url
            })
        except Exception as e:
            print(f"Skipped commit: {e}")
    save_json(commits_data, os.path.join(repo_folder, "commits.json"))

    # ---- Contributors ----
    contributors_data = []
    for i, user in enumerate(tqdm(repo.get_contributors(), desc=f"Contributors - {repo_name}")):
        if CONTRIBUTOR_LIMIT and i >= CONTRIBUTOR_LIMIT:
            break
        handle_rate_limit()
        try:
            contributors_data.append({
                "login": safe_login(user),
                "contributions": user.contributions,
                "profile_url": user.html_url
            })
        except Exception as e:
            print(f"Skipped contributor: {e}")
    save_json(contributors_data, os.path.join(repo_folder, "contributors.json"))

    # ---- Collaborators (skip if forbidden) ----
    collaborators_data = []
    try:
        for user in repo.get_collaborators():
            collaborators_data.append({
                "login": safe_login(user),
                "profile_url": user.html_url
            })
    except GithubException:
        print("Collaborators not accessible (likely permission issue)")
    save_json(collaborators_data, os.path.join(repo_folder, "collaborators.json"))

    # ---- Org Members ----
    org_members_data = []
    if repo.organization:
        try:
            for member in repo.organization.get_members():
                org_members_data.append({
                    "login": safe_login(member),
                    "profile_url": member.html_url
                })
        except GithubException:
            print("Organization members not accessible")
    save_json(org_members_data, os.path.join(repo_folder, "org_members.json"))

    # ---- Combine ----
    combined = {
        "repository": repo_meta,
        "issues": issues_data,
        "commits": commits_data,
        "contributors": contributors_data,
        "collaborators": collaborators_data,
        "org_members": org_members_data
    }
    save_json(combined, os.path.join(repo_folder, "repo_full_data.json"))

    print(f"✅ Completed collection for {repo_name}")

# ---- Main ----
if __name__ == "__main__":
    if not os.path.exists("../repos.txt"):
        raise SystemExit("Missing repos.txt file with repo names")

    with open("../repos.txt", "r") as f:
        repos = [r.strip() for r in f if r.strip()]

    print(f"Found {len(repos)} repositories in repos.txt")

    for repo in repos:
        try:
            collect_repo(repo)
        except Exception as e:
            print(f"Error collecting {repo}: {e}")

    print("\n🎉 All repositories processed successfully!")
