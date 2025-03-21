import os
import csv
import argparse
import shutil
import subprocess
import logging
import pandas as pd
import psutil
from git import Repo, GitCommandError
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

INTERMEDIATE_FAILURES_FILE = "intermediate_failed_repos.csv"
PROGRESS_LOG_FILE = "progress.log"
RUNTIME_LOG_FILE = "runtime.log"
PER_THREAD_MEMORY_MB = 800

lock = Lock()
completed_lock = Lock()
failed_repos = []
completed_count = 0

# Setup logging
logging.basicConfig(
    filename=RUNTIME_LOG_FILE,
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger()

def get_available_memory_mb():
    return psutil.virtual_memory().available // (1024 * 1024)

def compute_optimal_threads(per_thread_mb=400, cap=16, user_requested=None):
    available_mb = get_available_memory_mb()
    max_threads = available_mb // per_thread_mb
    if user_requested:
        return min(user_requested, max_threads)
    return min(max(1, max_threads), cap)

def is_merge_commit(commit):
    return len(commit.parents) > 1

def shallow_clone_by_date(repo_url, clone_dir, since_date_str):
    subprocess.run([
        "git", "clone",
        "--shallow-since", since_date_str,
        "--single-branch",
        repo_url, clone_dir
    ], check=True)

def fetch_commits(repo_url, since_date_str, output_file, clone_dir):
    try:
        if not os.path.exists(clone_dir):
            shallow_clone_by_date(repo_url, clone_dir, since_date_str)

        repo = Repo(clone_dir)
        since_date = datetime.strptime(since_date_str, "%Y-%m-%d")

        with open(output_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Hash", "Date", "Author", "Message", "IsMerge"])

            for commit in repo.iter_commits():
                commit_date = datetime.fromtimestamp(commit.committed_date)
                if commit_date >= since_date:
                    writer.writerow([
                        commit.hexsha,
                        commit_date.strftime("%Y-%m-%d %H:%M:%S"),
                        commit.author.name,
                        commit.message.strip(),
                        is_merge_commit(commit)
                    ])
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Clone failed for {repo_url}: {e}")
        return False
    except (GitCommandError, Exception) as e:
        logger.error(f"Error processing {repo_url}: {e}")
        return False

def append_failed_repo(index, row):
    file_exists = os.path.exists(INTERMEDIATE_FAILURES_FILE)
    with lock:
        with open(INTERMEDIATE_FAILURES_FILE, "a", newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=row.index.tolist())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())
            logger.info(f"[{index}] Written failed repo to intermediate log")

def process_repo(index, row, args):
    url = row['URL']
    repo_name = row.get('Repository Name', f'repo_{index}')
    clean_name = repo_name.replace('/', '_').replace(' ', '_')
    clone_dir = f"repos/{clean_name}_{index}"
    output_file = f"commits_{clean_name}_{index}.csv"

    logger.info(f"[{index}] Processing: {repo_name} ({url})")
    success = fetch_commits(url, args.since, output_file, clone_dir)

    if not success:
        logger.warning(f"[{index}] Failed: Logging to intermediate failure file")
        with lock:
            failed_repos.append((index, row))
            append_failed_repo(index, row)

    if args.cleanup and os.path.exists(clone_dir):
        logger.info(f"[{index}] Cleanup: Removing {clone_dir}")
        shutil.rmtree(clone_dir)

    global completed_count
    with completed_lock:
        completed_count += 1
        if completed_count % 50 == 0:
            with open(PROGRESS_LOG_FILE, "a") as log_file:
                log_file.write(f"{completed_count} repositories processed.\n")
            logger.info(f"{completed_count} repositories processed.")

def retry_failed(args):
    if not os.path.exists(INTERMEDIATE_FAILURES_FILE):
        logger.info("No intermediate failures found to retry.")
        return

    df_failed = pd.read_csv(INTERMEDIATE_FAILURES_FILE)
    still_failed = []

    logger.info(f"Retrying {len(df_failed)} failed repositories...")

    for index, row in df_failed.iterrows():
        url = row['URL']
        repo_name = row.get('Repository Name', f'repo_{index}')
        clean_name = repo_name.replace('/', '_').replace(' ', '_')
        clone_dir = f"repos/{clean_name}_{index}"
        output_file = f"commits_{clean_name}_{index}.csv"

        logger.info(f"[Retry {index}] Retrying: {repo_name} ({url})")
        success = fetch_commits(url, args.since, output_file, clone_dir)

        if not success:
            logger.warning(f"[Retry {index}] Failed again")
            still_failed.append((index, row))
        else:
            logger.info(f"[Retry {index}] Success on retry")

        if args.cleanup and os.path.exists(clone_dir):
            logger.info(f"[Retry {index}] Cleanup: Removing {clone_dir}")
            shutil.rmtree(clone_dir)

    if still_failed:
        logger.warning(f"{len(still_failed)} repositories still failed after retry. Writing to failed_repos.csv")
        with open("failed_repos.csv", "w", newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=still_failed[0][1].index.tolist())
            writer.writeheader()
            for _, row in still_failed:
                writer.writerow(row)
    else:
        logger.info("All failed repositories succeeded on retry.")
        if os.path.exists("failed_repos.csv"):
            os.remove("failed_repos.csv")

    if os.path.exists(INTERMEDIATE_FAILURES_FILE):
        os.remove(INTERMEDIATE_FAILURES_FILE)

    if still_failed:
        with open(INTERMEDIATE_FAILURES_FILE, "w", newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=still_failed[0][1].index.tolist())
            writer.writeheader()
            for _, row in still_failed:
                writer.writerow(row)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--since', required=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--cleanup', action='store_true')
    parser.add_argument('--workers', type=int, default=None)

    args = parser.parse_args()

    if os.path.exists(PROGRESS_LOG_FILE):
        os.remove(PROGRESS_LOG_FILE)

    df = pd.read_csv(args.csv)
    subset = df.iloc[args.start:args.end]

    optimal_workers = compute_optimal_threads(user_requested=args.workers)
    logger.info(f"Using {optimal_workers} worker threads based on available memory.")

    with ThreadPoolExecutor(max_workers=optimal_workers) as executor:
        futures = [executor.submit(process_repo, idx, row, args) for idx, row in subset.iterrows()]
        for future in as_completed(futures):
            future.result()

    retry_failed(args)

if __name__ == "__main__":
    main()
