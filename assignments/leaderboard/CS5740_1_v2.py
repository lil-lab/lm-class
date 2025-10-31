import base64
import dataclasses
import logging
import os
import pprint
from io import StringIO
from typing import Any, Dict, List

import pandas as pd
from github import Github, Organization, Repository, UnknownObjectException
from sklearn.metrics import accuracy_score
import argparse
from tqdm import tqdm
import concurrent.futures
import os

LEADERBOARD_REPO_NAME = "leaderboards"
CLASS = "cornell-cs5740-sp25"

STAFF = {
    "yoavartzi",
    "GSYfate",
    "chenzizhao",
    "evan-wang-13",
    "shankarp8",
    "yilun-hua",
}


@dataclasses.dataclass
class Submission:
    name: str
    members: List[str]
    repo: Repository
    result_files: Dict[str, Any] | None = None
    result_dfs: Dict[str, pd.DataFrame] | None = None

    def __str__(self):
        return pprint.pformat(self)

    def pull_results(self, accepted_file_names: List[str] = None):
        """step 1. pull results/ directory (IO bound)"""
        try:
            res_files = self.repo.get_contents("results")
        except UnknownObjectException:
            res_files = {}
            logging.error(f"results/ not found for {self.repo}. ")
        accepted_file_names = set(accepted_file_names or [])
        self.result_files = {
            result_file.name: result_file
            for result_file in res_files
            if result_file.name in accepted_file_names
        }
        return self

    def pull_result_contents(self):
        """step 2. pull dataframes in results/ directory (IO bound)"""
        if not self.result_files:
            return self

        def _load(path):
            content_encoded = self.repo.get_git_blob(path.sha).content
            content = base64.b64decode(content_encoded).decode("utf-8")
            try:
                df = pd.read_csv(StringIO(content))
            except ValueError as e:
                df = None
                logging.error(
                    f"failed to load {path} from {self.repo} as a CSV with error: {e}"
                )
            return df

        self.result_dfs = {
            file_name: _load(path)
            for file_name, path in self.result_files.items()
        }
        return self


def pull_submissions(org: Organization, prefix: str) -> List[Submission]:
    """step 0. pull submissions (IO bound)"""
    # TODO: threadpool
    subs = [
        Submission(
            name=repo.name,
            members=sorted(
                [
                    c.login
                    for c in repo.get_collaborators()
                    if c.login not in STAFF
                ]
            ),
            repo=repo,
        )
        for repo in org.get_repos()
        if repo.name.startswith(prefix)
    ]
    subs = [
        sub for sub in subs if not any(staff in sub.name for staff in STAFF)
    ]
    return subs


def pull_per_sub(sub: Submission):
    sub.pull_results(accepted_file_names=accepted_file_names)
    sub.pull_result_contents()


@dataclasses.dataclass
class LeaderboardEntry:
    leaderboard: str
    method: str
    dataset: str
    score: float
    member: str


def compute_entries(
    sub: Submission,
    test_data: Dict[str, pd.DataFrame],
    leaderboard_assignment_name: str,
) -> List[LeaderboardEntry]:
    """step 3. compute scores from dataframes (CPU bound)"""
    if not sub.result_dfs:
        return []

    def _f(file_name: str, df: pd.DataFrame) -> LeaderboardEntry:
        method, dataset, *_ = file_name.split("_")
        try:
            score = accuracy_score(test_data[dataset]["label"], df["label"])
        except ValueError as e:
            score = -1.0
            logging.error(
                f"failed to compute score for {file_name} from {sub.repo} with error: {e}"
            )
        return LeaderboardEntry(
            leaderboard=f"{leaderboard_assignment_name}/{dataset}.csv",
            method=method,
            dataset=dataset,
            score=round(score, 4),
            member=", ".join(sub.members),
        )

    return [_f(file_name, df) for file_name, df in sub.result_dfs.items()]


def aggregate_leaderboards(
    all_entries: List[LeaderboardEntry],
) -> Dict[str, pd.DataFrame]:
    """aims to be assignment-agonistic"""

    all_entries = [dataclasses.asdict(e) for e in all_entries]
    leaderboards = pd.DataFrame(all_entries)
    ret = {}
    for path, board in leaderboards.groupby("leaderboard"):
        board.sort_values("score", ascending=False, inplace=True)
        board = board[["score", "method", "member"]]
        board.reset_index(drop=True, inplace=True)
        ret[path] = board
    return ret


def update_leaderboard(
    path: str, board: pd.DataFrame, leaderboard_repo: Repository, deploy: bool
):
    csv_content = board.to_csv(index=False)

    full_path = os.path.join("public", path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w+") as f:
        f.write(csv_content)
    logging.info(f"Updated {full_path}")

    if deploy:
        leaderboard_file = leaderboard_repo.get_contents(path)
        commit_message = "Leaderboard Update"
        leaderboard_repo.update_file(
            leaderboard_file.path,
            commit_message,
            csv_content,
            leaderboard_file.sha,
        )
        logging.info(f"Updated {path} in {leaderboard_repo}")


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--gh_user", type=str)
    parser.add_argument("--gh_token", type=str)
    parser.add_argument("--deploy", action="store_true")
    args = parser.parse_args()

    GH_USER = os.environ.get("GH_USER") or args.gh_user
    GH_TOKEN = os.environ.get("GH_TOKEN") or args.gh_token
    DEPLOY = args.deploy
    REPO_ASSIGNMENT_PREFIX = "a1-"  # prefix of student repos
    LEADERBOARD_ASSIGNMENT_NAME = "a1"  # folder name in leaderboards repo
    logging.info(f"GH_USER: {GH_USER}, DEPLOY: {DEPLOY}")

    accepted_file_names = [
        "mlp_newsgroups_test_predictions.csv",
        "mlp_sst2_test_predictions.csv",
        "perceptron_newsgroups_test_predictions.csv",
        "perceptron_sst2_test_predictions.csv",
    ]
    test_data = {
        "newsgroups": pd.read_csv("../a1/test_data/newsgroups_test_labels.csv"),
        "sst2": pd.read_csv("../a1/test_data/sst2_test_labels.csv"),
    }

    gh = Github(GH_USER, GH_TOKEN)
    org = gh.get_organization(CLASS)
    subs = pull_submissions(org, REPO_ASSIGNMENT_PREFIX)
    logging.info(
        f"Found {len(subs)} submissions with prefix {REPO_ASSIGNMENT_PREFIX}"
    )
    if len(subs) == 0:
        logging.info("Exiting. No student submissions found.")
        exit()

    logging.info("Pulling csvs ...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        executor.map(pull_per_sub, subs)

    logging.info("Computing entries...")
    all_entries = []
    for sub in tqdm(subs, desc="computing entries"):
        all_entries += compute_entries(
            sub, test_data, LEADERBOARD_ASSIGNMENT_NAME
        )

    logging.info("Aggregating leaderboards...")
    all_leaderboards = aggregate_leaderboards(all_entries)

    leaderboard_repo = org.get_repo(LEADERBOARD_REPO_NAME) if DEPLOY else None
    for path, board in all_leaderboards.items():
        update_leaderboard(path, board, leaderboard_repo, DEPLOY)
    logging.info("Done.")
