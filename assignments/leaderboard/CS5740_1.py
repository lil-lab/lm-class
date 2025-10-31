################################################################################
# Imports.
################################################################################

import base64
from io import StringIO
import pandas as pd
from tqdm import tqdm
from github import Github
from sys import argv
import argparse
from IPython import embed

################################################################################
# TODO: Configuration.

# [x] create your github access token
# [x] create public leaderbords files under the github organization
# [x] this script is run in this folder, otherwise update relative import first

################################################################################

# argparsing for github token
parser = argparse.ArgumentParser()
parser.add_argument('--username', help='GitHub username')
parser.add_argument('--token', help='Access token for that GitHub username')
args = parser.parse_args()

# Write leaderboards to disk in current directory.

DRY_RUN = (argv[1] == "True")

# Username / password or access token
# See: https://github.com/PyGithub/PyGithub#simple-demo

GITHUB_TOKEN = [args.username, args.token]

# Organization name for the class.

CLASS = "cornell-cs5740-sp24"

# Exclude these people from collaborator list.

STAFF = {
    "yoavartzi",
    "momergul",
    "annshin",
    "Vrownie", 
    "sy464",
    "YiChen8185",
    "kanlanc",
}

# Name of the leaderboard repo.

LEADERBOARD_REPO_NAME = "leaderboards"

# Assignment directory in leaderboard repo.

LEADERBOARD_ASSIGMENT_NAME = "a1"

# GitHub Classroom prefix attached to every repo, used to find assigments.

REPO_ASSIGNMENT_PREFIX = "cs5740-sp24-assignment-1-"

################################################################################
# TODO: Compute and sort scores.
################################################################################

from sklearn.metrics import accuracy_score

try:
    # NOTE the import is relative, assuming running this script in this folder
    test_data = {
        "newsgroups":  pd.read_csv("../a1/test_data/newsgroups_test_labels.csv"),
        "sst2": pd.read_csv("../a1/test_data/sst2_test_labels.csv"),
    }
except Exception:
    print('Test data label cannot be imported')
    embed()

def compute_scores(file_name, pred, repo):

    try:

        method, dataset, *_ = file_name.split("_")
        true = test_data[dataset]

        comment = ""

    except:

        return

    try:

        if dataset == "newsgroups":
            accuracy = accuracy_score(true["label"], pred["label"]).round(5)
        elif dataset == "sst2":
            accuracy = accuracy_score(true["label"], pred["label"]).round(5)

    except:

        accuracy = None
        comment = "Error computing accuracy!"

    return {

        # Required: name of leaderboard file.
        "leaderboard": "leaderboard_" + dataset,

        "Score":       accuracy,
        "Method":      method,
        "Member":     " ".join(repo["member"]),
        "Comment":     comment,

    }

def sort_scores(leaderboards):

    return (
        leaderboards
        .sort_values([
            "Score",
            "Member",
            "Method"
        ], ascending = False)
    )

################################################################################
# API authentication, find organization and leaderboard repo.
################################################################################

git = Github(*GITHUB_TOKEN)
org = git.get_organization(CLASS)
leaderboard_repo = org.get_repo(LEADERBOARD_REPO_NAME)

# ################################################################################
# # Get all assignment repos, team names, members, etc.
# ################################################################################

print("Loading Repos...")

repos = [

    {

        "git":  repo,
        "name": repo.name,
        "member": sorted([
            c.login for c in repo.get_collaborators()
            if c.login not in STAFF
        ]),

    }

    for repo in org.get_repos()
    if repo.name.startswith(REPO_ASSIGNMENT_PREFIX)

]

# Remove all staff member teams.
# Remove repos for which any of the staff id is in the name
repos = [repo for repo in repos if not any(staff in repo["name"] for staff in STAFF)]

# ################################################################################
# # Extract repo files.
# ################################################################################

for repo in tqdm(repos, desc = "Finding files"):
    print(repo)
    # This check is to avoid students who removed the 'results' folder, raising a 404 error
    try:
        res_files = repo["git"].get_contents("results")
    except:
        print(f"Issue: results folder not found for {repo}")
        continue

    repo["files"] = {
        
        result_file.name: result_file
        for result_file in repo["git"].get_contents("results")
        if result_file.name in [
            "mlp_newsgroups_test_predictions.csv",
            "mlp_sst2_test_predictions.csv",
            "perceptron_newsgroups_test_predictions.csv",
            "perceptron_sst2_test_predictions.csv",
        ]
    }

# ################################################################################
# # Download files and load CSVs.
# ################################################################################

for repo in tqdm(repos, desc = "Downloading files"):
    repo["results"] = {}

    # This check is to avoid students who removed the 'results' folder, raising a 404 error
    if "files" not in repo.keys():
        continue

    for file_name, path in repo["files"].items():

        content_encoded = repo["git"].get_git_blob(path.sha).content
        content = base64.b64decode(content_encoded).decode("utf-8")

        data = pd.read_csv(StringIO(content))
        repo["results"][file_name] = data

# ################################################################################
# # Compute scores and create assignment-level master leaderboard.
# ################################################################################

leaderboards = []

for repo in repos:
    for result_name, result in repo["results"].items():

        score = compute_scores(result_name, result, repo)

        if score is not None:
            leaderboards.append(score)

leaderboards = sort_scores(pd.DataFrame(leaderboards))

# ################################################################################
# # Split master leaderboard into sub-boards and commit them.
# ################################################################################

for name, board in leaderboards.groupby("leaderboard"):

    del board["leaderboard"]

    csv_content = board.to_csv(index = False)
    csv_name    = name + ".csv"

    commit_message = "Leaderboard Update"

    if DRY_RUN:

        with open("public/" + csv_name, "w") as f:
            f.write(csv_content)

    else:

        print(LEADERBOARD_ASSIGMENT_NAME)
        print(csv_name)
        leaderboard_file = leaderboard_repo.get_contents(
            LEADERBOARD_ASSIGMENT_NAME + "/" + csv_name)

        print("Updating", leaderboard_file.path)

        leaderboard_repo.update_file(
            leaderboard_file.path,
            commit_message,
            csv_content,
            leaderboard_file.sha)

print("Done!")

# ################################################################################
# ################################################################################
# ################################################################################
# ################################################################################
# ################################################################################
