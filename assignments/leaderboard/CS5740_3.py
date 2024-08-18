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
import numpy as np

################################################################################
# TODO: Configuration.
################################################################################

# argparsing for github token
parser = argparse.ArgumentParser()
parser.add_argument('--username', help='GitHub username')
parser.add_argument('--token', help='Access token for that GitHub username')
args = parser.parse_args()

# Write leaderboards to disk in current directory.

DRY_RUN = (argv[1] == "True")
#DRY_RUN = False

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

LEADERBOARD_ASSIGMENT_NAME = "a3"

# GitHub Classroom prefix attached to every repo, used to find assigments.

REPO_ASSIGNMENT_PREFIX = "cs5740-sp24-assignment-3-"

################################################################################
# Compute and sort scores.
################################################################################

from scipy.stats import spearmanr

def get_similarity_scores(E1, E2):
    """
    From the assignment repo (to use the same computation)
    Function to compute the similarity scores between two sets of embeddings.
    """
    similarity_scores = []
    for idx in range(len(E1)):
        sim_score = round(np.dot(E1[idx][1], E2[idx][1]), 6)
        similarity_scores.append(sim_score)
    return similarity_scores

def compute_spearman_correlation(similarity_scores, human_scores):
    """
    From the assignment repo (to use the same computation)
    Function to compute the Spearman correlation between the similarity scores and human scores (labels).
    """
    return round(spearmanr(similarity_scores, human_scores).correlation, 6)

def read_embedding(rows):
    embeddings = []
    for i, row in enumerate(rows):
        word, *vector = row.split()
        embeddings.append((word, [float(x) for x in vector]))
        dim = len(vector)
    return embeddings, dim

isol_test = pd.read_csv('../a3/test_data/isolated_test_y.csv', index_col='id')
cont_test = pd.read_csv('../a3/test_data/contextual_test_y.csv', index_col='id')
isol_test.columns = ["actual"]
cont_test.columns = ["actual"]

try:
    test_data = {"isol": isol_test, "cont": cont_test}
except Exception:
    print('Test data label cannot be imported')

def enforce_embedding_size(embedding_list, max_allowed_embed_size=1024):
    """Check if all the embeddings have at most max_allowed_embed_size"""
    for embed in embedding_list:
        if len(embed[1]) > max_allowed_embed_size:
            return False
    return True

def compute_scores(result_files, repo):
    pred_embeds = {"cont": {}, "isol": {}}
    comments = {"cont": "", "isol": ""}
    scores = {"cont": None, "isol": None}
    dataset = ""
    for f, v in result_files.items():
        try:
            method, dataset, _, word_order, *_ = f.split("_")
            if dataset not in ['cont', 'isol']: return
            pred_embeds[dataset][word_order] = v
        except:
            print(f)
            if dataset != "":
                comments[dataset] = "Error reading result embeddings!"
            else:
                comments["cont"] = "Error reading result embeddings!"
                comments["isol"] = "Error reading result embeddings!"
    for _task in ["cont", "isol"]:
        if pred_embeds[_task] == {}:
            scores[_task] = None
            comments[_task] = "Error reading result embeddings!"
    if not pred_embeds["cont"] == {} and not pred_embeds['isol'] == {}:
        for task in ['cont', 'isol']:
            # Check: embedding size
            try:
                if not enforce_embedding_size(pred_embeds[task]["words1"]) or not enforce_embedding_size(pred_embeds[task]["words2"]):
                    comments[task] = "Embedding size exceeds 1024!"
                    continue
                else:
                    try:
                        similarity = get_similarity_scores(pred_embeds[task]["words1"], pred_embeds[task]["words2"])
                        data = test_data[task].copy()
                        data.columns = ["actual"]
                        data["predicted"] = similarity
                        score = round(spearmanr(data).correlation, 6)
                        if pd.isnull(score):
                            score = None
                            comments[task] = "Error computing correlation: the score is nan"
                        else:
                            score = score.round(5)
                        scores[task] = score
                    except:
                        comments[task] = "Error computing correlation!"
            except:
                continue
    # This is a general error catch, to avoid edge cases (e.g. where people used lfs => will get a None score)
    for _task in ["cont", "isol"]:
        if scores[_task] == None:
            comments[_task] = "Error computing correlation!"
    print("scores and comments", scores, comments)
    return {
        # Required: name of leaderboard file.
        "leaderboard": "leaderboard_isol",
        "Score":       scores["isol"],
        "Method":     method,
        "Member":     " ".join(repo["member"]),
        "Comment":     comments["isol"],
    }, {
        "leaderboard": "leaderboard_cont",
        "Score":       scores["cont"],
        "Method":     method,
        "Member":     " ".join(repo["member"]),
        "Comment":     comments["cont"]
    }

def sort_scores(leaderboards):

    if len(leaderboards) == 0:
        return leaderboards

    return (
        leaderboards
        .sort_values([
            "Score",
            "Member",
        ], ascending = False)
    )

################################################################################
# API authentication, find organization and leaderboard repo.
################################################################################

git = Github(*GITHUB_TOKEN)
org = git.get_organization(CLASS)
leaderboard_repo = org.get_repo(LEADERBOARD_REPO_NAME)

################################################################################
# Get all assignment repos, team names, members, etc.
################################################################################

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
repos = [repo for repo in repos if not any(staff in repo["name"] for staff in STAFF)]

################################################################################
# Extract repo files.
################################################################################

possible_files = [
    "bert_cont_test_words1_embeddings.txt",
    "bert_cont_test_words2_embeddings.txt",
    "bert_isol_test_words1_embeddings.txt",
    "bert_isol_test_words2_embeddings.txt",
    "gpt2_cont_test_words1_embeddings.txt",
    "gpt2_cont_test_words2_embeddings.txt",
    "gpt2_isol_test_words1_embeddings.txt",
    "gpt2_isol_test_words2_embeddings.txt",
    "word2vec_cont_test_words1_embeddings.txt",
    "word2vec_cont_test_words2_embeddings.txt",
    "word2vec_isol_test_words1_embeddings.txt",
    "word2vec_isol_test_words2_embeddings.txt",
]

for repo in tqdm(repos, desc = "Finding files"):

    repo["files"] = {
        
        result_file.name: result_file
        for result_file in repo["git"].get_contents("results")
        if result_file.name.endswith(".txt") and result_file.name in possible_files

    }
################################################################################
# Download files and load CSVs.
################################################################################

for repo in tqdm(repos, desc = "Downloading files"):
    print(repo['name'])

    repo["results"] = {
        "word2vec": {},
        "bert": {},
        "gpt2": {},
    }

    for file_name, path in repo["files"].items():
        content_encoded = repo["git"].get_git_blob(path.sha).content
        content = base64.b64decode(content_encoded).decode("utf-8")
        try:
            content_list = [x for x in content.split('\n') if x != '']
            data, dim = read_embedding(content_list)
        except:
            print("Except", file_name)
            data = None
        if "word2vec" in file_name:
            repo["results"]["word2vec"][file_name] = data
        elif "bert" in file_name:
            repo["results"]["bert"][file_name] = data
        elif "gpt2" in file_name:
            repo["results"]["gpt2"][file_name] = data

################################################################################
# Compute scores and create assignment-level master leaderboard.
################################################################################
leaderboards = []

for repo in repos:
    print(repo["member"])
    for model, file_names in repo["results"].items():
        print(model)
        if model not in ["word2vec", "bert", "gpt2"]: continue
        result_names = file_names.keys()
        results = file_names.values()
        try:
            score_isol, score_cont = compute_scores(file_names, repo)
        except:
            print("model", model)
            continue

        if score_isol is not None:
            leaderboards.append(score_isol)
        if score_cont is not None:
            leaderboards.append(score_cont)
leaderboards = sort_scores(pd.DataFrame(leaderboards))

################################################################################
# Split master leaderboard into sub-boards and commit them.
################################################################################
print("Checking the leaderboard...")
if len(leaderboards) != 0:

    for name, board in leaderboards.groupby("leaderboard"):

        del board["leaderboard"]

        csv_content = board.to_csv(index = False)
        csv_name    = name + ".csv"

        commit_message = "Leaderboard Update"

        if DRY_RUN:

            with open("public/" + csv_name, "w") as f:
                f.write(csv_content)

        else:

            leaderboard_file = leaderboard_repo.get_contents(
                LEADERBOARD_ASSIGMENT_NAME + "/" + csv_name)

            print("Updating", leaderboard_file.path)

            leaderboard_repo.update_file(
                leaderboard_file.path,
                commit_message,
                csv_content,
                leaderboard_file.sha)

print("Done!")

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
