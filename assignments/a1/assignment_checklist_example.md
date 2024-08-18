# How to check an assignment before it's sent to the students

Updated: 2024, with assignment 1

## Assignment 1

Code part.

### Assignment creation

- Needs to use a template repo from the same organization (not a public repo)

### Running through the assignment

- [ ] accept (via the invitation link) and clone the assignment from github classroom
- [ ] setup the environment
  - [ ] create a virtual env (e.g. conda) with the python package requirement (3.10.x for now)
  - [ ] install from requirements with pip's `--no-cache-dir` flat
- [ ] write/copy solutions to the repo

#### Offline

- [ ] run all pytests (cf. README) and make sure they pass locally
- [ ] check that the models train correctly
- [ ] check that the evaluation (dev/test) are correct

#### Online

- [ ] commit and push the completed assignment, make sure that they pass all the autograding tests (cf. Actions for details)
- [ ] check we get all the points, fix any issues

#### Leaderboard

- [ ] install Github CLI, authentificate, install the Classroom extension and clone the student repos (ex: `gh classroom clone student-repos -a xxxxxx`)
- [ ] install PyGithub (`pip install PyGithub`)
- [ ] create a `leaderboards` repo with the corresponding assignment subfolder + placeholder blank csvs
- [ ] running through the corresponding `leaderboard/*.py` and make sure that they update correctly the csvs

### (Unofficial, just self-reminder) Writeup

- [ ] review the assignment document, check for any errors, typo, inconsistencies, unclear/ambiguous formulation
- [ ] check the assignment submission template
