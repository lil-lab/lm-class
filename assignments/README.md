# Setting up assignments & Checklist

- The instructions & assignments .pdf files, the starter code and an example of solution are in the `a{1, 2, 3, 4}` subfolders

## Code

### How to create the assignment in Github Classroom, the assignment template repo and the autograding pipeline?

You can refer to the file on [Setting up autograding in GitHub Classroom](./scripts/github_autograding.md).

### How to set the leaderbaord?

You can refer to the file on [Setting a cronjob for the leaderboard](https://github.com/lil-lab/cs5740-assignments/blob/master/leaderboard/how_to_automatize_leaderboard_updates.md).

## Grading

- Create the assignment on Gradescope and Canvas (Yoav will take care of it for now)
    - Follow the Canvas link to Gradescope to create the assignment

## Assignment deployment checklist

- [ ] Instruction .pdf ready & reviewed
  - [ ] Make sure a milestone is included
- [ ] Report template ready & reviewed
- [ ] Code
  - [ ] Go through the starter code template and solutions & update as needed
  - [ ] Ensure the code is minimal, typing is included and that the docstring contains the information needed & is clear
  - [ ] Run through every experiment
  - [ ] Autograding: add/modify tests
- [ ] Assignment on Github Classroom
  - [ ] Create the assignment according to the instructions
  - [ ] Test the assignment submission pipeline (with a dummy submission from the TAs' accounts)
- [ ] Leaderboard
  - [ ] Create the pipeline in a private leaderboard repo according to the instructions
  - [ ] Test the pipeline in the private leaderboard repo with dummy submissions
  - [ ] The leaderboard script may need to be updated if there are new error cases from the students
  - [ ] The last refresh of the leaderboard may be a few hours earlier (e.g. ~2h) than the actual deadline, so that students have time to get the test result and update their report
  - [ ] After the deadline: make sure to end the automatic cronjob
- [ ] Rubric
  - [ ] Review & update the rubric
  - [ ] Add the rubric to Gradescope

## Grading

- [ ] Sync with the graders to set a grading session time (better to grade during the session if possible). If not everything can be graded during the session, fix a milestone to ensure progress
- [ ] For the grading session: send the instruction, report template and grading rubric to the graders
- [ ] Go through the grading rubric to make sure the graders are on the same page
- [ ] Go through a few assignments with the graders to align the grading criteria
