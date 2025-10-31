## Scripts for Leaderboard Updates

- `how_to_automate_leaderboard_updates.md` -- instructions on setting up a cronjob to update the leaderboard automatically. 
- Scripts with v2 desigration are much faster with threadpool for io (downloading csv files from gh), plus opinionated refactors.

### Note

Not recommended: Github Actions (CD template included in .github/workflows/python-app.yaml) are a natural fit to udpate the leaderboard. However, **you are like to run out of free hours towards the end of the semester, so it is not recommended.**
