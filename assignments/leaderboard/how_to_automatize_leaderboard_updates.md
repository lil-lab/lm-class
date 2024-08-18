## How to set a cronjob to update the leaderboard automatically

Note:
- We assume that you can already execute the update manually locally (e.g. by directly running the Python scripts).

### Tips

- To avoid developing using the public leaderboard repo, it's possible to create a private test repo for the leaderboard, and ensure that the updates are correct.

- Students can make unexpected errors, the `CS5740_{i}.py` files can be modified to catch those errors.

### Commands for automatizing the leaderboard script via cron

On the machine of your choice (e.g. a free local server, a free AWS instance):

1. Create a bash script that runs the Python script (directly executing the Python script with cron might lead to some issues with pandas).

File used for 2024: `assignment_cron.sh`

2. Change the permission of access to the files, e.g.:

```
chmod +x assignment_cron.sh
chmod +x CS5740_1.py
```

3. Open the crontab file (listing the jobs that cron will run) with `crontab -e`

4. Add the job, example:

```
0 9 * * * /path/to/assignment_cron.sh  >> /path/to/log_file.log 2>&1
```

This asks cron to execute the job everyday at 9:00am.

You should see "crontab: installing new crontab" after saving the crontab file.

5. Highly recommended: create your submission from the github classroom link and test the leaderboard script.


### Related

More information about cron: [link](https://en.wikipedia.org/wiki/Cron#Multi-user_capability)
