#!/usr/bin/env bash


#
cd /home/ph/LudwigCluster/scripts
bash kill_job.sh TreeTransitions
#bash reload_watcher.sh

echo "Submitting to Ludwig..."
cd /home/ph/TreeTransitions
source venv/bin/activate
python submit.py -r5 -s
deactivate
echo "Submission completed"

sleep 5
tail -n 6 /media/lab/stdout/*.out