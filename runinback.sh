#!/usr/bin/env bash

source activate master8_env
export PYTHONPATH="/home/mluser/master8_projects/clustering_vacancies/"
cd /home/mluser/master8_projects/clustering_vacancies/
chmod +x $1
nohup python $1 $2 &

ps -A | grep python


#usefull commands:
#Show processes
#kill -9 8979798
#htop
#du -sh -- *
#df -h