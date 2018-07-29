#!/bin/sh

BASE_NAME=$1

tmux start-server
tmux new-session -d -s $BASE_NAME -n htop
tmux new-window -t $BASE_NAME:1 -n bc100
tmux new-window -t $BASE_NAME:2 -n taco100
tmux new-window -t $BASE_NAME:3 -n ctc100
tmux new-window -t $BASE_NAME:4 -n ctcbi100

tmux new-window -t $BASE_NAME:5 -n bc300
tmux new-window -t $BASE_NAME:6 -n taco300
tmux new-window -t $BASE_NAME:7 -n ctc300
tmux new-window -t $BASE_NAME:8 -n ctcbi300


tmux new-window -t $BASE_NAME:9 -n bc500
tmux new-window -t $BASE_NAME:10 -n taco500
tmux new-window -t $BASE_NAME:11 -n ctc500
tmux new-window -t $BASE_NAME:12 -n ctcbi500


tmux send-keys -t $BASE_NAME:1 "python3 experiment_launch.py " $BASE_NAME " nav bc dataset_04.p -n 100 -a mlp" C-m
tmux send-keys -t $BASE_NAME:2 "python3 experiment_launch.py " $BASE_NAME " nav taco dataset_04.p -n 100 -a mlp" C-m
tmux send-keys -t $BASE_NAME:3 "python3 experiment_launch.py " $BASE_NAME " nav ctc dataset_04.p -n 100 -a mlp" C-m
tmux send-keys -t $BASE_NAME:4 "python3 experiment_launch.py " $BASE_NAME " nav ctc dataset_04.p -n 100 -a bi-recurrent" C-m


tmux send-keys -t $BASE_NAME:5 "python3 experiment_launch.py " $BASE_NAME " nav bc dataset_04.p -n 300 -a mlp" C-m
tmux send-keys -t $BASE_NAME:6 "python3 experiment_launch.py " $BASE_NAME " nav taco dataset_04.p -n 300 -a mlp" C-m
tmux send-keys -t $BASE_NAME:7 "python3 experiment_launch.py " $BASE_NAME " nav ctc dataset_04.p -n 300 -a mlp" C-m
tmux send-keys -t $BASE_NAME:8 "python3 experiment_launch.py " $BASE_NAME " nav ctc dataset_04.p -n 300 -a bi-recurrent" C-m


tmux send-keys -t $BASE_NAME:9 "python3 experiment_launch.py " $BASE_NAME " nav bc dataset_04.p -n 500 -a mlp" C-m
tmux send-keys -t $BASE_NAME:10 "python3 experiment_launch.py " $BASE_NAME " nav taco dataset_04.p -n 500 -a mlp" C-m
tmux send-keys -t $BASE_NAME:11 "python3 experiment_launch.py " $BASE_NAME " nav ctc dataset_04.p -n 500 -a mlp" C-m
tmux send-keys -t $BASE_NAME:12 "python3 experiment_launch.py " $BASE_NAME " nav ctc dataset_04.p -n 500 -a bi-recurrent" C-m


tmux attach-session -t $BASE_NAME


