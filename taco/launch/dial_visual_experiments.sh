#!/bin/sh

BASE_NAME=$1

tmux start-server
tmux new-session -d -s $BASE_NAME -n htop
tmux new-window -t $BASE_NAME:1 -n bc
tmux new-window -t $BASE_NAME:2 -n taco
tmux new-window -t $BASE_NAME:3 -n ctc
tmux new-window -t $BASE_NAME:4 -n ctcbi

tmux send-keys -t $BASE_NAME:1 "python3 experiment_launch.py " $BASE_NAME " dial_visual bc visual_dataset_merged.p -n 400 -a mlp" C-m
tmux send-keys -t $BASE_NAME:2 "python3 experiment_launch.py " $BASE_NAME " dial_visual taco visual_dataset_merged.p -n 400 -a mlp" C-m
tmux send-keys -t $BASE_NAME:3 "python3 experiment_launch.py " $BASE_NAME " dial_visual ctc visual_dataset_merged.p -n 400 -a mlp" C-
tmux send-keys -t $BASE_NAME:4 "python3 experiment_launch.py " $BASE_NAME " dial_visual ctc visual_dataset_merged.p -n 400 -a bi-recurrent" C-m

tmux attach-session -t $BASE_NAME


