# Gomoku_DQN
Construct Gomoku AI using DQN
Support flexible grid size and piece number to win

## Architecture
dqn.py: the definition of DQN architecture, Q(s, a), works as agent
gomoku.py: contains the game rule of gomoku, works as the environment
train.py: train model
game.py: GUI program to PVE or PVP

## Usage
python train.py: train model 
python game.py: play gomoku
