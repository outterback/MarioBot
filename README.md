## Welcome to the MarioBot dev site

This site is a documentation of my learning and experimentation with [SerpentAI](http://github.com/SerpentAI/SerpentAI/). My intention with this project is to learn more about machine learning via practical experiments on some of my favorite games, and at the same time document these experiments as a source of inspiration and knowledge for others interested in getting going with their own machine learning projects. It will not be a focused tutorial, but rather a collection of approaches to tackle different sub-problems encountered when applying machine learning models to computer games.


## Project details
The game I am working on in this project is the european version of Super Mario Bros 1 for the NES. The model being trained is a small conv-net, and the code around it is based on the DQN reinforcement learning tutorial from the PyTorch 0.3.1 documentation.


Implemented features:
- Localization of mario sprite
- Reading of numbers on the screen
- Semi-decent identification of game over
- Game loop that resets game to world 1-1
- A basic reward system
- Deep Q-Network with annealed epsilon-greedy policy 
- Replay memory with save features


#### Current WIP
- Adding documentation to digit classification notebook

#### Master branch
Will contain rendered jupyter notebooks, shell scripts for running examples. My hope is that there will be fun clips of gameplay as well, but that's not really up to me to decide, is it?

#### Dev branch
Likely messy and undocumented. 

The currently interesting things are:
- [Digit classification notebook](https://github.com/outterback/MarioBot/blob/dev/lab/ml/digit_classification/digits.ipynb)
- [Game Agent](https://github.com/outterback/MarioBot/blob/dev/plugins/SerpentMarioBros1GameAgentPlugin/files/serpent_MarioBros1_game_agent.py)
- [The DQN implementation](https://github.com/outterback/MarioBot/blob/dev/plugins/SerpentMarioBros1GameAgentPlugin/files/ml_models/cnn/dqn.py)
- [The Mario Bros API](https://github.com/outterback/MarioBot/blob/dev/plugins/SerpentMarioBros1GamePlugin/files/api/api.py)


## TODO:
- Improve logging
- Try out npyscreen to drive a TUI
- Move away from DQN tutorial model


## Requirements

- Python 3.6.4
- SerpentAI
- OpenCV 3.4.0
- Pytorch 0.3.1

