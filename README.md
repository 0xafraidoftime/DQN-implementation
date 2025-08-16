# Spaceship Survival Game

A Deep Q-Network (DQN) implementation for training an AI agent to navigate a spaceship through an asteroid field and maximize survival time.

## Overview

This project implements a reinforcement learning solution for a spaceship survival game where an AI agent learns to navigate through a dynamic asteroid field. The agent uses Deep Q-Network (DQN) with experience replay and target networks to learn optimal navigation strategies.

### Game Mechanics
- **Environment**: 2D grid-based asteroid field (default: 10x10)
- **Objective**: Survive as long as possible by avoiding asteroid collisions
- **Actions**: Move up, down, left, or right
- **Obstacles**: Randomly positioned asteroids throughout the grid
- **Scoring**: Survival time-based rewards with collision penalties

## Problem Statement

Design a deep neural network that takes the current game state (spaceship and asteroid positions) as input and outputs the optimal movement action to maximize survival time in a dynamic asteroid field environment.

## Architecture

### Deep Q-Network (DQN) Components

1. **Environment (`SpaceShipEnv`)**
   - Custom OpenAI Gym environment
   - 2D grid representation with spaceship and asteroids
   - Collision detection and reward system

2. **Neural Network Architecture**
   - 3 Convolutional layers with ReLU activation
   - Flatten layer followed by 2 fully connected layers
   - Output layer with Q-values for each action

3. **Experience Replay Buffer**
   - Stores agent experiences for stable training
   - Enables learning from past experiences

4. **Training Components**
   - Epsilon-greedy exploration strategy
   - Target network for stability
   - Q-learning updates with experience replay

## Installation

### Prerequisites
```bash
pip install gym
pip install tensorflow
pip install numpy
```

### Dependencies
- **OpenAI Gym**: Environment framework
- **TensorFlow**: Deep learning framework
- **NumPy**: Numerical computations
- **Collections**: Replay buffer implementation

## Usage

### Quick Start

1. **Import and Initialize Environment**
```python
from spaceship_env import SpaceShipEnv
env = SpaceShipEnv(grid_size=(10, 10), num_asteroids=10)
```

2. **Create and Train Agent**
```python
from dqn_agent import DQNAgent

state_size = env.observation_space.shape
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)

# Train the agent
agent.train(env, num_episodes=1000, batch_size=32)
```

3. **Test Trained Agent**
```python
# Evaluate performance
test_episodes = 10
total_rewards = []
for _ in range(test_episodes):
    state = env.reset()
    total_reward = 0
    while True:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
        if done:
            break
    total_rewards.append(total_reward)

average_reward = sum(total_rewards) / len(total_rewards)
print(f"Average reward: {average_reward}")
```

## State Representation

The game state is represented as a 2D numpy array where:
- `0`: Empty space
- `1`: Spaceship position
- `2`: Asteroid positions

## Action Space

The agent can perform 4 discrete actions:
- `0`: Move Up (↑)
- `1`: Move Down (↓)
- `2`: Move Left (←)
- `3`: Move Right (→)

## Reward System

- **Step Penalty**: -1 for each move (encourages efficiency)
- **Collision Penalty**: -10 for hitting an asteroid (terminates episode)
- **Survival Reward**: Implicit through step count maximization

## Hyperparameters

### Default Training Parameters
- **Episodes**: 1000
- **Batch Size**: 32
- **Grid Size**: 10x10
- **Number of Asteroids**: 10
- **Epsilon Decay**: 0.995
- **Minimum Epsilon**: 0.01

### Network Architecture
- **Conv Layer 1**: 32 filters, 8x8 kernel, stride 4
- **Conv Layer 2**: 64 filters, 4x4 kernel, stride 2
- **Conv Layer 3**: 64 filters, 3x3 kernel, stride 1
- **Dense Layer 1**: 512 neurons
- **Output Layer**: 4 neurons (one per action)

## Performance Metrics

The implementation tracks:
- **Total Reward per Episode**: Cumulative reward obtained
- **Survival Time**: Number of steps before collision
- **Average Performance**: Mean reward over test episodes

### Expected Results
- Training shows variable performance due to random asteroid placement
- Average test reward of approximately -30.5 after 1000 episodes
- Performance improves as epsilon decays and exploration decreases

## Training Process

1. **Exploration Phase**: High epsilon for random action selection
2. **Experience Collection**: Store state-action-reward transitions
3. **Network Updates**: Learn from replay buffer experiences
4. **Target Network Updates**: Periodic weight synchronization
5. **Exploitation Phase**: Gradually reduce exploration

## Key Features

- **Dynamic Environment**: Asteroids positioned randomly each episode
- **Stable Learning**: Target network prevents training instability
- **Experience Replay**: Breaks correlation between consecutive experiences
- **Epsilon-Greedy**: Balances exploration and exploitation
- **Collision Detection**: Realistic game physics implementation

## Code Structure

```
├── SpaceShipEnv          # Game environment implementation
├── DQN                   # Neural network architecture
├── DQNAgent             # Main agent with training logic
├── ReplayBuffer         # Experience storage and sampling
└── Training Loop        # Episode management and evaluation
```

## Potential Improvements

1. **Dynamic Asteroids**: Implement moving asteroids for increased difficulty
2. **Reward Shaping**: Add distance-based rewards for better guidance
3. **Network Architecture**: Experiment with different CNN architectures
4. **Hyperparameter Tuning**: Optimize learning rate, batch size, network size
5. **Advanced Algorithms**: Implement Double DQN, Dueling DQN, or Rainbow DQN
6. **Visual Interface**: Add game visualization for better monitoring

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- OpenAI Gym for the environment framework
- TensorFlow team for the deep learning framework
- Deep Q-Network research by DeepMind

## Support

For questions, issues, or contributions, please open an issue in the repository or contact the maintainers.

---

**Happy Training! 🎮🤖**
