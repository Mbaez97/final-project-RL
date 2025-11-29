# Multi-Agent Speaker-Listener Environment

## Table of Contents
- [Multi-Agent Speaker-Listener Environment](#multi-agent-speaker-listener-environment)
- [Overview](#overview)
- [Algorithms](#algorithms)
  - [MATD3 (Multi-Agent Twin Delayed DDPG)](#matd3-multi-agent-twin-delayed-ddpg)
  - [MADDPG (Multi-Agent Deep Deterministic Policy Gradient)](#maddpg-multi-agent-deep-deterministic-policy-gradient)
  - [IPPO (Independent Proximal Policy Optimization)](#ippo-independent-proximal-policy-optimization)
- [Implementation](#implementation)
  - [V1 – MATD3 baseline](#v1--matd3-baseline)
  - [V2 – MATD3](#v2--matd3)
  - [V3 – MATD3](#v3--matd3)
  - [V4 – MATD3](#v4--matd3)
  - [V5 – MADDPG](#v5--maddpg)
  - [V6 – IPPO](#v6--ippo)
- [Visualization](#visualization)
- [Results](#results)
- [Summary](#summary)

---

## Overview

This project implements a multi-agent reinforcement learning environment based on the Speaker-Listener environment, where two agents collaborate to solve communication and coordination tasks. The task is to implement a new multi-agent reinforcement learning algorithm. This algorithm should allow the listener to navigate to the goal faster than the MATD3 algorithm (baseline), that is, to achieve an average score higher than -60 (the average score of the current configuration).

The Speaker-Listener environment consists of two agents with distinct roles:

Speaker: Speaks but cannot move.

Listener: Listens to the Speaker's messages and needs to navigate to the target.

---

## Algorithms

### MATD3 (Multi-Agent Twin Delayed DDPG)

MATD3 is a multi-agent extension of the TD3 (Twin Delayed Deep Deterministic Policy Gradient) algorithm, which in turn is an improvement on DDPG. It is an off-policy algorithm that uses a replay buffer for learning from past experiences.

### MADDPG (Multi-Agent Deep Deterministic Policy Gradient)

MADDPG is an extension of the DDPG algorithm to multi-agent environments using centralized training and decentralized execution (CTDE). It is an off-policy algorithm, using a replay buffer to store joint transitions. 

### IPPO (Independent Proximal Policy Optimization)

IPPO is a multi-agent approach where each agent uses the PPO algorithm independently, treating other agents as part of the environment. PPO is an on-policy algorithm that uses clipping of the objective function to maintain stable policy updates. 

---

## Implementation

To compare different multi-agent RL strategies, we ran six versions of the training pipeline: four based on MATD3 (V1–V4) and two based on alternative algorithms (MADDPG and IPPO). Each version corresponds to a separate script and model directory.

### V1 – MATD3 baseline (main.py/models/MATD3)

Baseline implementation using MATD3 with the original network and hyperparameter configuration from the course setup. Agents are trained with population-based training (PBT) via create_population, which evolves a population of MATD3 agents over time. This version serves as the reference point for all subsequent modifications.

### V2 – MATD3 (main_V2.py/models/MATD3_V2)

This version tests whether a smaller network with more aggressive exploration and slightly different update schedule improves coordination.
Uses a more compact network:

Architecture: latent_dim = 64 with two hidden layers of 64 units for both actor and critic.

RL hyperparameters: stronger exploration (EXPL_NOISE = 0.2), higher discount factor (GAMMA = 0.99), smaller replay buffer (MEMORY_SIZE = 100000), and POLICY_FREQ = 2 (policy updated less often than critic).

PBT mutations: moderate probability of mutating architecture, parameters, and RL hyperparameters, encouraging exploration of both weights and hyperparameters.

### V3 – MATD3 (main_V3.py/models/MATD3_V3)

This version focuses on stabilizing learning, using decaying exploration and more conservative evolutionary changes to fine-tune around good solutions.
Keeps the same architecture as V2 (64-dim latent, 64×64 MLPs) but refines the learning dynamics:

Exploration schedule: starts from EXPL_NOISE = 0.1 and linearly decays to 0.02 over 200,000 steps (get_expl_noise), updated inside the training loop for each evolution step.

RL hyperparameters: slightly shorter horizon (GAMMA = 0.95), larger replay buffer (MEMORY_SIZE = 150000), and POLICY_FREQ = 1 (policy updated at every critic update).

PBT mutations: higher probability of no mutation and smaller mutation standard deviation (mutation_sd = 0.03), with reduced architecture and RL-hp changes.

### V4 – MATD3 (main_V4.py/models/MATD4_V4)

V4 therefore tests whether a higher-capacity MATD3 combined with online hyperparameter adaptation can outperform the smaller architectures.
In this version, MATD3 is kept as the underlying algorithm but model capacity and hyperparameter search are extended:

Architecture: larger network with latent_dim = 128 and 128×128 layers for actor and critic.

RL hyperparameters: similar MATD3 settings (EXPL_NOISE ≈ 0.1, GAMMA = 0.99, MEMORY_SIZE = 150000, POLICY_FREQ = 1).

Hyperparameter optimization: introduces a HyperparameterConfig with ranges for lr_actor, lr_critic, batch_size, and learn_step (via RLParameter), enabling PBT to adapt learning rates and update frequency online.

Mutations: same conservative mutation scheme as V3, but now also mutating these RL hyperparameters within specified bounds.

### V5 – MADDPG (main_MADDPG.py/models/MADDPG_V1)

This version evaluates whether using a centralized-critic algorithm (MADDPG) with similar capacity and evolutionary tuning can improve coordination compared to MATD3.
Here the algorithm is switched from MATD3 to MADDPG, while keeping a similar network size to V4:

Architecture: latent_dim = 128 with 128×128 layers.

RL hyperparameters: typical DDPG-style settings (O-U noise, EXPL_NOISE = 0.1, GAMMA = 0.99, MEMORY_SIZE = 150000, POLICY_FREQ = 1).

Exploration schedule: linear decay of EXPL_NOISE from 0.1 to 0.01 over 100,000 steps, updated inside the training loop.

Hyperparameter optimization: HyperparameterConfig with relatively narrow ranges for actor/critic learning rates, batch size and learn_step, adjusted via PBT.

Mutations: same conservative mutation settings as V3–V4.

### V6 – IPPO (main_IPPO.py/models/IPPO_V1)

This version explores how an on-policy, clipping-based method (IPPO) with shared architecture and PBT compares to off-policy algorithms in the same environment.
Finally, V6 replaces value-based deterministic methods with on-policy IPPO:

Architecture: same 128-dim / 128×128 network as V4–V5.

RL hyperparameters (PPO-style): large batch size (BATCH_SIZE = 512), rollout length LEARN_STEP = 2048, GAMMA = 0.99, GAE_LAMBDA = 0.95, clipping parameter (CLIP_EPS = 0.2), entropy coefficient (ENT_COEF = 0.01), value loss coefficient (VF_COEF = 0.5), and gradient clipping (MAX_GRAD_NORM = 0.5).

Hyperparameter optimization: PBT explores lr, batch size, and learn_step through HyperparameterConfig.

Rollout collection: for each PPO update, trajectories are collected for all agents (observations, actions, log-probs, values, rewards, dones) over steps_to_rollout = learn_step / num_envs, and then used for on-policy updates.

Mutations: slightly higher no mutation (0.6) and moderate changes to architecture, parameters, and RL hyperparameters, tuned for PPO.

Whatever version details and specific settings are used are described in versions.txt to guarantee reproducibility of the version.

---

## Visualization

All trained models and their corresponding results are stored in the models directory, where each experiment version generates the learned agent parameters (saved as serialized model weights) together with the full history of training scores. 

To analyze and compare performance across versions, we use the training-plot.ipynb, which automatically loads the score traces from all experiments, and generates an individual figure for each one. Each plot includes two reference lines: one indicating the target score (−60) and another marking the best score achieved by that model. Additionally, the figure reports the mean of the last training steps, providing a quick indicator of the model’s final stability and convergence. Also applies a smoothing window (moving average of 50 steps) in all the experiments to produces a unified visualization of learning progress. The notebook generates the combined figure “all_training_scores.png”, where each curve represents the smooth evolution of the reward over time and an optional reference line marks the target score threshold. 

---

## Results

The aggregated training curves in all_training_scores.png reveal clear differences in learning speed, stability, and final performance across the six tested algorithms. When comparing all experiments, four models successfully surpassed the target score of −60 closer to 100k steps, without ever again declining in performance, indicating effective learning and coordination in the environment. Notably, three of these models reach the target before 50k evolution steps, demonstrating strong early convergence and robust optimization.

### IPPO_V1 achieves the fastest improvement

The IPPO model rises above the target score earlier than any other algorithm, showing a steep and consistent learning curve. This rapid progression is likely due to the on-policy nature of PPO, which updates policies using fresh trajectories and significantly reduces the instability caused by stale replay-buffer data. Because each agent learns independently, IPPO can adjust more rapidly to changes in the environment, making it more responsive in the early stages of training. This explains why IPPO quickly surpasses the −60 threshold and maintains strong performance afterward.

### MATD3_V4 obtains the best overall score (−18.33)

Although IPPO learns fastest, MATD3_V4 ultimately achieves the highest performance, reaching a maximum score of −18.33, the best result of all models. This version benefits from a larger network architecture (128-dim layers) combined with online hyperparameter adaptation through PBT, which allows the algorithm to gradually refine learning rates and update rules as training evolves. Despite being derived from MATD3, V4 demonstrates that increasing model capacity and enabling dynamic hyperparameter evolution significantly boosts performance, making it the most effective architecture in the long run.

---

## Summary

Four models surpass the target score, and three achieve this before 50k steps, showing strong convergence behavior.

IPPO is the fastest learner, likely due to its responsive on-policy updates and independent-agent structure.

MATD3_V4 is the best-performing model overall, delivering the highest final score across all experiments.
