"""This tutorial shows how to train an MATD3 agent on the simple speaker listener multi-particle environment.

Authors: Michael (https://github.com/mikepratt1), Nickua (https://github.com/nicku-a)
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from pettingzoo.mpe import simple_speaker_listener_v4
from mpe2 import simple_speaker_listener_v4

from agilerl.algorithms import MATD3
from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.utils import (
    create_population,
    default_progress_bar,
    make_multi_agent_vect_envs,
)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("===== AgileRL Online Multi-Agent Demo V3 =====")

    # NET_CONFIG = {
    #     "latent_dim": 64,
    #     "encoder_config": {
    #         "hidden_size": [64, 64],  # Actor
    #     },
    #     "head_config": {
    #         "hidden_size": [64, 64],  # Critic
    #     },
    # }

    NET_CONFIG = {
        "latent_dim": 128,
        "encoder_config": {"hidden_size": [128, 128]},
        "head_config": {"hidden_size": [128, 128]},
    }   

    INIT_HP = {
    "POPULATION_SIZE": 6,     # a bit larger pop for PBT
    "ALGO": "MATD3",
    "BATCH_SIZE": 128,
    "O_U_NOISE": True,
    "EXPL_NOISE": 0.1,        # reduce the exploration agains v2
    "MEAN_NOISE": 0.0,
    "THETA": 0.15,
    "DT": 0.01,
    "LR_ACTOR": 3e-4,
    "LR_CRITIC": 3e-4,
    # "GAMMA": 0.95,
    "GAMMA": 0.99,            
    "MEMORY_SIZE": 150000,
    "LEARN_STEP": 75,         # learn more often
    "TAU": 0.005,             # softer target updates (TD3-style)
    "POLICY_FREQ": 1,         # reduce agains v2
    }

    # EXPL_NOISE_START = INIT_HP["EXPL_NOISE"]   # 0.1
    # EXPL_NOISE_END = 0.02
    # EXPL_NOISE_DECAY_STEPS = 200_000

    EXPL_NOISE_START = INIT_HP["EXPL_NOISE"]   # 0.1
    EXPL_NOISE_END = 0.01
    EXPL_NOISE_DECAY_STEPS = 100_000

    
    def get_expl_noise(total_steps):
        """Linear decay from EXPL_NOISE_START → EXPL_NOISE_END"""
        ratio = min(total_steps / EXPL_NOISE_DECAY_STEPS, 1.0)
        return EXPL_NOISE_START + ratio * (EXPL_NOISE_END - EXPL_NOISE_START)

    num_envs = 8

    def make_env():
        return simple_speaker_listener_v4.parallel_env(continuous_actions=True)

    env = make_multi_agent_vect_envs(env=make_env, num_envs=num_envs)

    # Configure the multi-agent algo input arguments
    observation_spaces = [env.single_observation_space(agent) for agent in env.agents]
    action_spaces = [env.single_action_space(agent) for agent in env.agents]

    # Append number of agents and agent IDs to the initial hyperparameter dictionary
    INIT_HP["AGENT_IDS"] = env.agents

    # Mutation config for RL hyperparameters
    # hp_config = HyperparameterConfig(
    #     lr_actor=RLParameter(min=1e-4, max=1e-2),
    #     lr_critic=RLParameter(min=1e-4, max=1e-2),
    #     batch_size=RLParameter(min=8, max=512, dtype=int),
    #     learn_step=RLParameter(
    #         min=20, max=200, dtype=int, grow_factor=1.5, shrink_factor=0.75
    #     ),
    # )

    hp_config = HyperparameterConfig(
        lr_actor=RLParameter(min=1e-4, max=5e-4),
        lr_critic=RLParameter(min=1e-4, max=5e-4),
        batch_size=RLParameter(min=64, max=256, dtype=int),
        learn_step=RLParameter(min=40, max=120, dtype=int, grow_factor=1.2, shrink_factor=0.8),
        )


    pop: list[MATD3] = create_population(
        algo=INIT_HP["ALGO"],
        net_config=NET_CONFIG,
        INIT_HP=INIT_HP,
        observation_space=observation_spaces,
        action_space=action_spaces,
        hp_config=hp_config,
        population_size=INIT_HP["POPULATION_SIZE"],
        num_envs=num_envs,
        device=device,
    )


    # Configure the multi-agent replay buffer
    field_names = ["obs", "action", "reward", "next_obs", "done"]
    memory = MultiAgentReplayBuffer(
        INIT_HP["MEMORY_SIZE"],
        field_names=field_names,
        agent_ids=INIT_HP["AGENT_IDS"],
        device=device,
    )

    # Instantiate a tournament selection object (used for HPO)
    tournament = TournamentSelection(
        tournament_size=2,  # Tournament selection size
        elitism=True,  # Elitism in tournament selection
        population_size=INIT_HP["POPULATION_SIZE"],  # Population size
        eval_loop=1,  # Evaluate using last N fitness scores
    )

    mutations = Mutations(
    no_mutation=0.5,       # more likely to keep good params
    architecture=0.05,     # fewer random layers
    new_layer_prob=0.05,
    parameters=0.2,        # safer
    activation=0.0,
    rl_hp=0.1,             # less hyperparam chaos
    mutation_sd=0.03,      # smaller jumps
    rand_seed=1,
    device=device,
    )

    # Define training loop parameters
    max_steps = 2_000_000  # Max steps (default: 2000000)
    learning_delay = 0  # Steps before starting learning
    evo_steps = 10_000  # Evolution frequency
    eval_steps = None  # Evaluation steps per episode - go until done
    eval_loop = 1  # Number of evaluation episodes
    elite = pop[0]  # Assign a placeholder "elite" agent
    total_steps = 0
    
    # Lista para armazenar pontuações médias para plotagem
    training_scores_history = []

    # TRAINING LOOP
    print("Training...")
    pbar = default_progress_bar(max_steps)
    while np.less([agent.steps[-1] for agent in pop], max_steps).all():
        pop_episode_scores = []
        for agent in pop:  # Loop through population
            agent.set_training_mode(True)
            obs, info = env.reset()  # Reset environment at start of episode
            scores = np.zeros(num_envs)
            completed_episode_scores = []
            steps = 0
            for idx_step in range(evo_steps // num_envs):
                # !--- Exploration noise linear decay ---
                expl_noise = get_expl_noise(total_steps)
                agent.EXPL_NOISE = expl_noise
                #! --------------------------------------
                action, raw_action = agent.get_action(obs=obs, infos=info) 
                next_obs, reward, termination, truncation, info = env.step(action) 
                scores += np.sum(np.array(list(reward.values())).transpose(), axis=-1)
                total_steps += num_envs
                steps += num_envs

                # Save experiences to replay buffer
                memory.save_to_memory(
                    obs,
                    raw_action,
                    reward,
                    next_obs,
                    termination,
                    is_vectorised=True,
                )

                # Learn according to learning frequency
                # Handle learn steps > num_envs
                if agent.learn_step > num_envs:
                    learn_step = agent.learn_step // num_envs
                    if (
                        idx_step % learn_step == 0
                        and len(memory) >= agent.batch_size
                        and memory.counter > learning_delay
                    ):
                        experiences = memory.sample(
                            agent.batch_size
                        )  # Sample replay buffer
                        agent.learn(
                            experiences
                        )  # Learn according to agent's RL algorithm

                # Handle num_envs > learn step; learn multiple times per step in env
                elif (
                    len(memory) >= agent.batch_size and memory.counter > learning_delay
                ):
                    for _ in range(num_envs // agent.learn_step):
                        experiences = memory.sample(
                            agent.batch_size
                        )  # Sample replay buffer
                        agent.learn(
                            experiences
                        )  # Learn according to agent's RL algorithm

                obs = next_obs

                # Calculate scores and reset noise for finished episodes
                reset_noise_indices = []
                term_array = np.array(list(termination.values())).transpose()
                trunc_array = np.array(list(truncation.values())).transpose()
                for idx, (d, t) in enumerate(zip(term_array, trunc_array)):
                    if np.any(d) or np.any(t):
                        completed_episode_scores.append(scores[idx])
                        agent.scores.append(scores[idx])
                        scores[idx] = 0
                        reset_noise_indices.append(idx)

                agent.reset_action_noise(reset_noise_indices)

            pbar.update(evo_steps // len(pop))

            agent.steps[-1] += steps
            pop_episode_scores.append(completed_episode_scores)

        # Evaluate population
        fitnesses = [
            agent.test(
                env,
                max_steps=eval_steps,
                loop=eval_loop,
            )
            for agent in pop
        ]
        mean_scores = [
            (
                np.mean(episode_scores)
                if len(episode_scores) > 0
                else 0
            )
            for episode_scores in pop_episode_scores
        ]
        
        # Salvar pontuação média da população para plotagem
        population_mean_score = np.mean([score for score in mean_scores if isinstance(score, (int, float))])
        training_scores_history.append(population_mean_score)

        mean_scores_display = [
            (
                score if isinstance(score, (int, float))
                else "0 completed episodes"
            )
            for score in mean_scores
        ]

        pbar.write(
            f"--- Global steps {total_steps} ---\n"
            f"Steps {[agent.steps[-1] for agent in pop]}\n"
            f"Scores: {mean_scores_display}\n"
            f"Fitnesses: {['%.2f' % fitness for fitness in fitnesses]}\n"
            f"5 fitness avgs: {['%.2f' % np.mean(agent.fitness[-5:]) for agent in pop]}\n"
            f"Mutations: {[agent.mut for agent in pop]}"
        )

        # Tournament selection and population mutation
        elite, pop = tournament.select(pop)
        pop = mutations.mutation(pop)

        # Update step counter
        for agent in pop:
            agent.steps.append(agent.steps[-1])

    # Save the trained algorithm
    path = "./models/MATD3_V3"
    filename = "MATD3_trained_agent_V3.pt"
    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, filename)
    elite.save_checkpoint(save_path)
    
    # Plotar e salvar a evolução das pontuações
    plt.figure(figsize=(12, 6))
    plt.plot(training_scores_history, linewidth=2)
    plt.title('Evolução das Pontuações Médias Durante o Treinamento', fontsize=14)
    plt.xlabel('Iterações de Evolução', fontsize=12)
    plt.ylabel('Pontuação Média da População', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Salvar o gráfico
    plot_path = os.path.join(path, "training_scores_evolution_V3.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Gráfico de evolução das pontuações salvo em: {plot_path}")
    
    # Salvar dados das pontuações em arquivo numpy
    scores_data_path = os.path.join(path, "training_scores_history_V3.npy")
    np.save(scores_data_path, np.array(training_scores_history))
    print(f"Dados das pontuações salvos em: {scores_data_path}")
    
    plt.show()

    pbar.close()
    env.close()
