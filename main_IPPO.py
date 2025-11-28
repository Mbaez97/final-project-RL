import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Silencing normal warnings from "Gymnasium"
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium") 

try:
    from mpe2 import simple_speaker_listener_v4
except ImportError:
    from pettingzoo.mpe import simple_speaker_listener_v4

from agilerl.algorithms import IPPO
from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter
# Dont import MultiAgentReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.utils import (
    create_population,
    default_progress_bar,
    make_multi_agent_vect_envs,
)

# Helper function to move tensors from 'GPU' to 'CPU Numpy'
def to_cpu_numpy(data):
    if isinstance(data, dict):
        new_dict = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                new_dict[k] = v.detach().cpu().numpy()
            else:
                new_dict[k] = v
        return new_dict
    elif isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return data

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("===== AgileRL Online Multi-Agent Demo IPPO (On-Policy) =====")
    print(f"Device: {device}")

    NET_CONFIG = {
        "latent_dim": 128,
        "encoder_config": {"hidden_size": [128, 128]},
        "head_config": {"hidden_size": [128, 128]},
    }   

    INIT_HP = {
        "POPULATION_SIZE": 4,     
        "ALGO": "IPPO",
        "BATCH_SIZE": 512, # Not used directly in learn() for full IPPO, but used in config
        "LEARN_STEP": 2048, # Steps to collect before training
        "LR": 3e-4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,         
        "CLIP_EPS": 0.2,          
        "ENT_COEF": 0.01,                 
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "MEMORY_SIZE": 10000, # Irrelevant now without buffer
    }

    num_envs = 8

    def make_env():
        return simple_speaker_listener_v4.parallel_env(
            continuous_actions=True, 
            max_cycles=25
        )

    env = make_multi_agent_vect_envs(env=make_env, num_envs=num_envs)

    try:
        observation_spaces = {agent: env.single_observation_space(agent) for agent in env.agents}
        action_spaces = {agent: env.single_action_space(agent) for agent in env.agents}
    except AttributeError:
        observation_spaces = [env.single_observation_space(agent) for agent in env.agents]
        action_spaces = [env.single_action_space(agent) for agent in env.agents]

    INIT_HP["AGENT_IDS"] = env.agents

    hp_config = HyperparameterConfig(
        lr=RLParameter(min=1e-5, max=1e-3), 
        batch_size=RLParameter(min=128, max=1024, dtype=int),
        learn_step=RLParameter(min=1024, max=4096, dtype=int, grow_factor=1.2, shrink_factor=0.8),
    )

    pop = create_population(
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

    # We dont use Replay Buffer for IPPO (ON-POLICY)

    tournament = TournamentSelection(
        tournament_size=2,  
        elitism=True,  
        population_size=INIT_HP["POPULATION_SIZE"],  
        eval_loop=1, 
    )

    mutations = Mutations(
        no_mutation=0.6, 
        architecture=0.1,  
        new_layer_prob=0.1,
        parameters=0.1, 
        activation=0.0,
        rl_hp=0.1,  
        mutation_sd=0.05, 
        rand_seed=1,
        device=device,
    )

    max_steps = 2_000_000 
    evo_steps = 20_000  
    eval_steps = None  
    eval_loop = 1  
    
    total_steps = 0
    training_scores_history = []

    print("Training IPPO (On-Policy)...")
    pbar = default_progress_bar(max_steps)
    
    while np.less([agent.steps[-1] for agent in pop], max_steps).all():
        pop_episode_scores = []

        for agent in pop: 
            agent.set_training_mode(True)
            obs, info = env.reset()
            
            scores = np.zeros(num_envs)
            completed_episode_scores = []
            steps = 0

            # Manual Storage Structure (Lists instead of Buffer)
            # This maintains the temporal order necessary for 'IPPO'
            mb_obs = {agent_id: [] for agent_id in env.agents}
            mb_actions = {agent_id: [] for agent_id in env.agents}
            mb_log_probs = {agent_id: [] for agent_id in env.agents}
            mb_rewards = {agent_id: [] for agent_id in env.agents}
            mb_dones = {agent_id: [] for agent_id in env.agents}
            mb_values = {agent_id: [] for agent_id in env.agents}

            # Calculate how many steps we need to fill the learn_step
            # If learn_step is 2048 and we have 8 envs, we iterate 256 times.
            steps_to_rollout = agent.learn_step // num_envs

            for idx_step in range(steps_to_rollout):
                
                # Get action
                action, log_prob, _, value = agent.get_action(obs=obs, infos=info)
                
                # Convert to CPU Numpy
                action = to_cpu_numpy(action)
                log_prob = to_cpu_numpy(log_prob)
                value = to_cpu_numpy(value)

                # Save data before Stepping (State t)
                for agent_id in env.agents:
                    mb_obs[agent_id].append(obs[agent_id])
                    mb_actions[agent_id].append(action[agent_id])
                    mb_log_probs[agent_id].append(log_prob[agent_id])
                    mb_values[agent_id].append(value[agent_id])

                # Clipping & Step
                clipped_action = {aid: np.clip(act, 0, 1) for aid, act in action.items()}
                next_obs, reward, termination, truncation, info = env.step(clipped_action)
                
                # Calculate 'Dones'
                dones_step = {}
                for agent_id in termination:
                    t_done = termination[agent_id]
                    t_trunc = truncation[agent_id]
                    dones_step[agent_id] = np.logical_or(t_done, t_trunc).astype(float)

                # Save resulting Rewards and Dones (t -> t+1)
                for agent_id in env.agents:
                    mb_rewards[agent_id].append(reward[agent_id])
                    mb_dones[agent_id].append(dones_step[agent_id])

                scores += np.sum(np.array(list(reward.values())).transpose(), axis=-1)
                total_steps += num_envs
                steps += num_envs

                # Update State
                obs = next_obs

                # Metrics
                term_array = np.array(list(termination.values())).transpose()
                trunc_array = np.array(list(truncation.values())).transpose()
                for idx, (d, t) in enumerate(zip(term_array, trunc_array)):
                    if np.any(d) or np.any(t):
                        completed_episode_scores.append(scores[idx])
                        agent.scores.append(scores[idx])
                        scores[idx] = 0

            # Learning: Process the collected experiences
            # We pack the lists into the structure that IPPO expects
            # IPPO needs final 'next_obs' and 'next_done' for bootstrapping
            experiences = (
                mb_obs,         # Dict of lists
                mb_actions,     # Dict of lists
                mb_log_probs,   # Dict of lists
                mb_rewards,     # Dict of lists
                mb_dones,       # Dict of lists
                mb_values,      # Dict of lists
                next_obs,       # Final observation (for bootstrap)
                dones_step      # Final done (for bootstrap)
            )
            
            agent.learn(experiences)
            
            pbar.update(steps // len(pop))
            agent.steps[-1] += steps
            pop_episode_scores.append(completed_episode_scores)

        # Evaluation
        if pop[0].steps[-1] % evo_steps < (agent.learn_step * 2):
            fitnesses = [
                agent.test(
                    env,
                    max_steps=eval_steps,
                    loop=eval_loop,
                )
                for agent in pop
            ]
            
            mean_scores = [
                (np.mean(episode_scores) if len(episode_scores) > 0 else 0)
                for episode_scores in pop_episode_scores
            ]
            
            val_score = np.mean([s for s in mean_scores if isinstance(s, (int, float))])
            training_scores_history.append(val_score)

            pbar.write(
                f"--- Global steps {total_steps} ---\n"
                f"Mean Score: {val_score:.2f}\n"
                f"Fitnesses: {['%.2f' % f for f in fitnesses]}\n"
            )

            elite, pop = tournament.select(pop)
            pop = mutations.mutation(pop)
            
            for agent in pop:
                if len(agent.steps) < len(elite.steps):
                    agent.steps = elite.steps[:]

    path = "./models/IPPO_V1"
    os.makedirs(path, exist_ok=True)
    elite = pop[0] 
    save_path = os.path.join(path, "IPPO_trained_agent_V1.pt")
    elite.save_checkpoint(save_path)
    
    plt.figure(figsize=(12, 6))
    plt.plot(training_scores_history, linewidth=2)
    plt.title('Training Score Evolution (IPPO)')
    plt.xlabel('Evolution Steps')
    plt.ylabel('Mean Score')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(path, "training_scores_evolution_V1.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    scores_data_path = os.path.join(path, "training_scores_history_V1.npy")
    np.save(scores_data_path, np.array(training_scores_history))
    
    pbar.close()
    env.close()