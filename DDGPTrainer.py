from tqdm import tqdm as tqdm
from unityagents import UnityEnvironment
from Agent import Agent
import torch
import numpy as np
from collections import deque
import pickle


def train_model(env, agent, brain_name, max_episodes, num_agents, max_t=10000, print_every=100):
    avg_scores = []
    max_scores = []
    solved = False

    scores_deque = deque(maxlen=print_every)
    with tqdm(range(1, max_episodes + 1)) as t:
        for i_episode in t:
            states = env.reset(train_mode=True)[brain_name].vector_observations
            agent.reset()
            scores = np.zeros(num_agents)

            for _ in range(max_t):
                actions = [agent.act(state) for state in states]
                env_info = env.step(actions)[brain_name]

                next_states = env_info.vector_observations
                rewards = env_info.rewards
                dones = env_info.local_done
                done = np.any(dones)

                for agent_idx in range(num_agents):
                    agent.step(states[agent_idx], actions[agent_idx], rewards[agent_idx], next_states[agent_idx], done)

                states = next_states
                scores += rewards
                if done:
                    break

            max_score = np.max(scores)
            scores_deque.append(max_score)
            max_scores.append(max_score)
            avg_scores.append(np.mean(scores_deque))

            t.set_postfix(avg_score=np.mean(scores_deque), score=scores, max_score=max_score)
            if i_episode % print_every == 0:
                t.write('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            if np.mean(scores_deque) >= 0.5:
                t.write('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                    i_episode - 100, np.mean(scores_deque)))
                solved = True
                break

    return max_scores, avg_scores, solved


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--unity_filename", default="Tennis_Linux/Tennis.x86_64")
    parser.add_argument("--actor_model_filename", default="actor.torch")
    parser.add_argument("--critic_model_filename", default="critic.torch")
    parser.add_argument("--scores_filename", default="tennis_scores.pkl")
    parser.add_argument("--seed", default=2)
    parser.add_argument("--max_episodes", default=2000, type=int)
    args = parser.parse_args()

    print("Initializing unity")
    env_ = UnityEnvironment(file_name=args.unity_filename, no_graphics=True)
    print("Unity ready")

    # get the default brain
    brain_name_ = env_.brain_names[0]
    brain_ = env_.brains[brain_name_]

    env_info_ = env_.reset(train_mode=True)[brain_name_]

    action_size_ = brain_.vector_action_space_size
    state_ = env_info_.vector_observations[0]
    state_size_ = len(state_)

    agent_ = Agent(state_size=state_size_, action_size=action_size_, random_seed=args.seed, )

    scores_, avg_scores_, solved_ = train_model(
        env=env_, agent=agent_, brain_name=brain_name_, max_episodes=args.max_episodes, num_agents=2)

    if solved_:
        torch.save(agent_.actor_local.state_dict(), open(args.actor_model_filename, "wb"))
        print("Actor network saved to", args.actor_model_filename)
        torch.save(agent_.critic_local.state_dict(), open(args.critic_model_filename, "wb"))
        print("Critic network saved to", args.critic_model_filename)
        pickle.dump({"scores": scores_, "avg_scores": avg_scores_}, open(args.scores_filename, "wb"))
        print("Scores saved to ", args.scores_filename)
    else:
        print("Failed to solve, network not saved")

    env_.close()
