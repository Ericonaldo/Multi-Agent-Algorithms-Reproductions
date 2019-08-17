import os
import sys
import time
import argparse
import operator
import numpy as np
import tensorflow as tf

sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '../ma_env/multiagent-particle-envs'))

import multiagent
import multiagent.scenarios as scenarios
from multiagent.environment import MultiAgentEnv
from common.utils import softmax
from algo.rule import RuleAgent



if __name__ == '__main__':
    parser = argparse.ArgumentParser("Rule agent experiments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max_episode_len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--episodes", type=int, default=100, help="number of episodes")
    parser.add_argument("--discrete", action="store_true", default=False, help="is the action discrete")
    # parser.add_argument("--num_agents", type=int, default=2, help="number of agents")
    # parser.add_argument("--good_policy", type=str, default="maddpg", help="policy for good agents")
    # parser.add_argument("--adv_policy", type=str, default="maddpg", help="policy of adversaries")
    # Checkpointing & Logging
    parser.add_argument("--exp_name", type=str, default="rule agents", help="name of the experiment")
    # Evaluation
    parser.add_argument("--render", action="store_true", default=False, help="do or not render")
    args = parser.parse_args()

    print('=== Configuration:\n', args)

    support = set(['simple_speaker_listener'])
    assert(args.scenario in support)

    # =========================== initialize environment =========================== #
    is_render = args.render    
    scenario = scenarios.load(args.scenario+".py").Scenario()
    world = scenario.make_world()

    env = multiagent.environment.MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    num_agents = env.n
    agent = RuleAgent(env, args.exp_name, args.scenario, args.discrete)

    # ======================================== main loop ======================================== #
    num_episodes = args.episodes
    print("Start...")

    total_step = 0
    episode_r_n_sum = []
    episode_r_n_all = []
    for ep in range(0, num_episodes):
        t_start = time.time()
        obs_n = env.reset()
        episode_r_n = [0. for _ in range(num_agents)]

        for step in range(0, args.max_episode_len):
            if is_render:
                # time.sleep(0.1)
                env.render(mode=None)
            act_n = agent.act(obs_n)
            next_obs_n, reward_n, done_n, info_n = env.step(act_n)
            done = all(done_n)
            # print("obs: {} | act: {}".format(obs_n, act_n))

            obs_n = next_obs_n
            episode_r_n = list(map(operator.add, episode_r_n, reward_n))
            total_step += 1

            if done:
                break

        print("\n--- episode-{} | [sum-reward]: {}".format(ep+1, np.sum(episode_r_n)))
        episode_r_n_all.append(episode_r_n)
        episode_r_n_sum.append(np.sum(episode_r_n))
    print("mean sum reward in {} episodes: {}".format(num_episodes, np.mean(episode_r_n_sum)))
    env.close()
