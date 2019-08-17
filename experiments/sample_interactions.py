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
from common.buffer import Dataset
from common.utils import BaseAgent
from algo.maddpg import MADDPG
from algo.rule import RuleAgent
from algo.behavior_clone import MABehavioralCloning
from algo.maiail_maddpg import MAIAIL

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)

# noinspection PyTypeChecker
def open_file_and_save(file_path, data):
    """
    :param file_path: type==string
    :param data:
    """
    try:
        with open(file_path, 'ab') as f_handle:
            np.savetxt(f_handle, data, fmt='%s')
    except FileNotFoundError:
        with open(file_path, 'wb') as f_handle:
            np.savetxt(f_handle, data, fmt='%s')

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Sample interactions experiments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--agent", type=str, default="rule", help="name of the sampling agents")
    parser.add_argument("--max_episode_len", type=int, default=50, help="maximum episode length")
    parser.add_argument("--discrete", action="store_true", default=False, help="is the action discrete")
    parser.add_argument("--episodes", type=int, default=100, help="number of episodes")
    # parser.add_argument("--num_agents", type=int, default=2, help="number of agents")
    # parser.add_argument("--good_policy", type=str, default="maddpg", help="policy for good agents")
    # parser.add_argument("--adv_policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument('--dataset_size', type=int, default=65536, help='Dataset size (default=2**16)')
    # Checkpointing & Logging
    parser.add_argument("--exp_name", type=str, default="sample_interactions", help="name of the experiment")
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--data_dir", type=str, default="./expert_data/", help="directory in which expert interactions are loaded")
    parser.add_argument("--load_dir", type=str, default="./trained_models/", help="directory in which training state and model are loaded")
    parser.add_argument("--load_epoch", type=int, default=None, help="the epoch of loaded model")
    # Evaluation
    parser.add_argument("--render", action="store_true", default=False, help="do or not render")
    args = parser.parse_args()

    print('=== Configuration:\n', args)

    # =========================== initialize environment =========================== #
    is_render = args.render    
    scenario = scenarios.load(args.scenario+".py").Scenario()
    world = scenario.make_world()

    env = multiagent.environment.MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)

    # =========================== initialize model and summary =========================== #
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # num_agents = min(args.num_agents, env.n)
    num_agents = env.n
    load_dir = os.path.join(args.load_dir, args.scenario)

    agent = BaseAgent(env, args.exp_name)
    if args.agent == 'maddpg':
        name = 'maddpg'
        agent = MADDPG(sess, env, name, num_agents)
        agent.init()
        agent.load(load_dir, epoch=args.load_epoch)
    elif args.agent == 'rule':
        agent = RuleAgent(env, args.exp_name, args.scenario, args.discrete)
    elif args.agent == 'bc':
        name = 'behavior_clone' +'/discrete-{}'.format(args.discrete)
        agent = BehaviorClone(sess, env, name, num_agents)
        agent.init()
        agent.load(load_dir, epoch=args.load_epoch)
    else:
        print("Not implement agent {}, use random agent!".format(args.agent))

    dataset = Dataset(args.scenario, num_agents, capacity=args.dataset_size)


    # ======================================== main loop ======================================== #
    num_episodes = args.episodes
    print("Start Sampling...")

    total_step = 0
    episode_r_n_sum = []
    for ep in range(0, num_episodes):
        t_start = time.time()
        obs_n = env.reset()
        print(obs_n)
        episode_r_n = [0. for _ in range(num_agents)]

        for step in range(0, args.max_episode_len):
            if is_render:
                # time.sleep(0.1)
                env.render(mode=None)
            act_n = agent.act(obs_n)
            next_obs_n, reward_n, done_n, info_n = env.step(act_n)
            done = all(done_n)

            # print(obs_n, act_n)
            flag = dataset.push(obs_n, act_n)

            obs_n = next_obs_n
            episode_r_n = list(map(operator.add, episode_r_n, reward_n))
            total_step += 1

            if done:
                break
        print(act_n)

        episode_r_n = [round(_, 3) for _ in episode_r_n]
        episode_r_n_sum.append(np.sum(episode_r_n))
        print("\n--- episode-{} | [reward]: {} | [sum-reward]: {}".format(ep+1, episode_r_n, sum(episode_r_n)))
        if not flag:
            break

    print("mean sum reward in {} episodes: {}".format(num_episodes, np.mean(episode_r_n_sum)))
    dataset.save_data(args.data_dir)
    env.close()
    sess.close()
