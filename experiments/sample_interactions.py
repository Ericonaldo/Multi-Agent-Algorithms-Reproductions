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
from algo.behavior_clone import BehavioralCloning

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
    parser.add_argument("--max_episode_len", type=int, default=40, help="maximum episode length")
    parser.add_argument("--train_episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--num_agents", type=int, default=2, help="number of agents")
    # parser.add_argument("--good_policy", type=str, default="maddpg", help="policy for good agents")
    # parser.add_argument("--adv_policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--actor_lr", type=float, default=1e-2, help="learning rate of actor for Adam optimizer")
    parser.add_argument("--critic_lr", type=float, default=1e-2, help="learning rate of critic for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help='Hyper-parameter for soft update (default=0.01)')
    parser.add_argument("--batch_size", type=int, default=512, help="number of episodes to optimize at the same time")
    parser.add_argument('--dataset_size', type=int, default=10**4, help='Memory size (default=10**4)')
    # Checkpointing & Logging
    parser.add_argument("--exp_name", type=str, default="sample_interactions", help="name of the experiment")
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--load_dir", type=str, default="./trained_models/", help="directory in which training state and model are loaded")
    parser.add_argument("--load_epoch", type=int, default=1000, help="the epoch of loaded model")
    # Evaluation
    parser.add_argument("--render", action="store_true", default=False, help="do or not render")
    args = parser.parse_args()

    print('=== Configuration:\n', args)

    # =========================== initialize environment =========================== #
    is_evaluate = args.is_evaluate
    is_render = args.render or args.is_evaluate

    scenario = scenarios.load(args.scenario+".py").Scenario()
    world = scenario.make_world()

    env = multiagent.environment.MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)

    # =========================== initialize model and summary =========================== #
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    num_agents = min(args.num_agents, env.n)

    maddpg = MADDPG(sess, env, args.exp_name, num_agents, args.batch_size, args.actor_lr, args.critic_lr, args.gamma,
                        args.tau)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, args.model)
        obs = env.reset()

        for iteration in range(args.iteration):  # episode
            observations = []
            actions = []
            run_steps = 0
            while True:
                run_steps += 1
                # prepare to feed placeholder Policy.obs
                obs = np.stack([obs]).astype(dtype=np.float32)

                act, _ = Policy.act(obs=obs, stochastic=True)
                act = np.asscalar(act)

                observations.append(obs)
                actions.append(act)

                next_obs, reward, done, info = env.step(act)

                if done:
                    print(run_steps)
                    obs = env.reset()
                    break
                else:
                    obs = next_obs

            observations = np.reshape(observations, newshape=[-1] + list(ob_space.shape))
            actions = np.array(actions).astype(dtype=np.int32)

            open_file_and_save('trajectory/observations.csv', observations)
            open_file_and_save('trajectory/actions.csv', actions)

