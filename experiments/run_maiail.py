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
from algo.maiail import MAIAIL
from common.utils import Dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Behavioral cloning experiments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max_episode_len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--episodes", type=int, default=200, help="number of episodes")
    parser.add_argument("--iterations", type=int, default=1000, help="number of training iterations")
    # parser.add_argument("--num_agents", type=int, default=2, help="number of agents")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--entcoeff", type=float, default=0.001, help="the coefficient of the entropy loss of the discriminators")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="soft update factor")
    parser.add_argument("--batch_size", type=int, default=64, help="the batch size to optimize at the same time")
    parser.add_argument("--memory_size", type=int, default=10**4, help="the memory size of replay buffer")
    parser.add_argument("--dataset_size", type=int, default=65536, help="the dataset size for learning discriminators")
    # Checkpointing & Logging
    parser.add_argument("--exp_name", type=str, default="maiail", help="name of the experiment")
    parser.add_argument("--save_interval", type=int, default=400, help='Interval episode for saving model(default=400)')
    parser.add_argument("--save_dir", type=str, default="./trained_models/", help="Parent directory in which trained models should be saved")
    parser.add_argument("--data_dir", type=str, default="./expert_data/", help="Directory in which expert data is saved")
    parser.add_argument("--log_dir", type=str, default="./logs/", help="directory of logging")
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--load_dir", type=str, default="./trained_models/", help="Parent directory in which trained models are loaded")
    parser.add_argument("--load_epoch", type=int, default=1000, help="the epoch of loaded model")
    parser.add_argument("--max_to_keep", type=int, default=10, help="number of models to save")
    # Evaluation
    parser.add_argument("--is_evaluate", action="store_true",default=False, help='is training or evalutaion')
    # parser.add_argument("--eval_interval", type=int, default=200, help='Evaluation interval episode and save model(default=1000)')
    parser.add_argument("--render", action="store_true", default=False, help="do or not render")
    # parser.add_argument("--plots_dir", type=str, default="./plots/", help="directory where plot data is saved")
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

    num_agents = env.n
    # learning_dataset = Dataset(args.scenario, num_agents, args.batch_size)
    expert_dataset = Dataset(args.scenario, num_agents, args.batch_size, capacity=args.dataset_size)
    expert_dataset.load_data(args.data_dir)
    maiail = MAIAIL(sess, env, args.scenario, args.exp_name, num_agents, expert_dataset, args.batch_size, args.entcoeff, args.lr, args.gamma, args.tau, args.memory_size)

    if not is_evaluate:
        # initialize summary
        summary_d_loss = [None for _ in range(num_agents)]
        # summary_p_loss = [None for _ in range(num_agents)]
        summary_pa_loss = [None for _ in range(num_agents)]
        summary_pc_loss = [None for _ in range(num_agents)]
        for i in range(num_agents):
            summary_d_loss[i] = tf.placeholder(tf.float32, None)
            # summary_p_loss[i] = tf.placeholder(tf.float32, None)
            summary_pa_loss[i] = tf.placeholder(tf.float32, None)
            summary_pc_loss[i] = tf.placeholder(tf.float32, None)

            tf.summary.scalar('Discriminator-Loss-{}'.format(i), summary_d_loss[i])
            # tf.summary.scalar('Policy-Loss-{}'.format(i), summary_p_loss[i])
            tf.summary.scalar('Policy-Actor-Loss-{}'.format(i), summary_pa_loss[i])
            tf.summary.scalar('Policy-Critic-Loss-{}'.format(i), summary_pc_loss[i])

        summary_dict = dict()
        summary_dict['d_loss'] = summary_d_loss
        # summary_dict['p_loss'] = summary_p_loss
        summary_dict['pa_loss'] = summary_pa_loss
        summary_dict['pc_loss'] = summary_pc_loss

        merged = tf.summary.merge_all()

        log_dir = args.log_dir + args.scenario
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        else:
            tf.gfile.DeleteRecursively(log_dir)
            
        summary_writer = tf.summary.FileWriter(log_dir)

        maiail.init()  # run self.sess.run(tf.global_variables_initializer())

    if args.restore or is_evaluate:
        load_dir = os.path.join(args.load_dir, args.scenario)
        maiail.load(load_dir, epoch=args.load_epoch)

    # ======================================== main loop ======================================== #
    batch_size = args.batch_size
    t_start = time.time()
    total_step = 0
    if is_evaluate:
        num_episodes = 100
    p_loss, d_loss = None, None
    if not is_evaluate:
        print("Start Training...")
    else:
        print("Start Evaluating...")

    t_start = time.time()
    start_time = time.time()
    total_step = 0
    for iteration in range(args.iterations):
        # sample interations
        print("sample interactions...")
        episode_r_all_sum = []
        for ep in range(args.episodes):
            obs_n = env.reset()
            episode_r_n = [0. for _ in range(num_agents)]
            run_policy_steps = 0
            for step in range(args.max_episode_len):
                total_step += 1
                run_policy_steps += 1

                if args.render:
                    env.render(mode=None)

                act_n = maiail.act(obs_n)
                next_obs_n, reward_n, done_n, info_n = env.step(act_n)

                flag = maiail.store_data(obs_n, act_n, next_obs_n, done_n)

                done = all(done_n) 

                if done:
                    continue

                obs_n = next_obs_n
                episode_r_n = list(map(operator.add, episode_r_n, reward_n))
            
            episode_r_all_sum.append(np.sum(episode_r_n))
            if not flag:
                break

        print("--- iteration-{} | [mean-sample-sum-reward]: {}".format(iteration, np.mean(episode_r_all_sum)))
        if not is_evaluate:
            # p_loss = [[] for _ in range(num_agents)]
            # d_loss = [[] for _ in range(num_agents)]
            # a_loss, c_loss = [[] for _ in range(num_agents)], [[] for _ in range(num_agents)]

            print("iteration {} | training models...".format(iteration))
            t_info = maiail.train()

            pa_loss, pc_loss, d_loss = t_info['pa_loss'], t_info['pc_loss'], t_info['d_loss']

            feed_dict = dict()
            # feed_dict.update(zip(summary_dict['p_loss'], p_loss))
            feed_dict.update(zip(summary_dict['pa_loss'], pa_loss))
            feed_dict.update(zip(summary_dict['pc_loss'], pc_loss))
            feed_dict.update(zip(summary_dict['d_loss'], d_loss))

            summary = sess.run(merged, feed_dict=feed_dict)
            summary_writer.add_summary(summary, iteration)

            print("-----------------------------------------------------------------")
            print("----[iteration]: {} | [pa-loss]: {} | [pc-loss]: {} | [d-loss]: {} | [inter-time]: {}".format(iteration, pa_loss, pc_loss, d_loss, round(time.time()-t_start),4))
            print("\n-----------------------------------------------------------------")
            t_start = time.time()

            if (iteration+1) % args.save_interval == 0:
                maiail.save(args.save_dir, iteration, args.max_to_keep)

    env.close()
    summary_writer.close()
    sess.close()

