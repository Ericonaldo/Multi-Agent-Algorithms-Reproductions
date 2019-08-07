import os
import sys
import time
import argparse
import operator
import tensorflow as tf
import numpy as np

sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '../ma_env/multiagent-particle-envs'))

import multiagent
import multiagent.scenarios as scenarios
from multiagent.environment import MultiAgentEnv
from algo.maddpg import MADDPG


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)


if __name__ == '__main__':

    parser = argparse.ArgumentParser("MADDPG experiments")
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
    parser.add_argument("--batch_size", type=int, default=512, help="the batch size to optimize at the same time")
    parser.add_argument('--memory_size', type=int, default=10**4, help='Memory size (default=10**4)')
    # Checkpointing & Logging
    parser.add_argument("--exp_name", type=str, default="maddpg", help="name of the experiment")
    parser.add_argument("--save_interval", type=int, default=400, help='Interval episode for saving model(default=400)')
    parser.add_argument("--save_dir", type=str, default="./trained_models/", help="directory in which training state and model should be saved")
    parser.add_argument("--log_dir", type=str, default="./logs/", help="directory of logging")
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--load_dir", type=str, default="./trained_models/", help="directory in which training state and model are loaded")
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
    step = 0
    scenario = scenarios.load(args.scenario+".py").Scenario()
    world = scenario.make_world()

    env = multiagent.environment.MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)

    # =========================== initialize model and summary =========================== #
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
  
    num_agents = min(args.num_agents, env.n)
    maddpg = MADDPG(sess, env, args.exp_name, num_agents, args.batch_size, args.actor_lr, args.critic_lr, args.gamma,
                        args.tau, args.memory_size, grad_norm_clipping=0.5)

    if not is_evaluate:
        # initialize summary
        summary_r = [None for _ in range(num_agents)]

        for i in range(num_agents):
            summary_r[i] = tf.placeholder(tf.float32, None)
            tf.summary.scalar('Episode-Reward-{}'.format(i), summary_r[i])

        summary_dict = {'reward': summary_r}

        summary_a_loss, summary_c_loss = [None for _ in range(num_agents)], [None for _ in range(num_agents)]
        for i in range(num_agents):
            summary_a_loss[i] = tf.placeholder(tf.float32, None)
            summary_c_loss[i] = tf.placeholder(tf.float32, None)

            tf.summary.scalar('Actor-Loss-{}'.format(i), summary_a_loss[i])
            tf.summary.scalar('Critic-Loss-{}'.format(i), summary_c_loss[i])

        summary_dict['a_loss'] = summary_a_loss
        summary_dict['c_loss'] = summary_c_loss

        merged = tf.summary.merge_all()

        log_dir = args.log_dir + args.scenario
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        else:
            tf.gfile.DeleteRecursively(log_dir)
            
        summary_writer = tf.summary.FileWriter(log_dir)

        maddpg.init()  # run self.sess.run(tf.global_variables_initializer()) and hard update

    if args.restore or is_evaluate:
        load_dir = args.load_dir + args.scenario
        maddpg.load(load_dir, epoch=args.load_epoch)

    # ======================================== main loop ======================================== #
    num_episodes = args.train_episodes
    if is_evaluate:
        num_episodes = 100
    a_loss, c_loss = None, None
    if not is_evaluate:
        print("Start Training...")
    else:
        print("Start Evaluating...")

    t_start = time.time()
    total_step = 0
    t_info_n = None
    episode_r_all = []
    for ep in range(0, num_episodes):
        a_loss = [[] for _ in range(num_agents)]
        c_loss = [[] for _ in range(num_agents)]

        obs_n = env.reset()
        episode_r_n = [0. for _ in range(num_agents)]

        for step in range(0, args.max_episode_len):
            if is_render:
                # time.sleep(0.1)
                env.render(mode=None)
            act_n = maddpg.act(obs_n)
            # print(act_n)
            next_obs_n, reward_n, done_n, info_n = env.step(act_n)
            done = all(done_n)

            if not is_evaluate:  # trigger for data collection
                maddpg.store_trans(obs_n, act_n, next_obs_n, reward_n, done_n)

            obs_n = next_obs_n

            episode_r_n = list(map(operator.add, episode_r_n, reward_n))

            total_step += 1

        # =============================== render / record / model saving ===============================
            # every step
            if not is_evaluate and ((step+1) % 20 ==0):  # trigger for training
                t_info_n = maddpg.train()
                # print('step: {}, {}'.format(step, t_info_n))

                if t_info_n is not None:
                    a_loss = map(lambda x, y: y + [x], t_info_n['a_loss'], a_loss)
                    c_loss = map(lambda x, y: y + [x], t_info_n['c_loss'], c_loss)
            if done:
                break

        # every episode
        episode_r_n = [round(_, 3) for _ in episode_r_n]
        episode_r_all.append(episode_r_n)

        # if t_info_n is not None:
        # if not is_evaluate:
        if (not is_evaluate) and (t_info_n is not None):
            a_loss = list(map(lambda x: round(sum(x) / len(x), 3), a_loss))
            c_loss = list(map(lambda x: round(sum(x) / len(x), 3), c_loss))
            feed_dict = dict()
            feed_dict.update(zip(summary_dict['reward'], episode_r_n))
            feed_dict.update(zip(summary_dict['a_loss'], a_loss))
            feed_dict.update(zip(summary_dict['c_loss'], c_loss))

            summary = sess.run(merged, feed_dict=feed_dict)
            summary_writer.add_summary(summary, ep)

            if (ep+1) % args.save_interval == 0:
                maddpg.save(args.save_dir+args.scenario, ep+1, args.max_to_keep)
                print("\n--- episode-{} | [a-loss]: {} | [c-loss]: {} | [mean-reward]: {} | [inter-time]: {}".format(ep+1, a_loss, c_loss, np.mean(episode_r_all[-args.save_interval:], axis=0), round(time.time()-t_start),4))
                t_start = time.time()

        if is_evaluate:
            print("\n--- episode-{} | [mean-reward]: {} | [inter-time]: {}".format(ep+1, np.mean(episode_r_n, axis=0), round(time.time()-t_start),4))
            t_start = time.time()
           
    env.close()
    if not is_evaluate:
        summary_writer.close()
    sess.close()
