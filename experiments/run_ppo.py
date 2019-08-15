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
from common.buffer import PPODataset
from algo.ppo import PPO

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("PPO experiments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max_episode_len", type=int, default=33, help="maximum episode length")
    parser.add_argument("--episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--epoch", type=int, default=6, help="number of training epochs")
    # parser.add_argument("--num_agents", type=int, default=2, help="number of agents")
    # Core training parameters
    parser.add_argument("--units", type=int, default=64, help="the hidden units of the nn")
    parser.add_argument("--actor_lr", type=float, default=1e-2, help="learning rate of actor for Adam optimizer")
    parser.add_argument("--critic_lr", type=float, default=1e-2, help="learning rate of critic for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--clip_value", type=float, default=0.2, help="clipped value")
    parser.add_argument("--batch_size", type=int, default=32, help="the batch size to optimize at the same time")
    # Checkpointing & Logging
    parser.add_argument("--exp_name", type=str, default="ppo", help="name of the experiment")
    parser.add_argument("--save_interval", type=int, default=100, help='Interval episode for saving model(default=400)')
    parser.add_argument("--save_dir", type=str, default="./trained_models/", help="directory in which training state and model should be saved")
    parser.add_argument("--log_dir", type=str, default="./logs/", help="directory of logging")
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--load_dir", type=str, default="./trained_models/", help="directory in which training state and model are loaded")
    parser.add_argument("--load_epoch", type=int, default=None, help="the epoch of loaded model")
    parser.add_argument("--max_to_keep", type=int, default=10, help="number of models to save")
    # Evaluation
    parser.add_argument("--is_evaluate", action="store_true",default=False, help='is training or evalutaion')
    parser.add_argument("--render", action="store_true", default=False, help="do or not render")
    # parser.add_argument("--plots_dir", type=str, default="./plots/", help="directory where plot data is saved")
    args = parser.parse_args()

    print('=== Configuration:\n', args)


    # =========================== initialize environment =========================== #
    is_render = args.render    
    scenario = scenarios.load(args.scenario+".py").Scenario()
    world = scenario.make_world()

    env = multiagent.environment.MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    num_agents = env.n

    # =========================== initialize model and summary =========================== #
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    is_evaluate = args.is_evaluate
  
    # num_agents = min(args.num_agents, env.n)
    num_agents = env.n
    ppo = [None for _ in range(num_agents)]
    for i in range(num_agents):
        ppo[i] = PPO(sess, env, args.exp_name, i, args.actor_lr, args.critic_lr, args.gamma,
                            clip_value=0.2, num_units=args.units)
        ppo[i].init()  # run self.sess.run(tf.global_variables_initializer()) 

    dataset = PPODataset(args.exp_name, num_agents, batch_size=args.batch_size)

    if not is_evaluate:
        # initialize summary
        summary_r = [None for _ in range(num_agents)]
        for i in range(num_agents):
            summary_r[i] = tf.placeholder(tf.float32, None)
            tf.summary.scalar('Episode-Reward-{}'.format(i), summary_r[i])

        summary_dict = {'agent_reward': summary_r}

        summary_r_all = tf.placeholder(tf.float32, None)
        tf.summary.scalar('Episode-Reward-Sum', summary_r_all)

        summary_dict['sum_reward'] = summary_r_all

        summary_a_loss, summary_c_loss = [None for _ in range(num_agents)], [None for _ in range(num_agents)]
        for i in range(num_agents):
            summary_a_loss[i] = tf.placeholder(tf.float32, None)
            summary_c_loss[i] = tf.placeholder(tf.float32, None)

            tf.summary.scalar('Actor-Loss-{}'.format(i), summary_a_loss[i])
            tf.summary.scalar('Critic-Loss-{}'.format(i), summary_c_loss[i])

        summary_dict['a_loss'] = summary_a_loss
        summary_dict['c_loss'] = summary_c_loss

        merged = tf.summary.merge_all()

        log_dir = args.log_dir + args.scenario + '/' + args.exp_name
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        else:
            tf.gfile.DeleteRecursively(log_dir)
            
        summary_writer = tf.summary.FileWriter(log_dir)

    if args.restore or is_evaluate:
        load_dir = os.path.join(args.load_dir, args.scenario)
        for i in range(num_agents):
            ppo[i].load(load_dir, epoch=args.load_epoch)

    # ======================================== main loop ======================================== #
    num_episodes = args.episodes
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
    episode_r_all_sum = []
    for ep in range(0, num_episodes):
        dataset.clear()
        a_loss = [[] for _ in range(num_agents)]
        c_loss = [[] for _ in range(num_agents)]
        obs_n = env.reset()
        episode_r_n = [0. for _ in range(num_agents)]

        for step in range(0, args.max_episode_len):
            if is_render:
                # time.sleep(0.1)
                env.render(mode=None)
            act_n = [None for _ in range(num_agents)]
            value_n = [None for _ in range(num_agents)]
            for i in range(num_agents):
                act_n[i], value_n[i] = ppo[i].act(obs_n[i])
            #print(act_n)
            #print(value_n)
            next_obs_n, reward_n, done_n, info_n = env.step(act_n)
            done = all(done_n)
            if not is_evaluate:  # trigger for data collection
                dataset.push(obs_n, act_n, reward_n, value_n)

            obs_n = next_obs_n

            episode_r_n = list(map(operator.add, episode_r_n, reward_n))
            total_step += 1

            if done:
                break

        # =============================== render / record / model saving ===============================
        # every episode
        if not is_evaluate:
            dataset.compute(args.gamma)
            for epoch in range(args.epoch):
                batch_obs_n, batch_act_n, batch_reward_n, batch_values_n, batch_next_values_n, batch_gaes_n = dataset.sample()
            for i in range(num_agents):
                t_info_n = ppo[i].train(batch_obs_n[i], batch_act_n[i], batch_reward_n[i], batch_next_values_n[i], batch_gaes_n[i])
                # print('step: {}, {}'.format(step, t_info_n))

                if t_info_n is not None:
                    a_loss[i].append(t_info_n['a_loss'])
                    c_loss[i].append(t_info_n['c_loss'])

        # episode_r_n = [round(_, 3) for _ in episode_r_n]
        episode_r_all.append(episode_r_n)
        episode_r_all_sum.append(np.sum(episode_r_n))

        # if t_info_n is not None:
        # if not is_evaluate:
        if (not is_evaluate) and (t_info_n is not None):
            a_loss = list(map(lambda x: round(sum(x) / len(x), 3), a_loss))
            c_loss = list(map(lambda x: round(sum(x) / len(x), 3), c_loss))
            feed_dict = dict()
            feed_dict.update(zip(summary_dict['agent_reward'], episode_r_n))
            feed_dict.update(zip(summary_dict['a_loss'], a_loss))
            feed_dict.update(zip(summary_dict['c_loss'], c_loss))
            feed_dict.update({summary_dict['sum_reward']: episode_r_all_sum[-1]})

            summary = sess.run(merged, feed_dict=feed_dict)
            summary_writer.add_summary(summary, ep)

            if (ep+1) % args.save_interval == 0:
                for i in range(num_agents):
                    ppo[i].save(args.save_dir+args.scenario, ep+1, args.max_to_keep)
                print("\n--- episode-{} | [a-loss]: {} | [c-loss]: {} | [mean-sum-reward]: {} | [inter-time]: {}".format(ep+1, a_loss, c_loss, np.mean(episode_r_all_sum[-args.save_interval:], axis=0), round(time.time()-t_start),4))
                t_start = time.time()

        if is_evaluate:
            print("\n--- episode-{} | [sum-reward]: {}".format(ep+1, episode_r_all_sum[-1]))
    if is_evaluate:       
        print("mean sum reward in {} episodes: {}".format(num_episodes, np.mean(episode_r_all_sum)))       
    env.close()
    if not is_evaluate:
        summary_writer.close()
    sess.close()
