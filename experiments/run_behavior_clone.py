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
from common.utils import Dataset
from algo.behavior_clone import MABehavioralCloning

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Behavioral cloning experiments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--iterations", type=int, default=1000, help="number of iterations")
    # parser.add_argument('--epochs', default=10, type=int)
    # parser.add_argument("--num_agents", type=int, default=2, help="number of agents")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    # parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch_size", type=int, default=256, help="the batch size to optimize at the same time")
    # Checkpointing & Logging
    parser.add_argument("--exp_name", type=str, default="behavior_clone", help="name of the experiment")
    parser.add_argument("--save_interval", type=int, default=400, help='Interval episode for saving model(default=400)')
    parser.add_argument("--save_dir", type=str, default="./trained_models/", help="directory in which training state and model should be saved")
    parser.add_argument("--log_dir", type=str, default="./logs/", help="directory of logging")
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--load_dir", type=str, default="./trained_models/", help="directory in which models are loaded")
    parser.add_argument("--data_dir", type=str, default="./expert_data/", help="parent directory in which data is loaded")
    parser.add_argument("--load_iteration", type=int, default=None, help="the iteration of loaded model")
    parser.add_argument("--max_to_keep", type=int, default=10, help="number of models to save")
    # Evaluation
    parser.add_argument("--is_evaluate", action="store_true",default=False, help='is training or evalutaion')
    # parser.add_argument("--eval_interval", type=int, default=200, help='Evaluation interval episode and save model(default=1000)')
    parser.add_argument("--render", action="store_true", default=False, help="do or not render")
    # parser.add_argument("--plots_dir", type=str, default="./plots/", help="directory where plot data is saved")
    args = parser.parse_args()

    print('=== Configuration:\n', args)

    is_evaluate = args.is_evaluate
    is_render = args.render or args.is_evaluate

    # =========================== initialize environment =========================== #

    scenario = scenarios.load(args.scenario+".py").Scenario()
    world = scenario.make_world()

    env = multiagent.environment.MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)

    # =========================== initialize model and summary =========================== #
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # num_agents = min(args.num_agents, env.n)
    num_agents = env.n
    bc = MABehavioralCloning(sess, env, args.exp_name, num_agents, args.batch_size, args.lr)
    dataset = Dataset(args.scenario, num_agents, args.batch_size)
    dataset.load_data(args.data_dir)
    if len(dataset) < args.batch_size:
        print("the size of data is less than the batch_size!")
        exit(0)

    if not is_evaluate:
        summary_dict = dict()
        # initialize summary
        summary_loss = [None for _ in range(num_agents)]
        for i in range(num_agents):
            summary_loss[i] = tf.placeholder(tf.float32, None)

            tf.summary.scalar('Loss-{}'.format(i), summary_loss[i])

        summary_dict['loss'] = summary_loss

        merged = tf.summary.merge_all()

        log_dir = args.log_dir + args.scenario
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        else:
            tf.gfile.DeleteRecursively(log_dir)
            
        summary_writer = tf.summary.FileWriter(log_dir)

        bc.init()  # run self.sess.run(tf.global_variables_initializer()) 

    if args.restore or is_evaluate:
        load_dir = os.path.join(args.load_dir, args.scenario)
        bc.load(load_dir, epoch=args.load_iteration)

    # ======================================== main loop ======================================== #
    batch_size = args.batch_size
    t_start = time.time()
    total_step = 0
    epochs = len(dataset) // batch_size 
    if not is_evaluate:
        for iteration in range(args.iterations):  # episode
            loss = [[] for _ in range(num_agents)]
            # shuffle dataset
            dataset.shuffle()
            # train
            for epoch in range(epochs):
                # select sampls
                batch_obs_n, batch_act_n = dataset.next()

                info_n = bc.train(obs=batch_obs_n, tar_act=batch_act_n)
                loss = map(lambda x, y: y + [x], info_n['loss'], loss)

            loss = list(map(lambda x: round(sum(x) / len(x), 3), loss))
            feed_dict = dict()
            feed_dict.update(zip(summary_dict['loss'], loss))
            summary = sess.run(merged, feed_dict=feed_dict)
            summary_writer.add_summary(summary, iteration)

            if (iteration+1) % args.save_interval == 0:
                bc.save(args.save_dir+args.scenario, iteration+1, args.max_to_keep)
                print("\n---- iteration: {} | [loss]: {} | [inter-time]: {}".format(iteration+1, loss, round(time.time()-t_start),4))
                t_start = time.time()

    # =========================== start evaluating =========================== #

    num_episodes = 10
    max_episode_len = 25
    for ep in range(0, num_episodes):
        obs_n = env.reset()
        episode_r_n = [0. for _ in range(num_agents)]
        for step in range(0, max_episode_len):
            if is_render:
                env.render(mode=None)
            act_n = bc.act(obs_n)
            next_obs_n, reward_n, done_n, info_n = env.step(act_n)
            done = all(done_n)
            obs_n = next_obs_n
            episode_r_n = list(map(operator.add, episode_r_n, reward_n))
        print("\n--- episode-{} | | [reward]: {} | [sum-reward]: {}".format(ep, episode_r_n, np.sum(episode_r_n)))

    env.close()
    summary_writer.close()
    sess.close()

