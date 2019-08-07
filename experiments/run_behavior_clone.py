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
from algo.behavior_clone import MABehavioralCloning

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Behavioral cloning experiments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--iterations", type=int, default=1000, help="number of iterations")
    # parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument("--num_agents", type=int, default=2, help="number of agents")
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

    scenario = scenarios.load(args.scenario+".py").Scenario()
    world = scenario.make_world()

    env = multiagent.environment.MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)

    # =========================== initialize model and summary =========================== #
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    num_agents = min(args.num_agents, env.n)
    bc = MABehavioralCloning(sess, env, args.exp_name, num_agents, args.batch_size, args.lr)
    dataset = Dataset(num_agents, args.batch_size)

    if not is_evaluate:
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

        bc.init()  # run self.sess.run(tf.global_variables_initializer()) and hard update

    if args.restore or is_evaluate:
        load_dir = args.load_dir + args.scenario
        bc.load(load_dir, epoch=args.load_epoch)

    # ======================================== main loop ======================================== #
    batch_size = args.batch_size
    t_start = time.time()
    total_step = 0
    epochs = len(act_n_E[0]) // batch_size 
    if not is_evaluate:
        for iteration in range(args.iterations):  # episode
            loss = [[] for _ in range(num_agents)]
            # shuffle dataset
            dataset.shuffule()
            # train
            for epoch in range(epochs):
                # select sampls
                batch_obs_n, batch_act_n = dataset.next()

                info_n = bc.train(obs=batch_obs_n, act=batch_act_n)
                loss = map(lambda x, y: y + [x], info_n['loss'], loss)

            if (iteration+1) % args.save_interval == 0:
                bc.save(args.save_dir+args.scenario, ep, args.max_to_keep)
                print("\n---- epoch: {} | [loss]: {} | [inter-time]: {}".format(ep, loss, round(time.time()-t_start),4))
                t_start = time.time()

            summary_writer.add_summary(summary, iteration)

    num_episodes = 10
    max_episode_len = 25
    if is_evaluate:
        for ep in range(0, num_episodes):
            obs_n = env.reset()
            episode_r_n = [0. for _ in range(num_agents)]
            for step in range(0, max_episode_len):
                env.render(mode=None)
                act_n = bc.act(obs_n)
                next_obs_n, reward_n, done_n, info_n = env.step(act_n)
                done = all(done_n)
                obs_n = next_obs_n
                episode_r_n = list(map(operator.add, episode_r_n, reward_n))
            print("\n--- episode-{} | [reward]: {}".format(ep, reward))


    summary_writer.close()
    sess.close()

