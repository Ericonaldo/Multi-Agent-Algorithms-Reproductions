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

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Behavioral cloning experiments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max_episode_len", type=int, default=40, help="maximum episode length")
    parser.add_argument("--train_episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--num_agents", type=int, default=2, help="number of agents")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch_size", type=int, default=64, help="number of episodes to optimize at the same time")
    # Checkpointing & Logging
    parser.add_argument("--exp_name", type=str, default="behavior_clone", help="name of the experiment")
    parser.add_argument("--save_interval", type=int, default=400, help='Interval episode for saving model(default=400)')
    parser.add_argument("--save_dir", type=str, default="./trained_models/", help="Parent directory in which trained models should be saved")
    parser.add_argument("--log_dir", type=str, default="./logs/", help="directory of logging")
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--load_dir", type=str, default="./trained_models/", help="Parent directory in which trained models are loaded")
    parser.add_argument("--load_epoch", type=int, default=1000, help="the epoch of loaded model")
    parser.add_argument("--max_to_keep", type=int, defaut=10, help="number of models to save")
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
    maiail = MAIAIL(sess, env, args.exp_name, num_agents, args.batch_size, args.lr, args.gamma)

    if not is_evaluate:
        # initialize summary
        summary_d_loss = [None for _ in range(num_agents)]
        summary_p_loss = [None for _ in range(num_agents)]
        for i in range(num_agents):
            summary_d_loss[i] = tf.placeholder(tf.float32, None)
            summary_p_loss[i] = tf.placeholder(tf.float32, None)

            tf.summary.scalar('Discriminator-Loss-{}'.format(i), summary_d_loss[i])
            tf.summary.scalar('Policy-Loss-{}'.format(i), summary_p_loss[i])

        summary_dict['d_loss'] = summary_d_loss
        summary_dict['p_loss'] = summary_p_loss

        merged = tf.summary.merge_all()

        log_dir = args.log_dir + args.scenario
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        else:
            tf.gfile.DeleteRecursively(log_dir)
            
        summary_writer = tf.summary.FileWriter(log_dir)

        maiail.init()  # run self.sess.run(tf.global_variables_initializer()) and hard update

    if args.restore or is_evaluate:
        load_dir = args.load_dir + args.scenario
        bc.load(load_dir, epoch=args.load_epoch)

    # ======================================== main loop ======================================== #
    obs_n_E = np.genfromtxt('interactions/observations.csv') # [len, n]
    act_n_E = np.genfromtxt('interactions/actions.csv', dtype=np.int32) # [len, n]

    batch_size = args.batch_size
    num_episodes = args.train_episodes
    max_episode_len = args.max_episode_len
    episode_reward_all = []
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
    observations_n = []
    actions_n = []

    for ep in range(0, num_episodes):
        p_loss = [[] for _ in range(num_agents)]
        d_loss = [[] for _ in range(num_agents)]
        obs_n = env.reset()
        episode_r_n = [0. for _ in range(num_agents)]
        run_policy_steps = 0
        for step in range(0, max_episode_len):
            total_step += 1
            run_policy_steps += 1

            if args.render:
                env.render(mode=None)
            act_n = maiail.act(obs_n)
            next_obs_n, reward_n, done_n, info_n = env.step(act_n)
            observations_n.append(obs_n)
            actions_n.append(act_n)
            done = all(done_n) 


            if done:
                continue

            obs_n = next_obs_n
            episode_r_n = list(map(operator.add, episode_r_n, reward_n))
        print("\n--- episode-{} | [reward]: {}".format(ep, reward))
    epochs = len(act_n_E[0]) / batch_size 
    
    if not is_evaluate:
        for iteration in range(args.iterations):  # episode
            loss = [[] for _ in range(num_agents)]
            # shuffle dataset
            indices = list(range(len(act_n_E[0])))
            random.shuffle(indices)
            obs_n = obs_n_E[indices]
            act_n = act_n_E[indices]]
            # train
            for epoch in range(epochs):
                # select sample indices in [low, high)
                batch_start_ind = epoch*batch_size
                batch_end_ind = epoch*batch_size+batch_size-1

                info_n = bc.train(obs=obs_n, act=act_n)
                loss = map(lambda x, y: y + [x], info_n['loss'], loss)

            if (iteration+1) % args.interval == 0:
                bc.save(args.save_dir+args.scenario, ep, args.max_to_keep)
                print("\n--- epoch-{} | [loss]: {} | [inter-time]: {}".format(ep, loss, round(time.time()-t_start),4)))
                t_start = time.time()

            summary_writer.add_summary(summary, iteration)

    env.close()
    summary_writer.close()
    sess.close()

