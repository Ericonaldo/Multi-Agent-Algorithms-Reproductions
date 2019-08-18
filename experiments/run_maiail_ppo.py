from common import DIMENSIONER, ESTIMATOR
from common.buffer import Dataset, PPODataset
from algo.maiail_ppo import MAIAIL
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
import multiagent
import os
import sys
import time
import argparse
import operator
import numpy as np
import tensorflow as tf

sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(
    sys.path[0], '../ma_env/multiagent-particle-envs'))

# from common.densityEstimator import KDEEstimator, GMMEstimator
# from common.dimensioner import PCADimensioner, FWDimensioner, AEDimensioner

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("MAIAIL experiments")
    # Environment
    parser.add_argument("--scenario", type=str,
                        default="simple", help="name of the scenario script")
    parser.add_argument("--max_episode_len", type=int,
                        default=25, help="maximum episode length")
    parser.add_argument("--discrete", action="store_true",
                        default=False, help="is the action discrete")
    parser.add_argument("--sample_episodes", type=int,
                        default=10, help="number of sampling episodes")
    parser.add_argument("--iterations", type=int, default=100,
                        help="number of training iterations")
    # parser.add_argument("--num_agents", type=int, default=2, help="number of agents")
    # Core training parameters
    parser.add_argument("--bc_init", action="store_true",
                        default=False, help="need to be initialized with bc")
    parser.add_argument("--density_estimator", type=str, default="kde",
                        choices={"kde", "gmm"}, help="the probability density estimator")
    parser.add_argument("--dimensioner", type=str, default="pca", choices={
                        "pca", "ae", "fw"}, help="the tool to decrease the dimension of (s,a) pairs")
    parser.add_argument("--lower_dimension", type=int, default=None,
                        help="the low dimension of the representation of (s,a) pairs")
    parser.add_argument("--bandwidth", type=float, default=0.1,
                        help="the bandwidth of the kde estimator")
    parser.add_argument("--kernel", type=str, default='gaussian',
                        help="the kernal of the kde estimator")
    parser.add_argument("--lbd", type=float, default=1.0,
                        help="hyperparameter of the importance sampling ratio")
    parser.add_argument("--e_w", type=float, default=1.0,
                        help="hyperparameter of the expert d-loss ratio")
    parser.add_argument("--a_w", type=float, default=1.0,
                        help="hyperparameter of the agent d-loss ratio")
    parser.add_argument("--p_step", type=int, default=4,
                        help="training times of policy network")
    parser.add_argument("--d_step", type=int, default=1,
                        help="training times of discriminator")
    parser.add_argument("--units", type=int, default=128,
                        help="the hidden units of the network")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate for Adam optimizer")
    parser.add_argument("--entcoeff", type=float, default=0.001,
                        help="the coefficient of the entropy loss of the discriminators")
    parser.add_argument("--gamma", type=float,
                        default=0.95, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01,
                        help="soft update factor")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="the batch size to optimize at the same time")
    parser.add_argument("--memory_size", type=int, default=10 **
                        4, help="the memory size of replay buffer")
    parser.add_argument("--dataset_size", type=int, default=65536,
                        help="the dataset size for learning discriminators")
    # Checkpointing & Logging
    parser.add_argument("--exp_name", type=str,
                        default="maiail-ppo", help="name of the experiment")
    parser.add_argument("--save_interval", type=int, default=1,
                        help='Interval episode for saving model(default=400)')
    parser.add_argument("--save_dir", type=str, default="./trained_models/",
                        help="Parent directory in which trained models should be saved")
    parser.add_argument("--data_dir", type=str, default="./expert_data/",
                        help="Directory in which expert data is saved")
    parser.add_argument("--log_dir", type=str,
                        default="./logs/", help="directory of logging")
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--load_dir", type=str, default="./trained_models/",
                        help="Parent directory in which trained models are loaded")
    parser.add_argument("--load_epoch", type=int,
                        default=None, help="the epoch of loaded model")
    parser.add_argument("--max_to_keep", type=int, default=0,
                        help="number of models to save")
    # Evaluation
    parser.add_argument("--is_evaluate", action="store_true",
                        default=False, help='is training or evalutaion')
    # parser.add_argument("--eval_interval", type=int, default=200, help='Evaluation interval episode and save model(default=1000)')
    parser.add_argument("--render", action="store_true",
                        default=False, help="do or not render")
    # parser.add_argument("--plots_dir", type=str, default="./plots/", help="directory where plot data is saved")
    args = parser.parse_args()

    print('=== Configuration:\n', args)

    # =========================== initialize environment =========================== #
    is_evaluate = args.is_evaluate
    is_render = args.render or args.is_evaluate

    scenario = scenarios.load(args.scenario + ".py").Scenario()
    world = scenario.make_world()

    env = multiagent.environment.MultiAgentEnv(
        world, scenario.reset_world, scenario.reward, scenario.observation)

    # =========================== initialize model / summary / estimator / dimensioner =========================== #
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    exp_name = "{}-{}-{}-{}-{}".format(
        args.exp_name,
        args.dimensioner,
        args.lower_dimension,
        args.density_estimator,
        args.bandwidth if args.density_estimator == 'kde' else '')

    num_agents = env.n

    estimator = ESTIMATOR[args.density_estimator]
    e_kwargs = {} if args.density_estimator == "gmm" else {
        "kernel": args.kernel, "bandwidth": args.bandwidth}

    # dimensioner = None
    d_kwargs = {'name': args.dimensioner,
                'lower_dimension': args.lower_dimension}
    dimensioner = DIMENSIONER[args.dimensioner]

    dm = [None for _ in range(num_agents)]
    for i in range(num_agents):
        d_kwargs['agent_id'] = i
        d_kwargs['s_dim'] = env.observation_space[i].shape[0]
        d_kwargs['a_dim'] = env.action_space[i].n
        dm[i] = dimensioner(**d_kwargs)
        dm[i].load()

    # learning_dataset = Dataset(args.scenario, num_agents, args.batch_size)
    expert_dataset = Dataset(args.scenario, num_agents,
                             args.batch_size, capacity=args.dataset_size)
    learning_dataset = PPODataset(
        args.exp_name, num_agents, batch_size=args.batch_size)
    expert_dataset.load_data(args.data_dir)
    maiail = MAIAIL(sess, env, args.scenario, exp_name, num_agents, args.batch_size, args.entcoeff, args.lr, args.gamma,
                    args.tau, args.memory_size, args.p_step, args.d_step, args.units, args.lbd, args.lower_dimension, 0.5, args.discrete)

    if not is_evaluate:
        # initialize summary
        summary_d_loss = [None for _ in range(num_agents)]
        summary_de_loss = [None for _ in range(num_agents)]
        summary_da_loss = [None for _ in range(num_agents)]
        # summary_p_loss = [None for _ in range(num_agents)]
        summary_pa_loss = [None for _ in range(num_agents)]
        summary_pc_loss = [None for _ in range(num_agents)]
        summary_reward = [None for _ in range(num_agents)]

        for i in range(num_agents):
            summary_d_loss[i] = tf.placeholder(tf.float32, None)
            summary_de_loss[i] = tf.placeholder(tf.float32, None)
            summary_da_loss[i] = tf.placeholder(tf.float32, None)
            # summary_p_loss[i] = tf.placeholder(tf.float32, None)
            summary_pa_loss[i] = tf.placeholder(tf.float32, None)
            summary_pc_loss[i] = tf.placeholder(tf.float32, None)
            summary_reward[i] = tf.placeholder(tf.float32, None)

            tf.summary.scalar(
                'Discriminator-Loss-{}'.format(i), summary_d_loss[i])
            tf.summary.scalar(
                'Discriminator-Expert-Loss-{}'.format(i), summary_de_loss[i])
            tf.summary.scalar(
                'Discriminator-Agent-Loss-{}'.format(i), summary_da_loss[i])
            # tf.summary.scalar('Policy-Loss-{}'.format(i), summary_p_loss[i])
            tf.summary.scalar(
                'Policy-Actor-Loss-{}'.format(i), summary_pa_loss[i])
            tf.summary.scalar(
                'Policy-Critic-Loss-{}'.format(i), summary_pc_loss[i])
            tf.summary.scalar('Reward-{}'.format(i), summary_reward[i])

        summary_dict = dict()
        summary_dict['d_loss'] = summary_d_loss
        summary_dict['de_loss'] = summary_de_loss
        summary_dict['da_loss'] = summary_da_loss
        # summary_dict['p_loss'] = summary_p_loss
        summary_dict['pa_loss'] = summary_pa_loss
        summary_dict['pc_loss'] = summary_pc_loss
        summary_dict['reward'] = summary_reward

        merged = tf.summary.merge_all()

        log_dir = args.log_dir + args.scenario + '/' + exp_name
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
        if args.bc_init:
            maiail.bc_init(100, expert_dataset)
    else:
        print("Start Evaluating...")

    t_start = time.time()
    start_time = time.time()
    total_step = 0

    if not is_evaluate:
        # get the probability estimators
        expert_pdf = [None for _ in range(num_agents)]
        for i in range(num_agents):
            e_kwargs['name'] = 'expert_' + args.density_estimator
            e_kwargs['agent_id'] = i
            expert_pdf[i] = estimator(**e_kwargs)
            x = dm[i].transform(expert_dataset.observations[i],
                                expert_dataset.actions[i])
            expert_pdf[i].fit(x)
        print("fitted the pdf of expert (s,a) pair for each agent")

        agent_pdf = [None for _ in range(num_agents)]
        for i in range(num_agents):
            e_kwargs['name'] = 'expert'
            e_kwargs['agent_id'] = i
            agent_pdf[i] = estimator(**e_kwargs)

    iterations = args.iterations
    num_episodes = args.sample_episodes

    if is_evaluate:
        iterations = 1
        num_episodes = 100

    for iteration in range(iterations):
        # sample interations
        print("\n-----------------------------------------------------------------")
        print("sample interactions...")

        all_episode_r_n = []
        episode_r_all_sum = []
        learning_dataset.clear()

        for ep in range(num_episodes):
            obs_n = env.reset()
            episode_r_n = [0. for _ in range(num_agents)]
            run_policy_steps = 0

            for step in range(args.max_episode_len):
                total_step += 1
                run_policy_steps += 1

                if args.render:
                    env.render(mode=None)

                act_n, values_n = maiail.act(obs_n)
                next_obs_n, reward_n, done_n, info_n = env.step(act_n)

                learning_dataset.push(obs_n, act_n, reward_n, values_n)

                done = all(done_n)

                if done:
                    continue

                obs_n = next_obs_n
                episode_r_n = list(map(operator.add, episode_r_n, reward_n))

            episode_r_all_sum.append(np.sum(episode_r_n))
            all_episode_r_n.append(episode_r_n)

        all_episode_r_n = np.mean(all_episode_r_n, axis=0)
        episode_r_all_sum = np.mean(episode_r_all_sum)

        print("--- iteration-{} | [mean-sample-agent-reward]: {} | [mean-sample-sum-reward]: {}".format(
            iteration, all_episode_r_n, episode_r_all_sum))

        # ======================================== training  ======================================== #
        if not is_evaluate:
            # update the probability estimators of agent data
            for i in range(num_agents):
                x = dm[i].transform(
                    learning_dataset.observations[i], learning_dataset.actions[i])
                agent_pdf[i].fit(x)

            print("updated the pdf of agent (s,a) pair for each agent")

            feed_dict = dict()
            feed_dict.update(zip(summary_dict['reward'], all_episode_r_n))
            # p_loss = [[] for _ in range(num_agents)]
            # d_loss = [[] for _ in range(num_agents)]
            # a_loss, c_loss = [[] for _ in range(num_agents)], [[] for _ in range(num_agents)]

            print("training models...")
            # alpha_value = [(iteration+1)*1.0/(iteration+25+1)] * num_agents
            # alpha_value = [1] * num_agents
            # TODO dimensioner not implement
            t_info = maiail.train(
                expert_dataset, learning_dataset, expert_pdf, agent_pdf, dm)

            pa_loss, pc_loss, d_loss, de_loss, da_loss = t_info['pa_loss'], t_info[
                'pc_loss'], t_info['d_loss'], t_info['de_loss'], t_info['da_loss']

            # feed_dict.update(zip(summary_dict['p_loss'], p_loss))
            feed_dict.update(zip(summary_dict['pa_loss'], pa_loss))
            feed_dict.update(zip(summary_dict['pc_loss'], pc_loss))
            feed_dict.update(zip(summary_dict['d_loss'], d_loss))
            feed_dict.update(zip(summary_dict['de_loss'], de_loss))
            feed_dict.update(zip(summary_dict['da_loss'], da_loss))

            summary = sess.run(merged, feed_dict=feed_dict)
            summary_writer.add_summary(summary, iteration)

            # print("-----------------------------------------------------------------")
            print("----[iteration]: {} | [pa-loss]: {} | [pc-loss]: {} | [d-loss]: {} | [de-loss]: {}| [da-loss]: {}  | [inter-time]: {}".format(
                iteration, pa_loss, pc_loss, d_loss, de_loss, da_loss, round(time.time() - t_start), 4))
            # print("-----------------------------------------------------------------\n")
            t_start = time.time()

            if (iteration + 1) % args.save_interval == 0:
                maiail.save(args.save_dir + args.scenario,
                            iteration, args.max_to_keep)

    if is_evaluate:
        print("mean sum reward in {} episodes: {}".format(
            num_episodes, episode_r_all_sum))

    env.close()
    if not is_evaluate:
        summary_writer.close()
    sess.close()
