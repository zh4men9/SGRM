#!/usr/bin/env python3

import argparse
import datetime
import os
import pprint
import random
import numpy as np
import torch
from mujoco_env import make_mujoco_env
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.exploration import GaussianNoise
from tianshou.policy import TD3Policy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import Actor, Critic

from utils.DiffNet import DiffNet
from utils.DiffCollector import DiffCollector

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Ant-v3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--buffer-size", type=int, default=1000000)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[256, 256])
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--exploration-noise", type=float, default=0.1)
    parser.add_argument("--policy-noise", type=float, default=0.2)
    parser.add_argument("--noise-clip", type=float, default=0.5)
    parser.add_argument("--update-actor-freq", type=int, default=2)
    parser.add_argument("--start-timesteps", type=int, default=25000) # 25000
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--step-per-epoch", type=int, default=5000)
    parser.add_argument("--step-per-collect", type=int, default=1)
    parser.add_argument("--update-per-step", type=int, default=1)
    parser.add_argument("--n-step", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--training-num", type=int, default=1)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument(
        "--logger",
        type=str,
        default="wandb",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument("--wandb-project", type=str, default="mujoco.benchmark")
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="watch the play of pre-trained policy only",
    )
    
    # diffnet
    parser.add_argument("--gradient_order", type=int, default=3)
    parser.add_argument("--gradient_mlp_output", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--gradient_mlp_hidden", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--gradient_mlp_output_rate", type=int, default=5)
    
    # deubg
    parser.add_argument("--debug", default=False, action="store_true")
    return parser.parse_args()


def test_td3(args=get_args()):
    
    # set seed
    random.seed(args.seed)
    # os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 打印gradient_order，gradient_mlp_output，gradient_mlp_hidden
    print("gradient_order: ", args.gradient_order)
    print("gradient_mlp_output: ", args.gradient_mlp_output)
    print("gradient_mlp_hidden: ", args.gradient_mlp_hidden)
    
    # MODIFIED
    # args.debug=False
    if args.debug:
        print("debug mode")
        args.logger = "tensorboard"
        test_order = 3
        args.epoch = 1
        args.start_timesteps = 120
        args.step_per_epoch = 10
        # test first 
        if test_order == 1:
            args.gradient_order=1
            args.gradient_mlp_output=[128]
            args.gradient_mlp_hidden=[[256]]
        elif test_order == 2:
            args.gradient_order=2
            args.gradient_mlp_output=[128]
            args.gradient_mlp_hidden=[[256]]
        elif test_order == 3:
            args.gradient_order=3
            args.gradient_mlp_output=[128, 128]
            args.gradient_mlp_hidden=[256, 256]
        elif test_order == 0:
            args.gradient_order=0
            args.gradient_mlp_output=[0]
            args.gradient_mlp_hidden=[0]
                
    env, train_envs, test_envs = make_mujoco_env(
        args.task, args.seed, args.training_num, args.test_num, obs_norm=False
    )
    # seed
    # train_envs.seed(args.seed)
    
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    args.exploration_noise = args.exploration_noise * args.max_action
    args.policy_noise = args.policy_noise * args.max_action
    args.noise_clip = args.noise_clip * args.max_action
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))

    
    # change output_dim
    prod_state_shape = int(np.prod(args.state_shape))
    if (args.gradient_order!=3 and args.gradient_order!=0):
        args.gradient_mlp_output = [max(int(prod_state_shape // args.gradient_mlp_output_rate), 1)]
    elif (args.gradient_order == 3):
        args.gradient_mlp_output = [max(int(prod_state_shape // args.gradient_mlp_output_rate), 1), 
                                    max(int(prod_state_shape // args.gradient_mlp_output_rate), 1)]
    # model
    net_a = DiffNet(gradient_order=args.gradient_order, 
                    gradient_mlp_output=args.gradient_mlp_output, 
                    gradient_mlp_hidden=args.gradient_mlp_hidden, 
                    state_shape=args.state_shape, 
                    hidden_sizes=args.hidden_sizes, 
                    device=args.device)
    actor = Actor( # add a mlp layer after net_a
        net_a, args.action_shape, max_action=args.max_action, device=args.device
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    net_c1 = DiffNet(gradient_order=args.gradient_order, 
                    gradient_mlp_output=args.gradient_mlp_output, 
                    gradient_mlp_hidden=args.gradient_mlp_hidden,
                    state_shape=args.state_shape,
                    action_shape=args.action_shape,
                    hidden_sizes=args.hidden_sizes,
                    concat=True,  # critic use obs_shape+action_shape as input
                    device=args.device,
    )
    net_c2 = DiffNet(gradient_order=args.gradient_order, 
                    gradient_mlp_output=args.gradient_mlp_output, 
                    gradient_mlp_hidden=args.gradient_mlp_hidden,
                    state_shape=args.state_shape,
                    action_shape=args.action_shape,
                    hidden_sizes=args.hidden_sizes,
                    concat=True,  # critic use obs_shape+action_shape as input
                    device=args.device,
    )
    critic1 = Critic(net_c1, device=args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2 = Critic(net_c2, device=args.device).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    policy = TD3Policy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        tau=args.tau,
        gamma=args.gamma,
        exploration_noise=GaussianNoise(sigma=args.exploration_noise),
        policy_noise=args.policy_noise,
        update_actor_freq=args.update_actor_freq,
        noise_clip=args.noise_clip,
        estimation_step=args.n_step,
        action_space=env.action_space,
    )

    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)

    # collector
    if args.training_num > 1:
        buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
    else:
        buffer = ReplayBuffer(args.buffer_size)
    train_collector = DiffCollector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = DiffCollector(policy, test_envs)
    train_collector.collect(n_step=args.start_timesteps, random=True)
    # data collected in start_timesteps
    print("Data collected in start_timesteps")
    print(train_collector.buffer)
    
    
    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    args.algo_name = "td3"
    # gradient information
    gi = str(args.gradient_order)+'_'+str(args.gradient_mlp_output)+'_'+str(args.gradient_mlp_hidden)+'_'+str(args.gradient_mlp_output_rate)
    log_name = os.path.join(args.task, args.algo_name, str(args.seed), gi, now)
    # log_name = log_name.replace(os.path.sep, "__")
    log_name = log_name.replace('[', "_")
    log_name = log_name.replace(']', "_")
    log_name = log_name.replace(',', "_")
    log_path = os.path.join(args.logdir, log_name)

    # logger
    if args.logger == "wandb":
        logger = WandbLogger(
            save_interval=1,
            name=log_name.replace(os.path.sep, "__"),
            run_id=args.resume_id,
            config=args,
            project=args.wandb_project,
        )
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    if args.logger == "tensorboard":
        logger = TensorboardLogger(writer)
    else:  # wandb
        logger.load(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    if not args.watch:
        # trainer
        result = offpolicy_trainer(
            policy,
            train_collector,
            test_collector,
            args.epoch,
            args.step_per_epoch,
            args.step_per_collect,
            args.test_num,
            args.batch_size,
            save_best_fn=save_best_fn,
            logger=logger,
            update_per_step=args.update_per_step,
            test_in_train=False,
        )
        pprint.pprint(result)

    # Let's watch its performance!
    policy.eval()
    test_envs.seed(args.seed)
    test_collector.reset()
    result = test_collector.collect(n_episode=args.test_num, render=args.render)
    print(f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean()}')


if __name__ == "__main__":
    test_td3()
