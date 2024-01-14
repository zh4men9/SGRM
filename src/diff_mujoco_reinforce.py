#!/usr/bin/env python3

import argparse
import datetime
import os
import pprint
import random
import numpy as np
import torch
from mujoco_env import make_mujoco_env
from torch import nn
from torch.distributions import Independent, Normal
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.policy import PGPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb

from utils.DiffNet import DiffNet
from utils.DiffCollector import DiffCollector
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Ant-v3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--buffer-size", type=int, default=4096)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[64, 64])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epoch", type=int, default=34)
    parser.add_argument("--step-per-epoch", type=int, default=30000)
    parser.add_argument("--step-per-collect", type=int, default=2048)
    parser.add_argument("--repeat-per-collect", type=int, default=1)
    # batch-size >> step-per-collect means calculating all data in one singe forward.
    parser.add_argument("--batch-size", type=int, default=99999)
    parser.add_argument("--training-num", type=int, default=64)
    parser.add_argument("--test-num", type=int, default=10)
    # reinforce special
    parser.add_argument("--rew-norm", type=int, default=True)
    # "clip" option also works well.
    parser.add_argument("--action-bound-method", type=str, default="tanh")
    parser.add_argument("--lr-decay", type=int, default=True)
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


def test_reinforce(args=get_args()):
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
        args.task, args.seed, args.training_num, args.test_num, obs_norm=True
    )
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
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
                    activation=nn.Tanh,
                    device=args.device)
    actor = ActorProb(
        net_a,
        args.action_shape,
        max_action=args.max_action,
        unbounded=True,
        device=args.device,
    ).to(args.device)
    torch.nn.init.constant_(actor.sigma_param, -0.5)
    for m in actor.modules():
        if isinstance(m, torch.nn.Linear):
            # orthogonal initialization
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)
    # do last policy layer scaling, this will make initial actions have (close to)
    # 0 mean and std, and will help boost performances,
    # see https://arxiv.org/abs/2006.05990, Fig.24 for details
    for m in actor.mu.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.zeros_(m.bias)
            m.weight.data.copy_(0.01 * m.weight.data)

    optim = torch.optim.Adam(actor.parameters(), lr=args.lr)
    lr_scheduler = None
    if args.lr_decay:
        # decay learning rate to 0 linearly
        max_update_num = np.ceil(
            args.step_per_epoch / args.step_per_collect
        ) * args.epoch

        lr_scheduler = LambdaLR(
            optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num
        )

    def dist(*logits):
        return Independent(Normal(*logits), 1)

    policy = PGPolicy(
        actor,
        optim,
        dist,
        discount_factor=args.gamma,
        reward_normalization=args.rew_norm,
        action_scaling=True,
        action_bound_method=args.action_bound_method,
        lr_scheduler=lr_scheduler,
        action_space=env.action_space,
    )

    # load a previous policy
    if args.resume_path:
        ckpt = torch.load(args.resume_path, map_location=args.device)
        policy.load_state_dict(ckpt["model"])
        train_envs.set_obs_rms(ckpt["obs_rms"])
        test_envs.set_obs_rms(ckpt["obs_rms"])
        print("Loaded agent from: ", args.resume_path)

    # collector
    if args.training_num > 1:
        buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
    else:
        buffer = ReplayBuffer(args.buffer_size)
    train_collector = DiffCollector(policy, train_envs, buffer, exploration_noise=True, seed=args.seed)
    test_collector = DiffCollector(policy, test_envs, seed=args.seed)

    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    args.algo_name = "reinforce"
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
        state = {"model": policy.state_dict(), "obs_rms": train_envs.get_obs_rms()}
        torch.save(state, os.path.join(log_path, "policy.pth"))

    if not args.watch:
        # trainer
        result = onpolicy_trainer(
            policy,
            train_collector,
            test_collector,
            args.epoch,
            args.step_per_epoch,
            args.repeat_per_collect,
            args.test_num,
            args.batch_size,
            step_per_collect=args.step_per_collect,
            save_best_fn=save_best_fn,
            logger=logger,
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
    test_reinforce()
