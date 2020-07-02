from __future__ import print_function

import argparse
import os

import torch
import torch.multiprocessing as mp

import my_optim
from envs import create_atari_env
from model import ActorCritic
from test import test
from train import train

parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--gae-lambda', type=float, default=1.00,                    # used in gae calculation
                    help='lambda parameter for GAE (default: 1.00)')
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help='value loss coefficient (default: 0.5)')
parser.add_argument('--max-grad-norm', type=float, default=50,                    #used for gradient clipping
                    help='value loss coefficient (default: 50)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--num-processes', type=int, default=10,
                    help='how many training processes to use (default: 4)')
parser.add_argument('--num-steps', type=int, default=20,
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--max-episode-length', type=int, default=1000000,
                    help='maximum length of an episode (default: 1000000)')
parser.add_argument('--env-name', default='SpaceInvadersDeterministic-v4',
                    help='environment to train on (default: BreakoutDeterministic-v4)')
parser.add_argument('--no-shared', default=False,
                    help='use an optimizer without shared momentum.')
parser.add_argument('--test_mode', default=True,
                    help='Keep it true for infrence')


if __name__ == '__main__':
    mp.set_start_method("spawn")
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    #parse the arguments
    args = parser.parse_args()

    #set the seed of random no generator (default=1)
    torch.manual_seed(args.seed)
    env = create_atari_env(args.env_name)
    if not args.test_mode:
        #create instance of the model
        shared_model = ActorCritic(env.observation_space.shape[0], env.action_space)
        shared_model.share_memory()

        if args.no_shared:
            optimizer = None
        else:
            optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=args.lr)
            optimizer.share_memory()
    else:
        shared_model= torch.load('a3c.pkl')

    processes = []
    epoch_reward=[]

    #i indicates signed integer 
    #These shared objects will be process and thread-safe.
    #counter will remain in shared memory
    counter = mp.Value('i', 0)
    #one can use a lock to ensure that only one process prints to standard output at a time
    #Without using the lock output from the different processes is liable to get all mixed up.
    lock = mp.Lock()

    p = mp.Process(target=test, args=(args.num_processes, args, shared_model, counter,epoch_reward))
    p.start()
    processes.append(p)
    if not args.test_mode:
        for rank in range(0, args.num_processes):
            p = mp.Process(target=train, args=(rank, args, shared_model, counter, lock, optimizer))
            p.start()
            processes.append(p)
    for p in processes:
        #Without the join(), the main process can complete before the child process does
        p.join()
