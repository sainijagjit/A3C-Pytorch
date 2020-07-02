import time
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import time

import torch
import torch.nn.functional as F

from envs import create_atari_env
from model import ActorCritic

def plot(epoch_reward):
    epochs=np.zeros((len(epoch_reward),1))
    reward=np.zeros((len(epoch_reward),1))
    for i in range(len(epoch_reward)):
        epochs[i][0]=epoch_reward[i][0]
        reward[i][0]=epoch_reward[i][1]
    plt.plot(epochs.ravel(),reward.ravel(), color='blue', label='Reward') 
    plt.legend()
    plt.xlabel('Epochs', fontsize=10)
    plt.ylabel('Reward', fontsize=10)
    plt.show()

def test(rank, args, shared_model, counter,epoch_reward):
    torch.manual_seed(args.seed + rank)

    env = create_atari_env(args.env_name)
    env.seed(args.seed + rank)

    model = ActorCritic(env.observation_space.shape[0], env.action_space)

    model.eval()

    state = env.reset()
    state = torch.from_numpy(state)
    reward_sum = 0
    done = True

    start_time = time.time()

    # a quick hack to prevent the agent from stucking
    actions = deque(maxlen=100)
    episode_length = 0
    while True:
        if args.test_mode:
            env.render()
            time.sleep(0.01)
        episode_length += 1
        # Sync with the shared model
        if done:
            model.load_state_dict(shared_model.state_dict())
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
        else:
            cx = cx.detach()
            hx = hx.detach()

        with torch.no_grad():
            value, logit, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))
        prob = F.softmax(logit, dim=-1)
        action = prob.max(1, keepdim=True)[1].numpy()

        state, reward, done, _ = env.step(action[0, 0])
        done = done or episode_length >= args.max_episode_length
        reward_sum += reward

        # a quick hack to prevent the agent from stucking
        actions.append(action[0, 0])
        if actions.count(actions[0]) == actions.maxlen:
            done = True

        if done:
            f=open("rewards.txt", "a")
            f.write(str(reward_sum)+"\n")
            f.close()
            print("Time {}, num steps {}, FPS {:.0f}, episode reward {}, episode length {}".format(time.strftime("%Hh %Mm %Ss",time.gmtime(time.time() - start_time)),counter.value, counter.value / (time.time() - start_time),reward_sum, episode_length))
            ######################################################
            epoch_reward.append([len(epoch_reward),reward_sum])
            if (len(epoch_reward)%50==0 and len(epoch_reward)!=0):
                plot(epoch_reward)
            torch.save(model, f'a3c.pkl')
            print(f'Length => {len(epoch_reward)}')
            #######################################################
            reward_sum = 0
            episode_length = 0
            actions.clear()
            state = env.reset()
            if not args.test_mode:
                time.sleep(60)

        state = torch.from_numpy(state)
        
