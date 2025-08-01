import numpy as np
import torch
import argparse
import os
import math
import gym
import sys
import random
import time
import json
import dmc2gym
import copy
from wrappers import FixedLengthEpisodeWrapper, PixelRenormalization
import utils
from logger import Logger
from video import VideoRecorder

from curl_sac import CurlSacAgent
from torchvision import transforms


from distracting_control import suite

import wandb


def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--image_size_distracting_env', default=100, type=int)
    parser.add_argument('--static_camera',action='store_true')
    parser.add_argument('--env_name',type=str,default='Cheetah-run-easy-v1')
    parser.add_argument('--static_distractors',action='store_true')
    parser.add_argument('--domain_name', default='cheetah')
    parser.add_argument('--task_name', default='run')
    parser.add_argument('--pre_transform_image_size', default=100, type=int)

    parser.add_argument('--image_size', default=84, type=int)
    parser.add_argument('--action_repeat', default=4, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    # train
    parser.add_argument('--agent', default='curl_sac', type=str)
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--num_train_steps', default=1000000, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    # eval
    parser.add_argument('--eval_freq', default=1000, type=int)
    parser.add_argument('--num_eval_episodes', default=10, type=int)
    # critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float) # try 0.05 or 0.1
    parser.add_argument('--critic_target_update_freq', default=2, type=int) # try to change it to 1 and retain 0.01 above
    # actor
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    # encoder
    parser.add_argument('--encoder_type', default='pixel', type=str)
    parser.add_argument('--encoder_feature_dim', default=50, type=int)
    parser.add_argument('--encoder_lr', default=1e-3, type=float)
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    parser.add_argument('--curl_latent_dim', default=128, type=int)
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    parser.add_argument('--alpha_beta', default=0.5, type=float)
    # misc
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--save_video', default=False, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--detach_encoder', default=False, action='store_true')
    parser.add_argument('--n_final_eval_eps',type=int,default=100)

    parser.add_argument('--log_interval', default=100, type=int)
    args = parser.parse_args()
    args.ep_len = 1000/args.action_repeat
    return args


def evaluate(env, agent, video, num_episodes, L, step, args, final_eval = False):
    all_ep_rewards = []

    def run_eval_loop(sample_stochastically=True, final_eval= False):
        if not final_eval:

            start_time = time.time()
            prefix = 'stochastic_' if sample_stochastically else ''
            for i in range(num_episodes):
                obs = env.reset()
                video.init(enabled=(i == 0))
                done = False
                episode_reward = 0
                while not done:
                    # center crop image
                    if args.encoder_type == 'pixel':
                        obs = utils.center_crop_image(obs,args.image_size)
                    with utils.eval_mode(agent):
                        if sample_stochastically:
                            action = agent.sample_action(obs)
                        else:
                            action = agent.select_action(obs)
                    obs, reward, done, _ = env.step(action)
                    video.record(env)
                    episode_reward += reward

                video.save('%d.mp4' % step)
                L.log('eval/' + prefix + 'episode_reward', episode_reward, step)
                wandb.log({'eval/' + prefix + 'episode_reward': episode_reward}, step=step)
                all_ep_rewards.append(episode_reward)
            
            L.log('eval/' + prefix + 'eval_time', time.time()-start_time , step)
            wandb.log({'eval/' + prefix + 'eval_time': time.time()-start_time}, step=step)
            mean_ep_reward = np.mean(all_ep_rewards)
            best_ep_reward = np.max(all_ep_rewards)
            L.log('eval/' + prefix + 'mean_episode_reward', mean_ep_reward, step)
            wandb.log({'eval/' + prefix + 'mean_episode_reward': mean_ep_reward}, step=step)
            L.log('eval/' + prefix + 'best_episode_reward', best_ep_reward, step)
            wandb.log({'eval/' + prefix + 'best_episode_reward': best_ep_reward}, step=step)
        else:
            start_time = time.time()
            prefix = 'stochastic_' if sample_stochastically else ''
            for i in range(num_episodes):
                obs = env.reset()
                video.init(enabled=(i == 0))
                done = False
                episode_reward = 0
                while not done:
                    # center crop image
                    if args.encoder_type == 'pixel':
                        obs = utils.center_crop_image(obs,args.image_size)
                    with utils.eval_mode(agent):
                        if sample_stochastically:
                            action = agent.sample_action(obs)
                        else:
                            action = agent.select_action(obs)
                    obs, reward, done, _ = env.step(action)
                    video.record(env)
                    episode_reward += reward

                video.save('%d.mp4' % step)
                L.log('Final eval/' + prefix + 'episode_reward', episode_reward, step)
                wandb.log({'Final eval/' + prefix + 'episode_reward': episode_reward}, step=step)
                all_ep_rewards.append(episode_reward)
            
            L.log('Final eval/' + prefix + 'eval_time', time.time()-start_time , step)
            wandb.log({'Final eval/' + prefix + 'eval_time': time.time()-start_time}, step=step)
            mean_ep_reward = np.mean(all_ep_rewards)
            best_ep_reward = np.max(all_ep_rewards)
            L.log('Final eval/' + prefix + 'mean_episode_reward', mean_ep_reward, step)
            wandb.log({'Final eval/' + prefix + 'mean_episode_reward': mean_ep_reward}, step=step)
            L.log('Final eval/' + prefix + 'best_episode_reward', best_ep_reward, step)
            wandb.log({'Final eval/' + prefix + 'best_episode_reward': best_ep_reward}, step=step)

    run_eval_loop(sample_stochastically=False, final_eval= final_eval)
    L.dump(step)


def make_agent(obs_shape, action_shape, args, device):
    if args.agent == 'curl_sac':
        return CurlSacAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            log_interval=args.log_interval,
            detach_encoder=args.detach_encoder,
            curl_latent_dim=args.curl_latent_dim

        )
    else:
        assert 'agent is not supported: %s' % args.agent

def main():
    args = parse_args()
    if args.seed == -1: 
        args.__dict__["seed"] = np.random.randint(1,1000000)
    utils.set_seed_everywhere(args.seed)
    print(args.encoder_type)

    wandb.login(key="")
    wandb.init(project='CURL DCS', config=vars(args), name=f"{args.domain_name}-{args.task_name}-seed{args.seed}")
    # env = dmc2gym.make(
    #     domain_name=args.domain_name,
    #     task_name=args.task_name,
    #     seed=args.seed,
    #     visualize_reward=False,
    #     from_pixels=(args.encoder_type == 'pixel'),
    #     height=args.pre_transform_image_size,
    #     width=args.pre_transform_image_size,
    #     frame_skip=args.action_repeat
    # )
    # env.max_episode_steps = args.ep_len



    def make_distracting_env(args,val=False):
	# ipdb.set_trace()
        davis_path = '/home/biorobotics/Downloads/DAVIS/JPEGImages/480p/'
        suite.BG_DATA_PATH=davis_path 
        dynamic = not args.static_distractors
        
        partition='val' if val else 'train'

        if args.static_camera:
            env = gym.make(args.env_name,from_pixels=True,dynamic=dynamic,visualize_reward=False,height=args.image_size_distracting_env,width=args.image_size_distracting_env,frame_skip=args.action_repeat,distraction_types=('background','color'),background_dataset_videos=partition)
        else:
            # ipdb.set_trace()
            env = gym.make(args.env_name,from_pixels=True,dynamic=dynamic,visualize_reward=False,height=args.image_size_distracting_env,width=args.image_size_distracting_env,frame_skip=args.action_repeat,background_dataset_videos=partition)
        # if hasattr(env, 'observation_space') and env.observation_space.dtype == np.uint8:
        #     env.observation_space = gym.spaces.Box(
        #         low=0,
        #         high=255,
        #         shape=env.observation_space.shape,
        #         dtype=np.uint8
        #     )
        env.max_episode_steps = args.ep_len


        env = FixedLengthEpisodeWrapper(env,args.ep_len)
        env = PixelRenormalization(env)

        return env

    env = make_distracting_env(args)
 
    # env.seed(args.seed)

    # stack several consecutive frames together
    if args.encoder_type == 'pixel':
        env = utils.FrameStack(env, k=args.frame_stack)
    
    # make directory
    ts = time.gmtime() 
    ts = time.strftime("%m-%d", ts)    
    env_name = args.domain_name + '-' + args.task_name
    exp_name = env_name + '-' + ts + '-im' + str(args.image_size) +'-b'  \
    + str(args.batch_size) + '-s' + str(args.seed)  + '-' + args.encoder_type
    args.work_dir = args.work_dir + '/'  + exp_name

    utils.make_dir(args.work_dir)
    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    buffer_dir = utils.make_dir(os.path.join(args.work_dir, 'buffer'))

    video = VideoRecorder(video_dir if args.save_video else None)

    # with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
    #     json.dump(vars(args), f, sort_keys=True, indent=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Checking if GPU is being used {device}")

    action_shape = env.action_space.shape

    if args.encoder_type == 'pixel':
        obs_shape = (3*args.frame_stack, args.image_size, args.image_size)
        pre_aug_obs_shape = (3*args.frame_stack,args.pre_transform_image_size,args.pre_transform_image_size)
    else:
        obs_shape = env.observation_space.shape
        pre_aug_obs_shape = obs_shape


    # print(f'obs_shape is {(pre_aug_obs_shape)}')
    replay_buffer = utils.ReplayBuffer(
        obs_shape=pre_aug_obs_shape,
        action_shape=action_shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device,
        image_size=args.image_size,
    )

    agent = make_agent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        args=args,
        device=device
    )

    L = Logger(args.work_dir, use_tb=args.save_tb)

    episode, episode_reward, done = 0, 0, True
    start_time = time.time()

    for step in range(args.num_train_steps):
        # evaluate agent periodically

        if step % args.eval_freq == 0:
            L.log('eval/episode', episode, step)
            wandb.log({'eval/episode': episode}, step=step)
            evaluate(env, agent, video, args.num_eval_episodes, L, step,args)
            if args.save_model:
                agent.save_curl(model_dir, step)
            if args.save_buffer:
                replay_buffer.save(buffer_dir)

        if done:
            if step > 0:
                if step % args.log_interval == 0:
                    L.log('train/duration', time.time() - start_time, step)
                    L.dump(step)
                start_time = time.time()
            if step % args.log_interval == 0:
                L.log('train/episode_reward', episode_reward, step)
                wandb.log({'train/episode_reward': episode_reward}, step=step)

            obs = env.reset()
            # # print(f"Printing observation{obs}")
            # # print(f"Printing observation shape {obs.shape}")
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1
            if step % args.log_interval == 0:
                L.log('train/episode', episode, step)
                wandb.log({'train/episode': episode}, step=step)

        # sample action for data collection
        if step < args.init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs)

        # run training update
        if step >= args.init_steps:
            num_updates = 1 
            for _ in range(num_updates):
                agent.update(replay_buffer, L, step)

        next_obs, reward, done, _ = env.step(action)

        # allow infinit bootstrap
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(
            done
        )
        episode_reward += reward
        # print(obs.shape)
        # print()
        # print(obs.dtype)
        replay_buffer.add(obs, action, reward, next_obs, done_bool)

        obs = next_obs
        episode_step += 1

    print("FINAL EVAL!!!")
    evaluate(env, agent, video, args.final_eval_eps, L, 10,args, final_eval= True)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    main()

