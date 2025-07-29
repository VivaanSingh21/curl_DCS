import numpy as np
import gym
import cv2
import ipdb
import matplotlib.pyplot as plt
from gym import core, spaces
from dm_env import specs

class ActionNoiseWrapper(gym.ActionWrapper):
	def __init__(self, env, noise_std=0.1):
		super().__init__(env)
		self.noise_std = noise_std
		# ipdb.set_trace()

	def action(self, action):
		noise = np.random.normal(loc=0.0, scale=self.noise_std, size=action.shape)
		noisy_action = action + noise
		
		# If the action space is bounded, clip the action to stay within bounds
		if isinstance(self.action_space, gym.spaces.Box):
			noisy_action = np.clip(noisy_action, self.action_space.low, self.action_space.high)
		
		return noisy_action

class ImageObservationWrapper(gym.Wrapper):
	def __init__(self, env, img_size=84, normalize=True):
		super().__init__(env)
		self.env = env
		self.img_size = img_size
		self.normalize = normalize
		# re-do observation space
		shape = [3, img_size, img_size]
		if self.normalize:
			self.observation_space = spaces.Box(low=-.5, high=.5, shape=shape)
		else:
			self.observation_space = spaces.Box(low=0, high=255, shape=shape)
		self.current_state = env.reset()
		
	def reset(self):
		state = self.env.reset()
		self.current_state = state
		image = self.env.render(mode='rgb_array')
		image = self.convert_image(image)
		return image
	
	def step(self, action):
		state, reward, done, info = self.env.step(action)
		self.current_state = state
		info['state'] = state
		image = self.env.render(mode='rgb_array')
		image = self.convert_image(image)
		return image, reward, done, info

	def convert_image(self,image):
		# image = cv2.resize(image, (self.img_size,self.img_size))/255. - .5
		image = cv2.resize(image, (self.img_size,self.img_size))
		if self.normalize:
			image = image/255. - .5
		image =  np.transpose(image,axes = (2,0,1))
		return image
	
	def render(self):
		return self.env.render(mode='rgb_array')


class DistractorWrapper(gym.Wrapper):
	def __init__(self, env, freq=.1, n_groups = 10, n_per_group=10):
		super().__init__(env)
		self.env = env
		self._max_episode_steps = env._max_episode_steps
		self.n_groups = n_groups
		self.n_per_group = n_per_group

		state_dim = env.observation_space.shape[0] + n_groups*n_per_group
		self.observation_space = gym.spaces.Box(shape=(state_dim,), low=-np.inf, high=np.inf)
		self.action_space = env.action_space

		phases = 2*np.pi*np.random.uniform(size=self.n_groups)
		self.phases = np.concatenate([self.n_per_group*[phases[i]] for i in range(self.n_groups)])
		self.freq = freq

	def reset(self):
		state = self.env.reset()
		phases = 2*np.pi*np.random.uniform(size=self.n_groups)
		self.phases = np.concatenate([self.n_per_group*[phases[i]] for i in range(self.n_groups)])
		self.t = 0

		distractors = np.sin(self.freq*self.t + self.phases)
		return np.concatenate([state,distractors])


	
	def step(self,action):
		ns,r,d,info = self.env.step(action)

		distractors = np.sin(self.freq*self.t + self.phases)

		self.t += 1

		ns = np.concatenate([ns,distractors])

		return ns,r,d,info

class FixedLengthEpisodeWrapper(gym.Wrapper):
	def __init__(self, env, episode_length):
		super().__init__(env)
		self.episode_length = episode_length
		self.current_step = 0

	def reset(self, **kwargs):
		self.current_step = 0
		return self.env.reset(**kwargs)

	def step(self, action):
		done = False
		obs, reward, d, info = self.env.step(action)
		self.current_step += 1

		info['episode_done'] = d

		if self.current_step >= self.episode_length:
			done = True
			
		return obs, reward, done, info
	
	@property
	def current_state(self):
		return self.env.current_state

class PixelRenormalization(gym.ObservationWrapper):
	def __init__(self, env):
		super().__init__(env)

	def observation(self, obs):
		return obs / 255.0 - 0.5

	@property
	def current_state(self):
		return self.env.current_state

# class DistractingControlWrapper(core.Env):
#     def __init__(
#         self,
#          domain_name,
#          task_name,
#          difficulty=None,
#          dynamic=False,
#          background_dataset_path=None,
#          background_dataset_videos="train",
#          background_kwargs=None,
#          camera_kwargs=None,
#          color_kwargs=None,
#          task_kwargs=None,
#          environment_kwargs=None,
#          visualize_reward=False,
#          render_kwargs=None,
#          pixels_only=True,
#          pixels_observation_key="pixels",
#          env_state_wrappers=None):
		
		
		
		
		
		
# 		assert 'random' in task_kwargs, 'please specify a seed, for deterministic behaviour'
#         self._height = height
#         self._width = width
#         self._camera_id = camera_id
#         self._frame_skip = frame_skip
#         self._channels_first = channels_first

#         # create task
#         self._env = suite.load(
#             domain_name=domain_name,
#          task_name=task_name,
#          difficulty=difficulty,
#          dynamic=dynamic,
#          background_dataset_path=background_dataset_path,
#          background_dataset_videos=background_dataset_videos,
#          background_kwargs=background_kwargs,
#          camera_kwargs=camera_kwargs,
#          color_kwargs=color_kwargs,
#          task_kwargs=task_kwargs,
#          environment_kwargs=environment_kwargs,
#          visualize_reward=visualize_reward,
#          render_kwargs=render_kwargs,
#          pixels_only=pixels_only,
#          pixels_observation_key=pixels_observation_key,
#          env_state_wrappers=env_state_wrappers
#         )

#         # true and normalized action spaces
#         self._true_action_space = _spec_to_box([self._env.action_spec()], np.float32)
#         self._norm_action_space = spaces.Box(
#             low=-1.0,
#             high=1.0,
#             shape=self._true_action_space.shape,
#             dtype=np.float32
#         )

#         # create observation space
#         if from_pixels:
#             shape = [3, height, width] if channels_first else [height, width, 3]
#             self._observation_space = spaces.Box(
#                 low=0, high=255, shape=shape, dtype=np.uint8
#             )
#         else:
#             self._observation_space = _spec_to_box(
#                 self._env.observation_spec().values(),
#                 np.float64
#             )
			
#         self._state_space = _spec_to_box(
#             self._env.observation_spec().values(),
#             np.float64
#         )
		
#         self.current_state = None

#         # set seed
#         self.seed(seed=task_kwargs.get('random', 1))

#     def __getattr__(self, name):
#         return getattr(self._env, name)

#     def _get_obs(self, time_step):
#         if self._from_pixels:
#             obs = self.render(
#                 height=self._height,
#                 width=self._width,
#                 camera_id=self._camera_id
#             )
#             if self._channels_first:
#                 obs = obs.transpose(2, 0, 1).copy()
#         else:
#             obs = _flatten_obs(time_step.observation)
#         return obs

#     def _convert_action(self, action):
#         action = action.astype(np.float64)
#         true_delta = self._true_action_space.high - self._true_action_space.low
#         norm_delta = self._norm_action_space.high - self._norm_action_space.low
#         action = (action - self._norm_action_space.low) / norm_delta
#         action = action * true_delta + self._true_action_space.low
#         action = action.astype(np.float32)
#         return action

#     @property
#     def observation_space(self):
#         return self._observation_space

#     @property
#     def state_space(self):
#         return self._state_space

#     @property
#     def action_space(self):
#         return self._norm_action_space

#     @property
#     def reward_range(self):
#         return 0, self._frame_skip

#     def seed(self, seed):
#         self._true_action_space.seed(seed)
#         self._norm_action_space.seed(seed)
#         self._observation_space.seed(seed)

#     def step(self, action):
#         assert self._norm_action_space.contains(action)
#         action = self._convert_action(action)
#         assert self._true_action_space.contains(action)
#         reward = 0
#         extra = {'internal_state': self._env.physics.get_state().copy()}

#         for _ in range(self._frame_skip):
#             time_step = self._env.step(action)
#             reward += time_step.reward or 0
#             done = time_step.last()
#             if done:
#                 break
#         obs = self._get_obs(time_step)
#         self.current_state = _flatten_obs(time_step.observation)
#         extra['discount'] = time_step.discount
#         return obs, reward, done, extra

#     def reset(self):
#         time_step = self._env.reset()
#         self.current_state = _flatten_obs(time_step.observation)
#         obs = self._get_obs(time_step)
#         return obs

#     def render(self, mode='rgb_array', height=None, width=None, camera_id=0):
#         assert mode == 'rgb_array', 'only support rgb_array mode, given %s' % mode
#         height = height or self._height
#         width = width or self._width
#         camera_id = camera_id or self._camera_id
#         return self._env.physics.render(
#             height=height, width=width, camera_id=camera_id
#         )


# class DistractingControlWrapper(gym.Wrapper):
# 	def __init__(self, env):
# 		super().__init__(env)
# 		self.env = env

# 		h,w,ch = env.observation_spec()['pixels'].shape
# 		self._observation_space = spaces.Box(
# 				low=0, high=255, shape=(ch,h,w), dtype=np.uint8
# 			)

# 		self._action_space = 

# 	def reset(self, **kwargs):
# 		time_step = env.reset(**kwargs)
# 		return time_step.observation['pixels']


# 	def step(self, action):
# 		time_step = self.env.step(action)
# 		reward = time_step.reward
# 		if time_step.reward == False or time_step.reward == None:
# 			ipdb.set_trace() 
# 		done = time_step.last()
# 		obs = time_step.observation['pixels']
# 		obs = obs.transpose(2, 0, 1).copy() # channels first

# 		return obs,reward,done,time_step

# 	def _convert_action(self, action):
#         action = action.astype(np.float64)
#         true_delta = self._true_action_space.high - self._true_action_space.low
#         norm_delta = self._norm_action_space.high - self._norm_action_space.low
#         action = (action - self._norm_action_space.low) / norm_delta
#         action = action * true_delta + self._true_action_space.low
#         action = action.astype(np.float32)
#         return action

# 	@property
# 	def observation_space(self):
# 		return self._observation_space

# 	@property
# 	def action_space(self):
# 		return self._norm_action_space

def _spec_to_box(spec, dtype):
	def extract_min_max(s):
		assert s.dtype == np.float64 or s.dtype == np.float32
		dim = np.int(np.prod(s.shape))
		if type(s) == specs.Array:
			bound = np.inf * np.ones(dim, dtype=np.float32)
			return -bound, bound
		elif type(s) == specs.BoundedArray:
			zeros = np.zeros(dim, dtype=np.float32)
			return s.minimum + zeros, s.maximum + zeros

	mins, maxs = [], []
	for s in spec:
		mn, mx = extract_min_max(s)
		mins.append(mn)
		maxs.append(mx)
	low = np.concatenate(mins, axis=0).astype(dtype)
	high = np.concatenate(maxs, axis=0).astype(dtype)
	assert low.shape == high.shape
	return spaces.Box(low, high, dtype=dtype)

# for getting distance of finger to target in reacher:
# env.unwrapped._env.finger_to_target_dist()


if __name__ == '__main__':

	#### MEMORY TEST #####
	import sys
	import torch
	env = gym.make('HalfCheetah-v2')
	env = ImageObservationWrapper(env,img_size=84)

	imgs = []
	for i in range(10**3):
		imgs.append(env.reset())
	
	imgs = np.stack(imgs)
	print('imgs: ',sys.getsizeof(imgs)/10**6)
	img_ints = (255*(imgs+.5)).astype(np.uint8)
	print('img_ints: ',sys.getsizeof(img_ints)/10**6)
	# ipdb.set_trace()

	img_torch = torch.tensor(imgs,dtype=torch.float32)
	print('img_torch: ',(img_torch.element_size() * img_torch.nelement())/10**6)
	
	img_torch_int = torch.tensor(img_ints,dtype=torch.uint8)
	print('img_torch_int: ',(img_torch_int.element_size() * img_torch_int.nelement())/10**6)
	ipdb.set_trace()




	# env = gym.make('HalfCheetah-v2')
	# env = ImageObservationWrapper(env)

	# o = env.reset()
	# for i in range(10):
	# 	o,r,d,info = env.step(env.action_space.sample())
	# 	print('o.shape: ', o.shape)
	# 	ipdb.set_trace()