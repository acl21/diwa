"""
Multi-step wrapper. Allow executing multiple environmnt steps. Returns stacked observation and optionally stacked previous action.

Modified from https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/gym_util/multistep_wrapper.py

TODO: allow cond_steps != img_cond_steps (should be implemented in training scripts, not here)
"""

import gym
from typing import Optional
from gym import spaces
import numpy as np
from collections import defaultdict, deque
from env.wrapper.calvin_multistep import (
    repeated_space,
    stack_last_n_obs,
    aggregate,
    dict_take_last_n,
)


class RealMultiStep(gym.Wrapper):

    def __init__(
        self,
        env,
        n_obs_steps=1,
        n_action_steps=1,
        reward_agg_method="sum",  # never use other types
        prev_action=True,
        reset_within_step=False,
        pass_full_observations=False,
        verbose=False,
        **kwargs,
    ):
        super().__init__(env)
        self._single_action_space = env.action_space
        self._action_space = repeated_space(env.action_space, n_action_steps)
        self._observation_space = repeated_space(env.observation_space, n_obs_steps)
        self.max_episode_steps = env.max_episode_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.reward_agg_method = reward_agg_method
        self.prev_action = prev_action
        self.reset_within_step = reset_within_step
        self.pass_full_observations = pass_full_observations
        self.verbose = verbose

    def reset(
        self,
        options: dict = {},
        robot_obs: Optional[np.ndarray] = None,
        target_pos: Optional[np.ndarray] = None,
        target_orn: Optional[np.ndarray] = None,
        gripper_state: Optional[str] = None,
    ):
        """Resets the environment."""
        obs, info = self.env.reset(
            options=options,
            robot_obs=robot_obs,
            target_pos=target_pos,
            target_orn=target_orn,
            gripper_state=gripper_state,
        )
        self.obs = deque([obs], maxlen=max(self.n_obs_steps + 1, self.n_action_steps))
        if self.prev_action:
            self.action = deque(
                [self._single_action_space.sample()], maxlen=self.n_obs_steps
            )
        self.reward = list()
        self.terminated = list()
        self.truncated = list()
        self.info = defaultdict(lambda: deque(maxlen=self.n_obs_steps + 1))
        obs = self._get_obs(self.n_obs_steps)
        self.metadata = {}

        self.cnt = 0
        return obs, info

    def step(self, action):
        """
        actions: (n_action_steps,) + action_shape
        """
        if action.ndim == 1:  # in case action_steps = 1
            action = action[None]
        truncated = False
        terminated = False
        for act_step, act in enumerate(action):
            self.cnt += 1

            if terminated or truncated:
                # termination
                break
            observation, reward, terminated, truncated, info = self.env.step(act)

            self.obs.append(observation)
            self.action.append(act)
            self.reward.append(reward)
            self.terminated.append(terminated)
            self.truncated.append(truncated)
            self._add_info(info)

        observation = self._get_obs(self.n_obs_steps)
        reward = aggregate(self.reward, self.reward_agg_method)
        terminated = aggregate(self.terminated, "max")
        truncated = aggregate(self.truncated, "max")
        info = dict_take_last_n(self.info, self.n_obs_steps)
        if self.pass_full_observations:  # right now this assume n_obs_steps = 1
            info["full_obs"] = self._get_obs(act_step + 1)

        # In mujoco case, done can happen within the loop above
        if self.reset_within_step and self.terminated[-1]:
            observation = (
                self.reset()
            )  # TODO: arguments? this cannot handle video recording right now since needs to pass in options
            self.verbose and print("Reset env within wrapper.")

        # reset reward and done for next step
        self.reward = list()
        self.terminated = list()
        self.truncated = list()
        return observation, reward, terminated, truncated, info

    def _get_obs(self, n_steps=1):
        """
        Output (n_steps,) + obs_shape
        """
        assert len(self.obs) > 0
        if isinstance(self.observation_space, spaces.Box):
            return stack_last_n_obs(self.obs, n_steps)
        elif isinstance(self.observation_space, spaces.Dict):
            result = dict()
            for key in self.observation_space.keys():
                result[key] = stack_last_n_obs([obs[key] for obs in self.obs], n_steps)
            return result
        else:
            raise RuntimeError("Unsupported space type")

    def get_prev_action(self, n_steps=None):
        if n_steps is None:
            n_steps = self.n_obs_steps - 1  # exclude current step
        assert len(self.action) > 0
        return stack_last_n_obs(self.action, n_steps)

    def _add_info(self, info):
        for key, value in info.items():
            self.info[key].append(value)

    def render(self, **kwargs):
        """Not the best design"""
        return self.env.render(**kwargs)
