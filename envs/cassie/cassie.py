# isolated cassie env
import math
from math import floor
import gym
from gym import spaces
import numpy as np
import os
import random
import copy
import pickle
import mujoco
import mujoco_viewer
import yaml


class CassieRefEnv(gym.Env):
    def __init__(self, cfg, **kwargs):
        self.config = cfg
        self.model = self.config['system']['root_path'] + self.config['system']['mjcf_path']
        self.visual = self.config['system']['visual']
        self.model = mujoco.MjModel.from_xml_path(self.model)
        self.data = mujoco.MjData(self.model)
        if self.visual:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)

        self.dynamics_randomization = self.config['system']['dynamics_randomization']
        self.termination = False

        # state buffer
        self.state_buffer = []
        self.buffer_size = self.config['env']['state_buffer_size']  # 3

        # Observation space and State space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.buffer_size * 39 + 2 + 2,))
        # self.action_space = spaces.Box(low=np.array([-1] * 10), high=np.array([1] * 10))

        self.action_high = np.array([0.26, 0.39, 0.9, 0.55, 0.8, 0.26, 0.39, 0.9, 0.55, 0.8], dtype=np.float32)
        self.action_space = spaces.Box(-self.action_high, self.action_high, dtype=np.float32)

        self.P = np.array(self.config['control']['P'])
        self.D = np.array(self.config['control']['D'])
        self.foot_pos = [0] * 6
        self.joint_vel_left = np.zeros(5, dtype=np.float32)
        self.joint_vel_right = np.zeros(5, dtype=np.float32)

        self.simrate = self.config['control']['decimation']  # simulate X mujoco steps with same pd target. 50 brings simulation from 2000Hz to exactly 40Hz
        self.time = 0  # number of time steps in current episode
        self.phase = 0  # portion of the phase the robot is in
        self.counter = 0  # number of phase cycles completed in episode
        self.time_limit = self.config['env']['time_limit']
        self.offset = np.array(self.config['init_state']['default_left_joint_angles']
                             + self.config['init_state']['default_right_joint_angles'])
        self.time_buf = 0

        self.max_speed = self.config['commands']['lin_vel_x'][1]
        self.min_speed = self.config['commands']['lin_vel_x'][0]
        self.max_side_speed = self.config['commands']['lin_vel_y'][1]
        self.min_side_speed = self.config['commands']['lin_vel_y'][0]

        #### Dynamics Randomization ####

        self.max_pitch_incline = 0.03
        self.max_roll_incline = 0.03
        self.encoder_noise = 0.01
        self.damping_low = 0.3
        self.damping_high = 5.0
        self.mass_low = 0.5
        self.mass_high = 1.5
        self.fric_low = 0.4
        self.fric_high = 1.1
        self.speed = np.random.uniform(self.min_speed, self.max_speed)
        self.side_speed = np.random.uniform(self.min_side_speed, self.max_side_speed)
        self.orient_add = 0

        # Default dynamics parameters
        self.default_damping = self.model.dof_damping.copy()
        self.default_mass = self.model.body_mass.copy()
        self.default_ipos = self.model.body_ipos.copy()
        self.default_fric = self.model.geom_friction.copy()
        self.default_rgba = self.model.geom_rgba.copy()
        self.default_quat = self.model.geom_quat.copy()

        self.motor_encoder_noise = np.zeros(10)
        self.joint_encoder_noise = np.zeros(6)

        # rew_buf
        self.rew_ref = 0
        self.rew_spring = 0
        self.rew_ori = 0
        self.rew_vel = 0
        self.rew_termin = 0
        self.reward = 0
        self.rew_ref_buf = 0
        self.rew_spring_buf = 0
        self.rew_ori_buf = 0
        self.rew_vel_buf = 0
        self.rew_termin_buf = 0
        self.reward_buf = 0
        self.omega_buf = 0
        self.omega = 0


    def custom_footheight(self):
        # 定制化的脚步高度，在sin<0时，取0
        phase = self.phase
        h = 0.15
        h1 = max(0, h * np.sin(2 * np.pi / 28 * phase) - 0.2 * h)
        h2 = max(0, h * np.sin(np.pi + 2 * np.pi / 28 * phase) - 0.2 * h)
        return [h1, h2]

    def get_foot_pos(self):
        # left foot no 13 , 13*3=39
        left_foot_pos = np.copy(self.data.xpos[13])
        # right foot no 25 , 25*3=75
        right_foot_pos = np.copy(self.data.xpos[25])
        offset_footJoint2midFoot = math.sqrt(pow(0.01762, 2) + pow(0.05219, 2))

        left_foot_pos[2] = left_foot_pos[2] - offset_footJoint2midFoot
        right_foot_pos[2] = right_foot_pos[2] - offset_footJoint2midFoot

        self.foot_pos = np.concatenate([left_foot_pos, right_foot_pos])

    def set_const(self):
        qpos_init = [0, 0, 1.01, 1, 0, 0, 0,
                   0.0045, 0, 0.4973, 0.9785, -0.0164, 0.01787, -0.2049,
                   -1.1997, 0, 1.4267, 0, -1.5244, 1.5244, -1.5968,
                   -0.0045, 0, 0.4973, 0.9786, 0.00386, -0.01524, -0.2051,
                   -1.1997, 0, 1.4267, 0, -1.5244, 1.5244, -1.5968]

        qvel_zero = [0] * self.model.nv
        qacc_zero = [0] * self.model.nv

        self.data.qpos = qpos_init
        self.data.qvel = qvel_zero
        self.data.qacc = qacc_zero
        self.data.time = 0.0
        # mujoco.mju_copy(self.data.qpos, qpos_init, self.model.nq)
        # mujoco.mju_copy(self.data.qvel, qvel_zero, self.model.nv)
        # mujoco.mju_copy(self.data.qacc, qacc_zero, self.model.nv)
        mujoco.mj_forward(self.model, self.data)

    def step_simulation(self, action):
        target = action + self.offset

        self.joint_vel_left[0] = 0.7 * self.joint_vel_left[0] + 0.3 * self.data.qvel[6]
        self.joint_vel_left[1] = 0.7 * self.joint_vel_left[1] + 0.3 * self.data.qvel[7]
        self.joint_vel_left[2] = 0.7 * self.joint_vel_left[2] + 0.3 * self.data.qvel[8]
        self.joint_vel_left[3] = 0.7 * self.joint_vel_left[3] + 0.3 * self.data.qvel[12]
        self.joint_vel_left[4] = 0.7 * self.joint_vel_left[4] + 0.3 * self.data.qvel[18]

        self.joint_vel_right[0] = 0.7 * self.joint_vel_right[0] + 0.3 * self.data.qvel[19]
        self.joint_vel_right[1] = 0.7 * self.joint_vel_right[1] + 0.3 * self.data.qvel[20]
        self.joint_vel_right[2] = 0.7 * self.joint_vel_right[2] + 0.3 * self.data.qvel[21]
        self.joint_vel_right[3] = 0.7 * self.joint_vel_right[3] + 0.3 * self.data.qvel[25]
        self.joint_vel_right[4] = 0.7 * self.joint_vel_right[4] + 0.3 * self.data.qvel[31]

        self.data.ctrl[0] = self.P[0] * (target[0] - self.data.qpos[7]) - self.D[0] * self.joint_vel_left[0]
        self.data.ctrl[1] = self.P[1] * (target[1] - self.data.qpos[8]) - self.D[1] * self.joint_vel_left[1]
        self.data.ctrl[2] = self.P[2] * (target[2] - self.data.qpos[9]) - self.D[2] * self.joint_vel_left[2]
        self.data.ctrl[3] = self.P[3] * (target[3] - self.data.qpos[14]) - self.D[3] * self.joint_vel_left[3]
        self.data.ctrl[4] = self.P[4] * (target[4] - self.data.qpos[20]) - self.D[4] * self.joint_vel_left[4]

        self.data.ctrl[5] = self.P[0] * (target[5] - self.data.qpos[21]) - self.D[0] * self.joint_vel_right[0]
        self.data.ctrl[6] = self.P[1] * (target[6] - self.data.qpos[22]) - self.D[1] * self.joint_vel_right[1]
        self.data.ctrl[7] = self.P[2] * (target[7] - self.data.qpos[23]) - self.D[2] * self.joint_vel_right[2]
        self.data.ctrl[8] = self.P[3] * (target[8] - self.data.qpos[28]) - self.D[3] * self.joint_vel_right[3]
        self.data.ctrl[9] = self.P[4] * (target[9] - self.data.qpos[34]) - self.D[4] * self.joint_vel_right[4]

        # self.data.ctrl[0] = self.P[0] * (target[0] - self.data.sensordata[0]) - self.D[0] * self.data.qvel[6]
        # self.data.ctrl[1] = self.P[1] * (target[1] - self.data.sensordata[1]) - self.D[1] * self.data.qvel[7]
        # self.data.ctrl[2] = self.P[2] * (target[2] - self.data.sensordata[2]) - self.D[2] * self.data.qvel[8]
        # self.data.ctrl[3] = self.P[3] * (target[3] - self.data.sensordata[3]) - self.D[3] * self.data.qvel[12]
        # self.data.ctrl[4] = self.P[4] * (target[4] - self.data.sensordata[4]) - self.D[4] * self.data.qvel[18]
        #
        # self.data.ctrl[5] = self.P[0] * (target[5] - self.data.sensordata[8]) - self.D[0] * self.data.qvel[19]
        # self.data.ctrl[6] = self.P[1] * (target[6] - self.data.sensordata[9]) - self.D[1] * self.data.qvel[20]
        # self.data.ctrl[7] = self.P[2] * (target[7] - self.data.sensordata[10]) - self.D[2] * self.data.qvel[21]
        # self.data.ctrl[8] = self.P[3] * (target[8] - self.data.sensordata[11]) - self.D[3] * self.data.qvel[25]
        # self.data.ctrl[9] = self.P[4] * (target[9] - self.data.sensordata[12]) - self.D[4] * self.data.qvel[31]

        # self.data.ctrl[0] = self.P[0] * (target[0] - self.data.qpos[7]) - self.D[0] * self.data.qvel[6]
        # self.data.ctrl[1] = self.P[1] * (target[1] - self.data.qpos[8]) - self.D[1] * self.data.qvel[7]
        # self.data.ctrl[2] = self.P[2] * (target[2] - self.data.qpos[9]) - self.D[2] * self.data.qvel[8]
        # self.data.ctrl[3] = self.P[3] * (target[3] - self.data.qpos[14]) - self.D[3] * self.data.qvel[12]
        # self.data.ctrl[4] = self.P[4] * (target[4] - self.data.qpos[20]) - self.D[4] * self.data.qvel[18]
        #
        # self.data.ctrl[5] = self.P[0] * (target[5] - self.data.qpos[21]) - self.D[0] * self.data.qvel[19]
        # self.data.ctrl[6] = self.P[1] * (target[6] - self.data.qpos[22]) - self.D[1] * self.data.qvel[20]
        # self.data.ctrl[7] = self.P[2] * (target[7] - self.data.qpos[23]) - self.D[2] * self.data.qvel[21]
        # self.data.ctrl[8] = self.P[3] * (target[8] - self.data.qpos[28]) - self.D[3] * self.data.qvel[25]
        # self.data.ctrl[9] = self.P[4] * (target[9] - self.data.qpos[34]) - self.D[4] * self.data.qvel[31]

        # for i in range(5):
        #     left_leg_torque = self.P[i] * (target[i] - self.data.sensordata[i]) - self.D[i] * self.data.sensordata[i+10]
        #     right_leg_torque = self.P[i] * (target[i+5] - self.data.sensordata[i+8]) - self.D[i] * self.data.sensordata[i+5+10]
        #     self.data.ctrl[i] = left_leg_torque
        #     self.data.ctrl[i+5] = right_leg_torque
        mujoco.mj_step(self.model, self.data)

    def step(self, action):

        for _ in range(self.simrate):
            self.step_simulation(action)

        self.time += 1
        self.phase += 1
        if self.phase >= 28:
            self.phase = 0
            self.counter += 1

        obs = self.get_state()

        self.get_foot_pos()
        # 保持身体高度
        xpos, ypos, height = self.qpos[0], self.qpos[1], self.qpos[2]
        xtarget, ytarget, ztarget = self.ref_pos[0], self.ref_pos[1], self.ref_pos[2]
        pos2target = (xpos - xtarget) ** 2 + (ypos - ytarget) ** 2 + (height - ztarget) ** 2
        die_radii = 0.6 + (self.speed ** 2 + self.side_speed ** 2) ** 0.5
        # 为什么要把速度的方差和身体位置方差作比较
        self.termination = height < 0.6 or height > 1.2 or pos2target > die_radii ** 2
        done = self.termination or self.time >= self.time_limit

        if self.visual:
            self.render()
        reward = self.compute_reward(action)

        return obs, reward, done, {}

    def reset(self):
        if self.time != 0:
            self.rew_ref_buf = self.rew_ref / self.time
            self.rew_spring_buf = self.rew_spring / self.time
            self.rew_ori_buf = self.rew_ori / self.time
            self.rew_vel_buf = self.rew_vel / self.time
            self.rew_termin_buf = self.rew_termin / self.time
            self.reward_buf = self.reward  # / self.time
            self.time_buf = self.time
            self.omega_buf = self.omega / self.time

        self.rew_ref = 0
        self.rew_spring = 0
        self.rew_ori = 0
        self.rew_vel = 0
        self.rew_termin = 0
        self.reward = 0
        self.omega = 0

        self.speed = 0.6
        self.side_speed = 0
        self.time = 0
        self.counter = 0
        self.termination = False
        self.phase = int((random.random() > 0.5) * 14)  # random phase: 0 or 14

        # Randomize dynamics:
        if self.dynamics_randomization:
            damp = self.default_damping.copy()

            pelvis_damp_range = [[damp[0], damp[0]],
                                 [damp[1], damp[1]],
                                 [damp[2], damp[2]],
                                 [damp[3], damp[3]],
                                 [damp[4], damp[4]],
                                 [damp[5], damp[5]]]  # 0->5

            hip_damp_range = [[damp[6] * self.damping_low, damp[6] * self.damping_high],
                              [damp[7] * self.damping_low, damp[7] * self.damping_high],
                              [damp[8] * self.damping_low, damp[8] * self.damping_high]]  # 6->8 and 19->21

            achilles_damp_range = [[damp[9] * self.damping_low, damp[9] * self.damping_high],
                                   [damp[10] * self.damping_low, damp[10] * self.damping_high],
                                   [damp[11] * self.damping_low, damp[11] * self.damping_high]]  # 9->11 and 22->24

            knee_damp_range = [[damp[12] * self.damping_low, damp[12] * self.damping_high]]  # 12 and 25
            shin_damp_range = [[damp[13] * self.damping_low, damp[13] * self.damping_high]]  # 13 and 26
            tarsus_damp_range = [[damp[14] * self.damping_low, damp[14] * self.damping_high]]  # 14 and 27

            heel_damp_range = [[damp[15], damp[15]]]  # 15 and 28
            fcrank_damp_range = [[damp[16] * self.damping_low, damp[16] * self.damping_high]]  # 16 and 29
            prod_damp_range = [[damp[17], damp[17]]]  # 17 and 30
            foot_damp_range = [[damp[18] * self.damping_low, damp[18] * self.damping_high]]  # 18 and 31

            side_damp = hip_damp_range + achilles_damp_range + knee_damp_range + shin_damp_range + tarsus_damp_range + heel_damp_range + fcrank_damp_range + prod_damp_range + foot_damp_range
            damp_range = pelvis_damp_range + side_damp + side_damp
            # 此处会在运行一段时间后报错，待审核
            # File "mtrand.pyx", line 1116, in numpy.random.mtrand.RandomState.uniform
            # OverflowError: Range exceeds valid bounds
            damp_noise = [np.random.uniform(a, b) for a, b in damp_range]

            m = self.default_mass.copy()
            pelvis_mass_range = [[self.mass_low * m[1], self.mass_high * m[1]]]  # 1
            hip_mass_range = [[self.mass_low * m[2], self.mass_high * m[2]],  # 2->4 and 14->16
                              [self.mass_low * m[3], self.mass_high * m[3]],
                              [self.mass_low * m[4], self.mass_high * m[4]]]

            achilles_mass_range = [[self.mass_low * m[5], self.mass_high * m[5]]]  # 5 and 17
            knee_mass_range = [[self.mass_low * m[6], self.mass_high * m[6]]]  # 6 and 18
            knee_spring_mass_range = [[self.mass_low * m[7], self.mass_high * m[7]]]  # 7 and 19
            shin_mass_range = [[self.mass_low * m[8], self.mass_high * m[8]]]  # 8 and 20
            tarsus_mass_range = [[self.mass_low * m[9], self.mass_high * m[9]]]  # 9 and 21
            heel_spring_mass_range = [[self.mass_low * m[10], self.mass_high * m[10]]]  # 10 and 22
            fcrank_mass_range = [[self.mass_low * m[11], self.mass_high * m[11]]]  # 11 and 23
            prod_mass_range = [[self.mass_low * m[12], self.mass_high * m[12]]]  # 12 and 24
            foot_mass_range = [[self.mass_low * m[13], self.mass_high * m[13]]]  # 13 and 25

            side_mass = hip_mass_range + achilles_mass_range \
                        + knee_mass_range + knee_spring_mass_range \
                        + shin_mass_range + tarsus_mass_range \
                        + heel_spring_mass_range + fcrank_mass_range \
                        + prod_mass_range + foot_mass_range

            mass_range = [[0, 0]] + pelvis_mass_range + side_mass + side_mass
            mass_noise = [np.random.uniform(a, b) for a, b in mass_range]

            delta = 0.0
            com_noise_zero = np.zeros([3,3])
            com_noise_random = [np.random.uniform(val - delta, val + delta) for val in self.default_ipos[3:]]
            com_noise = np.concatenate((com_noise_zero, com_noise_random))
            fric_noise = np.zeros([50,3])
            translational = np.random.uniform(self.fric_low, self.fric_high)
            torsional = np.random.uniform(1e-4, 5e-4)
            rolling = np.random.uniform(1e-4, 2e-4)
            for i in range(int(len(self.default_fric))):
                fric_noise[i] = [translational, torsional, rolling]

            self.model.dof_damping = np.clip(damp_noise, 0, None)
            self.model.body_mass = np.clip(mass_noise, 0, None)
            self.model.body_ipos = com_noise
            self.model.geom_friction = np.clip(fric_noise, 0, None)
        else:
            self.model.dof_damping = self.default_damping
            self.model.body_mass = self.default_mass
            self.model.body_ipos = self.default_ipos
            self.model.geom_friction = self.default_fric
        # 随机速度
        self.speed = np.random.uniform(self.min_speed, self.max_speed)
        self.side_speed = np.random.uniform(self.min_side_speed, self.max_side_speed)

        self.model.geom_quat = self.default_quat
        self.set_const()
        return self.get_state()

    def get_state(self):
        self.qpos = np.copy(self.data.qpos)  # dim=35 see cassiemujoco.h for details
        self.qvel = np.copy(self.data.qvel)  # dim=32
        self.state_buffer.append((self.qpos, self.qvel))

        if len(self.state_buffer) > self.buffer_size:
            self.state_buffer.pop(0)
        else:
            while len(self.state_buffer) < self.buffer_size:
                self.state_buffer.append((self.qpos, self.qvel))

        pos = np.array([x[0] for x in self.state_buffer])
        vel = np.array([x[1] for x in self.state_buffer])

        self.ref_pos, self.ref_vel = self.get_kin_next_state()
        command = [self.speed, self.side_speed]
        '''
		Position [1], [2] 				-> Pelvis y, z
				 [3], [4], [5], [6] 	-> Pelvis Orientation qw, qx, qy, qz
				 [7], [8], [9]			-> Left Hip Roll (Motor[0]), Yaw (Motor[1]), Pitch (Motor[2])
				 [14]					-> Left Knee   	(Motor[3])
				 [15]					-> Left Shin   	(Joint[0])
				 [16]					-> Left Tarsus 	(Joint[1])
				 [20]					-> Left Foot   	(Motor[4], Joint[2])
				 [21], [22], [23]		-> Right Hip Roll (Motor[5]), Yaw (Motor[6]), Pitch (Motor[7])
				 [28]					-> Rigt Knee   	(Motor[8])
				 [29]					-> Rigt Shin   	(Joint[3])
				 [30]					-> Rigt Tarsus 	(Joint[4])
				 [34]					-> Rigt Foot   	(Motor[9], Joint[5])
		'''
        pos_index = np.array([2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 16, 20, 21, 22, 23, 28, 29, 30, 34])
        '''
		Velocity [0], [1], [2] 			-> Pelvis x, y, z
				 [3], [4], [5]		 	-> Pelvis Orientation wx, wy, wz
				 [6], [7], [8]			-> Left Hip Roll (Motor[0]), Yaw (Motor[1]), Pitch (Motor[2])
				 [12]					-> Left Knee   	(Motor[3])
				 [13]					-> Left Shin   	(Joint[0])
				 [14]					-> Left Tarsus 	(Joint[1])
				 [18]					-> Left Foot   	(Motor[4], Joint[2])
				 [19], [20], [21]		-> Right Hip Roll (Motor[5]), Yaw (Motor[6]), Pitch (Motor[7])
				 [25]					-> Rigt Knee   	(Motor[8])
				 [26]					-> Rigt Shin   	(Joint[3])
				 [27]					-> Rigt Tarsus 	(Joint[4])
				 [31]					-> Rigt Foot   	(Motor[9], Joint[5])
		'''
        vel_index = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 18, 19, 20, 21, 25, 26, 27, 31])
        # next todo: x,y,z in state -> delta xyz to target + target velo
        return np.concatenate([pos[:, pos_index].reshape(-1), vel[:, vel_index].reshape(-1),
                               [np.sin(self.phase / 28 * 2 * np.pi), np.sin(self.phase / 28 * 2 * np.pi + np.pi)],
                               # self.ref_pos[pos_index], self.ref_vel[vel_index]
                               command
                               ])

    def compute_reward(self, action):
        # 把实际脚的高度和目标脚的高度取方差
        # ref_penalty 数量级0.1
        ref_penalty = 0
        custom_footheight = np.array(self.custom_footheight())
        real_footheight = np.array([self.foot_pos[2], self.foot_pos[5]])
        ref_penalty = np.sum(np.square(custom_footheight - real_footheight))
        ref_penalty = ref_penalty / 0.0025
        # qpos [3] [4] [5] [6]是四元数，初始值 1,0,0,0，除了[3]，其他值保持执行都应该是越小越好
        # orientation_penalty 数量级0.1
        orientation_penalty = (self.qpos[4]) ** 2 + (self.qpos[5]) ** 2 + (self.qpos[6]) ** 2
        orientation_penalty = orientation_penalty / 0.1
        # 速度惩罚，对z轴速度直接惩罚，对x和y轴速度跟踪
        # vel_penalty 数量级 1
        vel_penalty = (self.speed - self.qvel[0]) ** 2 + (self.side_speed - self.qvel[1]) ** 2 + (self.qvel[2]) ** 2
        vel_penalty = vel_penalty / max(0.5 * (self.speed * self.speed + self.side_speed * self.side_speed), 0.01)
        # 弹跳惩罚，对大腿关节在y轴上运动进行惩罚
        # spring_penalty 数量级 1
        spring_penalty = (self.data.qpos[15]) ** 2 + (self.data.qpos[29]) ** 2
        spring_penalty *= 1000

        rew_ref = 0.5 * np.exp(-ref_penalty)
        rew_spring = 0.1 * np.exp(-spring_penalty)
        rew_ori = 0.125 * np.exp(-orientation_penalty)
        rew_vel = 0.375 * np.exp(-vel_penalty)  #
        rew_termin = -10 * self.termination

        R_star = 1
        Rp = (0.75 * np.exp(-vel_penalty) + 0.25 * np.exp(-orientation_penalty)) / R_star
        Ri = np.exp(-ref_penalty) / R_star
        Ri = (Ri - 0.4) / (1.0 - 0.4)

        omega = 0.5

        reward = (1 - omega) * Ri + omega * Rp + rew_spring + rew_termin

        self.rew_ref += rew_ref
        self.rew_spring += rew_spring
        self.rew_ori += rew_ori
        self.rew_vel += rew_vel
        self.rew_termin += rew_termin
        self.reward += reward
        self.omega += omega

        return reward

    def render(self):
        return self.viewer.render()

    def get_kin_state(self):
        pose = np.array([0] * 3)
        vel = np.array([0] * 3)
        pose[0] = self.speed * (self.counter * 28 + self.phase) * (self.simrate / 2000)
        pose[1] = self.side_speed * (self.counter * 28 + self.phase) * (self.simrate / 2000)
        pose[2] = 1.03  #
        vel[0] = self.speed
        return pose, vel

    def get_kin_next_state(self):
        phase = self.phase + 1
        counter = self.counter
        if phase >= 28:
            phase = 0
            counter += 1
        # 修复int错误
        pose = np.array([0] * 3, dtype=np.float32)
        vel = np.array([0] * 3, dtype=np.float32)
        pose[0] = self.speed * (counter * 28 + phase) * (self.simrate / 2000)
        pose[1] = self.side_speed * (counter * 28 + phase) * (self.simrate / 2000)
        pose[2] = 1.03  #
        vel[0] = self.speed
        return pose, vel


if __name__ == "__main__":
    # 导入动作规范化wrapper
    import sys
    sys.path.append("../..")
    from utils.NormalizeActionWrapper import NormalizeActionWrapper

    with open('./config.yaml', 'rb') as stream:
        config = yaml.safe_load(stream)
    config['system']['root_path'] = '../..'
    config['system']['visual'] = True

    env = CassieRefEnv(cfg=config)
    print(env.action_space.high)
    print(env.action_space.low)

    env = NormalizeActionWrapper(env)
    obs = env.reset()
    print(env.action_space)
    for i in range(10000):
        action = np.random.random(10)
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()


