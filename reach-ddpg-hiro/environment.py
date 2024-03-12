"""
A baxter picks up a cup with its left arm and then passes it to its right arm.
This script contains examples of:
    - Path planning (linear and non-linear).
    - Using multiple arms.
    - Using a gripper.
"""
from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.robots.arms.baxter import BaxterLeft
from pyrep.robots.end_effectors.baxter_gripper import BaxterGripper
from pyrep.objects.dummy import Dummy
from pyrep.objects.shape import Shape
import gymnasium as gym
import numpy as np
from pyrep.errors import IKError

EPISODES = 5
EPISODE_LENGTH = 300

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

SCENE_FILE = join(dirname(abspath(__file__)), 'scene_reach.ttt')

class ReachEnv(gym.Env):
    def __init__(self, headless=True):
        super(ReachEnv, self).__init__()
        
        self.pr = PyRep()

        self.pr.launch(SCENE_FILE, headless=headless)
        self.pr.start()

        self.left_arm = BaxterLeft()
        self.left_arm_ee_tip = self.left_arm.get_tip()
        self.left_arm.set_joint_positions([-0.2, -0.86, -0.15, 1.58, 0.13, 0.85, 0.47], disable_dynamics=True)
        self.left_arm.set_control_loop_enabled(True)
        self.initial_joint_positions = self.left_arm.get_joint_positions()

        self.DISTANCE_THRESHOLD = 0.05

        self.episode_counter = 0
        self.euclidean_dist = 0

        self.bb = Shape('Boundary')
        self.bbArm = Shape('ArmBoundary')
        
        self.bb_min = np.zeros(3)
        self.bb_min[0] = self.bb.get_position()[0] + self.bb.get_bounding_box()[0] # min
        self.bb_min[1] = self.bb.get_position()[1] + self.bb.get_bounding_box()[2] # min
        self.bb_min[2] = self.bb.get_position()[2] + self.bb.get_bounding_box()[4] # min
        
        self.bb_max = np.zeros(3)
        self.bb_max[0] = self.bb.get_position()[0] + self.bb.get_bounding_box()[1] # max
        self.bb_max[1] = self.bb.get_position()[1] + self.bb.get_bounding_box()[3] # max
        self.bb_max[2] = self.bb.get_position()[2] + self.bb.get_bounding_box()[5] # max

        self.bbArm_min = np.zeros(3)
        self.bbArm_min[0] = self.bbArm.get_position()[0] + self.bbArm.get_bounding_box()[0] # min
        self.bbArm_min[1] = self.bbArm.get_position()[1] + self.bbArm.get_bounding_box()[2] # min
        self.bbArm_min[2] = self.bbArm.get_position()[2] + self.bbArm.get_bounding_box()[4] # min
        
        self.bbArm_max = np.zeros(3)
        self.bbArm_max[0] = self.bbArm.get_position()[0] + self.bbArm.get_bounding_box()[1] # max
        self.bbArm_max[1] = self.bbArm.get_position()[1] + self.bbArm.get_bounding_box()[3] # max
        self.bbArm_max[2] = self.bbArm.get_position()[2] + self.bbArm.get_bounding_box()[5] # max

        self.observation_space = gym.spaces.Box(low = np.array([self.bbArm_min[0], self.bbArm_min[1], self.bbArm_min[2], self.bbArm_min[0], self.bbArm_min[1], self.bbArm_min[2], -100,-100,-100]),
                                                high = np.array([self.bbArm_max[0], self.bbArm_max[1], self.bbArm_max[2], self.bbArm_max[0], self.bbArm_max[1], self.bbArm_max[2], 100,100,100]),
                                                dtype=np.float32)
        self.action_space = gym.spaces.Box(low = -0.005, high = 0.005, shape=(3,), dtype=np.float32)

        self.cub = Shape('Cuboid')
        self.sample_obj_location(self.cub)
        self.cub.set_dynamic(False)
        print(self.cub.get_position())

    def step(self, action):
        # action = np.clip(action, self.action_space.low, self.action_space.high) # action: [dx, dy, dz]
        # relative_to = self.left_arm_ee_tip
        #print(action)
        ee_pos = self.left_arm_ee_tip.get_position()
        self.old_dist = np.linalg.norm(self.left_arm_ee_tip.get_position()- self.cub.get_position())
        #self.ee_quat = self.left_arm_ee_tip.get_quaternion(relative_to=self.left_arm_ee_tip)
        # print(action,np.linalg.norm(action))
        
        try:
            joint_positions = self.left_arm.solve_ik_via_jacobian(position=ee_pos+action,quaternion=self.ee_quat)
            self.left_arm.set_joint_target_positions(joint_positions)
        except IKError as e:
            print(bcolors.BOLD +  bcolors.FAIL + "IKError: " + bcolors.ENDC + str(e))
            return self.get_observation(), -100, True, True, {}

        self.pr.step()  # Step the simulator
        self.episode_counter += 1
        if self.episode_counter >= EPISODE_LENGTH:
            return self.get_observation(), -100, True, True, {}
        else:
            return self.get_observation(), self.get_reward("dense"), self.is_done(), False, {}
        
    def reset(self, seed=0):
        print(bcolors.BOLD + bcolors.OKCYAN + "Step: " + str(self.episode_counter) + bcolors.ENDC)
        self.episode_counter = 0
        self.left_arm.set_joint_positions([-0.2, -0.86, -0.15, 1.58, 0.13, 0.85, 0.47], disable_dynamics=True)
        self.left_arm.set_control_loop_enabled(True)

        self.ee_quat = self.left_arm_ee_tip.get_quaternion()
        self.sample_obj_location(self.cub)
        self.cub.set_dynamic(False)
        print(self.cub.get_position())

        return self.get_observation(), {}

    def sample_obj_location(self, obj, min_change=None, max_change=None):
        pos = obj.get_position()
        random_pos = np.random.uniform(low=self.bb_min[:2], high=self.bb_max[:2], size=(2,))
        random_pos = np.append(random_pos, 9.2267e-01)
        if min_change is not None:
            random_pos = np.clip(random_pos, pos - min_change, pos + min_change)
        if max_change is not None:
            random_pos = np.clip(random_pos, pos - max_change, pos + max_change)
        obj.set_position(random_pos)
        
    def get_observation(self) -> np.array:
        curr_tip_pos = self.left_arm_ee_tip.get_position()
        delta_pos = self.left_arm_ee_tip.get_position(relative_to=self.cub)
        distx = np.linalg.norm(self.left_arm_ee_tip.get_position()[0] - self.cub.get_position()[0])
        disty = np.linalg.norm(self.left_arm_ee_tip.get_position()[1] - self.cub.get_position()[1])
        distz = np.linalg.norm(self.left_arm_ee_tip.get_position()[2] - self.cub.get_position()[2])
        return np.array([curr_tip_pos, delta_pos, np.array([distx,disty,distz])]).reshape(9)
    
    def is_done(self) -> bool:
        curr_tip_pos = self.left_arm_ee_tip.get_position()
        cub_pos = self.cub.get_position()
        self.euclidean_dist = np.linalg.norm(curr_tip_pos - cub_pos)
        return self.is_out_boundary() or (self.euclidean_dist < self.DISTANCE_THRESHOLD)
    
    def is_out_boundary(self):
        ee_pos = self.left_arm_ee_tip.get_position()
        return (ee_pos[2] < 8.0e-01) or (ee_pos[0] < self.bbArm_min[0] or ee_pos[0] > self.bbArm_max[0]) or (ee_pos[1] < self.bbArm_min[1] or ee_pos[1] > self.bbArm_max[1])

    def get_reward(self, reward_type='sparse'):
        # 1) Goal is put the cup to the above the bowl (x,y coordinates as close as possible)
        # 2) episodes should not last too long
        # 3) end effector should not take the cup too much away from the bowl
        # 4) cup orientation is constant during episode
        # 5) distance: euclidean_distance between cup and bowl in xy plane
        if reward_type == 'sparse':
            return 1 if self.euclidean_dist < self.DISTANCE_THRESHOLD else 0
        elif reward_type == 'dense':
            self.euclidean_dist = np.linalg.norm(self.left_arm_ee_tip.get_position()- self.cub.get_position())
            
            #if self.old_dist - self.euclidean_dist <= 0:
            #    distance_reward = -1*(1 - 1 / (self.euclidean_dist + 1))
            #else:
            #    distance_reward = 1 / (self.euclidean_dist + 1)
            out_boundary_penalty = -100 if self.is_out_boundary() else 0
            reached_reward = 2000 if self.euclidean_dist < self.DISTANCE_THRESHOLD else 0
            return (1 - self.euclidean_dist / self.DISTANCE_THRESHOLD) + reached_reward + out_boundary_penalty

            #duration_penalty = -0.01 * self.episode_counter
            #
            #reward = 0.6 * distance_reward + 0.2 * duration_penalty + 0.2 * out_boundary_penalty
            #return reward

        

            #self.euclidean_dist = np.linalg.norm(self.left_arm_ee_tip.get_position()- self.cub.get_position())

            #reached_reward = 1000 if self.euclidean_dist < self.DISTANCE_THRESHOLD else 0
            #return (1 - self.euclidean_dist / self.DISTANCE_THRESHOLD) + reached_reward

            
        else:
            raise NotImplementedError
        
    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()
