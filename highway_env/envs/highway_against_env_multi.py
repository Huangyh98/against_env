from typing import Dict, Optional, Text

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
import random


from highway_env.vehicle.graphics import VehicleGraphics

Observation = np.ndarray


class HighwayAgainstEnvMulti(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.

    """
    def __init__(self, config: dict = None, render_mode: Optional[str] = None) -> None:
        super().__init__(config, render_mode)
        # model = PPO.load("scripts/against/highway_against_ppo/model")


    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            # "observation": {
            #     "type": "Kinematics"
            # },
            # "action": {
            #     "type": "DiscreteMetaAction",
            # },
            "observation": {
            "type": "MultiAgentObservation",
            "observation_config": {
                    "type": "Kinematics",
                    "features": [
                        "presence",
                        "x",
                        "y",
                        "vx",
                        "vy",
                        "cos_h",
                        "sin_h"
                    ],
            },

            "absolute": False
        },
            "action": {
            "type": "MultiAgentAction",
            "action_config": {
                "type": "DiscreteMetaAction",
                }
            },
            "lanes_count": 4,
            "vehicles_count": 50,
            "controlled_vehicles": 2,
            "initial_lane_id": None,
            "duration": 40,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 1,
            "collision_reward": - 0.7,    # The reward received when colliding with a vehicle.
            "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                                       # zero for other lanes.
            "high_speed_reward": 0.5,  # The reward received when driving at full speed, linearly mapped to zero for
                                       # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": 0,   # The reward received at each lane change action.
            "sparse_reward": 1,
            "on_road_reward": 0.5,
            "reward_speed_range": [20, 30],
            "normalize_reward": True,
            "offroad_terminal": False
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        color_i = 0 
        color_list_i = [VehicleGraphics.EGO_COLOR,VehicleGraphics.PURPLE]
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                lane_id = self.config["initial_lane_id"],
                spacing = self.config["ego_spacing"]
            )
            vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)

            vehicle.color = color_list_i[color_i]
            color_i +=1
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            i = 1
            for _ in range(others):
                # 随机生成一个标志，用于确定车辆出现在前方还是后方
                is_behind = random.choice([True, False])
                
                if is_behind:
                    # 如果在后方，可以使用负的间距
                    spacing = -random.uniform(i * (1 / self.config["vehicles_density"]), (i + 1) * (1 / self.config["vehicles_density"]))
                else:
                    # 如果在前方，使用正的间距
                    spacing = random.uniform(i * (1 / self.config["vehicles_density"]), (i + 1) * (1 / self.config["vehicles_density"]))
                
                vehicle = other_vehicles_type.create_random(self.road, spacing=spacing)
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)
                i += 1

    def _reward(self, action: Action) -> float:
        rewards = self._rewards(action)
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        if self.config["normalize_reward"]:
            reward = utils.lmap(reward,
                                [self.config["collision_reward"],
                                  self.config["sparse_reward"]],
                                [0, 1])
        reward *= rewards['on_road_reward']
        return reward

    # def _reward(self, action: Action) -> float:
    #     rewards = self._rewards(action)
    #     reward_ego = (- self.config.get("collision_reward") * rewards["collision_reward"])
    #     + self.config.get("collision_reward", 0) * rewards["collision_reward"]
    #     + self.config.get("right_lane_reward", 0) * rewards["collision_reward"]
    #     + self.config.get("high_speed_reward", 0) * rewards["collision_reward"] 
    #     + self.config.get("on_road_reward", 0) * rewards["collision_reward"]   
                       
    #     reward_against = self.config.get("collision_reward", 0) * rewards["collision_reward"]
    #     + self.config.get("lane_change_reward", 0) * rewards["collision_reward"]
    #     + self.config.get("on_road_reward", 0) * rewards["collision_reward"]
    #     + self.config.get("sparse_reward", 0) * rewards["collision_reward"]

    #     print(reward_ego, reward_against)

    #     reward = (reward_ego, reward_against)

        # print(reward)

        # if self.config["normalize_reward"]:
        #     reward = utils.lmap(reward,
        #                         [self.config["lane_change_reward"],
        #                           self.config["sparse_reward"] + self.config["collision_reward"]],
        #                         [0, 1])
        # reward *= rewards['on_road_reward']
        # return reward

        
    def _rewards(self, action: Action) -> Dict[Text, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        
        # 距离稀疏奖励
        neighbors = self.road.vehicles  # 获取道路上的所有车辆
        vehicle_position = self.vehicle.position  # 获取被控车辆的位置
        min_distance = float('inf')  # 初始化最小距离为正无穷

        for neighbor in neighbors:
            if neighbor != self.vehicle:
                distance = np.linalg.norm(neighbor.position - vehicle_position)
                min_distance = min(min_distance, distance)
        # 设置稀疏奖励，距离越近奖励越高，距离越远惩罚越高
        sparse_reward = 1.0 / (1.0 + min_distance)

        lane_change_reward = 0.0  # 初始化变道奖励为0+
        if action in [0, 2]:
            lane_change_reward = -1

        # # 碰撞奖励
        # # 获取被控车辆和其他车辆的信息
        # ego_vehicle = self.vehicle
        # other_vehicles = self.unwrapped.road.vehicles

        # # 检查是否有碰撞
        # collision_with_ego = any(ego_vehicle.check_collision(vehicle) for vehicle in other_vehicles)

        # # 根据碰撞与否给予奖励或惩罚
        # if collision_with_ego:
        #     collision_reward_ego = 1  # 与 ego_vehicle 碰撞，给予奖励
        # else:
        #     collision_reward_ego = - 1  # 未与 ego_vehicle 碰撞，给予惩罚




        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1), 
            "on_road_reward": float(self.vehicle.on_road), # 在车道奖励
            # "sparse_reward": sparse_reward, # 距离奖励
            # "lane_change_reward": lane_change_reward, # 连续变道惩罚
            # "close_lane_change_reward": close_lane_change_reward,
            # "lane_reward": lane_reward  # 新增的车道奖励
        }

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return (self.vehicle.crashed or
                self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]


# class HighwayAgainstEnvFast(HighwayAgainstEnv):
#     """
#     A variant of highway-v0 with faster execution:
#         - lower simulation frequency
#         - fewer vehicles in the scene (and fewer lanes, shorter episode duration)
#         - only check collision of controlled vehicles with others
#     """
#     @classmethod
#     def default_config(cls) -> dict:
#         cfg = super().default_config()
#         cfg.update({
#             "simulation_frequency": 5,
#             "lanes_count": 3,
#             "vehicles_count": 20,
#             "duration": 30,  # [s]
#             "ego_spacing": 1.5,
#         })
#         return cfg

#     def _create_vehicles(self) -> None:
#         super()._create_vehicles()
#         # Disable collision check for uncontrolled vehicles
#         for vehicle in self.road.vehicles:
#             if vehicle not in self.controlled_vehicles:
#                 vehicle.check_collisions = False

