import logging
from functools import cached_property

import numpy as np
import yaml
from dexjoco.tasks import CONFIG_MAPPING
from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots import Robot
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from scipy.spatial.transform import Rotation as R
from typing_extensions import override

from hybrid_insert.integration import state_to_dual_arm_action44

from .config_dexjoco_robot import DexJoCoRobotConfig


def _observation_dict_to_state46(observation: dict) -> np.ndarray:
    values = [float(observation[f"state_{i}"]) for i in range(46)]
    return np.asarray(values, dtype=np.float64)


class DexJoCoRobot(Robot):
    config_class = DexJoCoRobotConfig
    name = "dexjoco_robot"

    def __init__(self, config: DexJoCoRobotConfig):
        self.config = config
        assert config.id is not None, (
            "RobotConfig.id must be specified (as exp_name for CONFIG_MAPPING) for DexJoCo"
        )
        self.exp_name = config.id

        with open(config.config_path, "r") as f:
            cfg = yaml.safe_load(f)
            self.observation_features_cfg = cfg["observation_features"]
            self.action_features_cfg = cfg["action_features"]
            self.single_arm = cfg["single_arm"]
            self.right_arm_action_only = bool(cfg.get("right_arm_action_only", False))
            self.model_env_image_map = cfg.get("model_env_image_map")
        self._left_arm_hold22: np.ndarray | None = None

        self.done = False
        self.success = False
        self._is_connected = False

        self.seed = config.seed
        self.randomize = config.randomize
        self.randomize_dynamics = config.randomize_dynamics
        self.render_mode = config.render_mode
        self.hybrid_insert = None

    @check_if_not_connected
    def reset(self) -> None:
        obs, _ = self.env.reset()
        self.observation = self._process_observation(obs)
        self.done = False
        self.success = False
        if self.hybrid_insert is not None:
            self.hybrid_insert.on_reset(self.env)

    @check_if_already_connected
    @override
    def connect(self, calibrate: bool = True) -> None:
        _ = calibrate

        config = CONFIG_MAPPING[self.exp_name]()
        self.env = config.get_environment(
            policy_mode=True,
            render_mode=self.render_mode,
            seed=self.seed,
            randomize=self.randomize,
            randomize_dynamics=self.randomize_dynamics,
        )

        self._is_connected = True
        logging.info(f"{self.exp_name} simulator created")

    @check_if_not_connected
    @override
    def disconnect(self) -> None:
        self.env.close()
        self._is_connected = False
        logging.info(f"{self.exp_name} simulator closed")

    @cached_property
    @override
    def observation_features(self) -> dict[str, type | tuple]:
        features = {}
        for item in self.observation_features_cfg["state"]:
            if isinstance(item, list):
                for name in item:
                    features[name] = float
            elif isinstance(item, dict):
                assert len(item) == 1, (
                    "Only one state feature is allowed in each dict item"
                )
                name, length = next(iter(item.items()))
                features.update({f"{name}_{i}": float for i in range(length)})
            else:
                raise ValueError("Invalid observation_features config format")

        for name, shape in self.observation_features_cfg["images"].items():
            features[name] = tuple(shape)

        return features

    @cached_property
    @override
    def action_features(self) -> dict[str, type]:
        features = {}
        for name in self.action_features_cfg:
            if isinstance(name, list):
                for n in name:
                    features[n] = float
            elif isinstance(name, dict):
                assert len(name) == 1, (
                    "Only one action feature is allowed in each dict item"
                )
                name, length = next(iter(name.items()))
                features.update({f"{name}_{i}": float for i in range(length)})
            else:
                raise ValueError("Invalid action_features config format")

        return features

    @check_if_not_connected
    @override
    def get_observation(self) -> RobotObservation:
        return self.observation

    def set_left_arm_hold22(self, left22: np.ndarray) -> None:
        hold = np.asarray(left22, dtype=np.float64).reshape(-1).copy()
        if hold.shape[0] != 22:
            raise ValueError(f"left hold must be 22-dim, got {hold.shape[0]}")
        self._left_arm_hold22 = hold

    @check_if_not_connected
    @override
    def send_action(self, action: RobotAction) -> RobotAction:
        action_array = np.array([float(action[k]) for k in self.action_features.keys()])

        if (
            not self.single_arm
            and self.right_arm_action_only
            and action_array.shape[0] == 22
        ):
            if self._left_arm_hold22 is None:
                raise RuntimeError("right_arm_action_only policy requires set_left_arm_hold22()")
            action_array = np.concatenate([action_array, self._left_arm_hold22], dtype=np.float64)

        if self.hybrid_insert is not None and self.hybrid_insert.enabled:
            if not self.hybrid_insert.active:
                self.hybrid_insert.observe(self.env, action_array)
            action_array = self.hybrid_insert.merge(self.env, action_array).astype(np.float64)

        if self.single_arm:
            xyz = action_array[:3]
            rot_vec = action_array[3:6]
            hand = action_array[6:]
            quat = R.from_rotvec(rot_vec).as_quat(scalar_first=True)
            action_array = np.concatenate([xyz, quat, hand])
        else:
            r_xyz = action_array[:3]
            r_rot_vec = action_array[3:6]
            r_hand = action_array[6:22]
            l_xyz = action_array[22:25]
            l_rot_vec = action_array[25:28]
            l_hand = action_array[28:44]

            r_quat = R.from_rotvec(r_rot_vec).as_quat(scalar_first=True)
            l_quat = R.from_rotvec(l_rot_vec).as_quat(scalar_first=True)

            action_array = np.concatenate([
                r_xyz,
                r_quat,
                l_xyz,
                l_quat,
                r_hand,
                l_hand,
            ])

        obs, reward, terminated, truncated, info = self.env.step(action_array)

        self.observation = self._process_observation(obs)
        self.done = bool(terminated)
        self.success = info["succeed"]

        return action

    def stay(self, continue_stay: bool):
        if self.hybrid_insert is not None and self.hybrid_insert.active:
            if continue_stay and hasattr(self, "last_stay_action"):
                hold_action = self.last_stay_action
            else:
                hold_action = state_to_dual_arm_action44(
                    _observation_dict_to_state46(self.observation)
                )
                self.last_stay_action = hold_action
            action_array = self.hybrid_insert.merge(self.env, hold_action).astype(np.float64)
            if self.single_arm:
                xyz = action_array[:3]
                rot_vec = action_array[3:6]
                hand = action_array[6:]
                quat = R.from_rotvec(rot_vec).as_quat(scalar_first=True)
                env_action = np.concatenate([xyz, quat, hand])
            else:
                r_xyz = action_array[:3]
                r_rot_vec = action_array[3:6]
                r_hand = action_array[6:22]
                l_xyz = action_array[22:25]
                l_rot_vec = action_array[25:28]
                l_hand = action_array[28:44]
                r_quat = R.from_rotvec(r_rot_vec).as_quat(scalar_first=True)
                l_quat = R.from_rotvec(l_rot_vec).as_quat(scalar_first=True)
                env_action = np.concatenate([r_xyz, r_quat, l_xyz, l_quat, r_hand, l_hand])
            obs, reward, terminated, truncated, info = self.env.step(env_action)
            self.observation = self._process_observation(obs)
            self.done = bool(terminated)
            self.success = info["succeed"]
            return

        if continue_stay:
            assert hasattr(self, "last_stay_action"), (
                "last_stay_action not found, cannot continue stay"
            )
            stay_action = self.last_stay_action
        else:
            action_list = []
            for key in self.observation_features.keys():
                if self.observation_features[key] is float:
                    action_list.append(float(self.observation[key]))
            stay_action = np.array(action_list)
            self.last_stay_action = stay_action

        obs, reward, terminated, truncated, info = self.env.step(stay_action)
        self.observation = self._process_observation(obs)
        self.done = bool(terminated)
        self.success = info["succeed"]

    @property
    def is_done(self) -> bool:
        return self.done

    @property
    def is_success(self) -> bool:
        return self.success

    def _process_observation(self, obs):
        if self.single_arm:
            arm_state = obs["state"][:7]
            hand_state = obs["state"][7:23]
            all_state = np.concatenate([arm_state, hand_state])
            if self.config.pad_state_dim46:
                all_state = np.concatenate([all_state, np.zeros(46 - len(all_state))])
        else:
            r_arm_state = obs["state"][:7]
            l_arm_state = obs["state"][7:14]
            arm_state = np.concatenate([r_arm_state, l_arm_state])
            r_hand_state = obs["state"][14:30]
            l_hand_state = obs["state"][30:46]
            hand_state = np.concatenate([r_hand_state, l_hand_state])
            all_state = np.concatenate([arm_state, hand_state])

        observations = {}
        state_idx = 0
        for name, dtype in self.observation_features.items():
            if dtype is not float:
                continue
            observations[name] = float(all_state[state_idx])
            state_idx += 1
        assert state_idx == len(all_state)

        for model_img_name in self.observation_features_cfg["images"]:
            if self.model_env_image_map is not None:
                env_image_name = self.model_env_image_map[model_img_name]
            else:
                env_image_name = model_img_name
            observations[model_img_name] = obs[env_image_name]
        return observations

    @property
    @override
    def is_connected(self) -> bool:
        return self._is_connected

    @override
    def calibrate(self) -> None:
        logging.info("Calibration not required")

    @property
    @override
    def is_calibrated(self) -> bool:
        return True

    @override
    def configure(self) -> None:
        pass
