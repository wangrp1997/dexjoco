import os
import time
import numpy as np
import mujoco
import mujoco.viewer

import onnx
from onnx import helper, TensorProto
from onnx import StringStringEntryProto
from tqdm import tqdm

from track_mj.envs.g1_tracking.g1_tracking_constants import FEET_ALL_SITES, DEFAULT_QPOS


def recalculate_traj_angular_velocity(qpos: np.ndarray, qvel: np.ndarray, frequency: float):
    def quat_mul_angle_axis(q1, q2):
        w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        s = 2 * (w**2) - 1
        angle = np.arccos(np.clip(s, -1, 1))
        axis = np.stack([x, y, z], axis=1)
        axis /= np.linalg.norm(axis, axis=-1, keepdims=True).clip(min=1e-9)
        return angle, axis

    freejoint_quat = qpos[:, 3:7]
    freejoint_quat_inv = np.concatenate([freejoint_quat[:, :1], -freejoint_quat[:, 1:]], axis=1)
    angle, axis = quat_mul_angle_axis(freejoint_quat_inv[:-1], freejoint_quat[1:])
    freejoint_angvel = axis * angle[..., np.newaxis] * frequency
    qvel[:-1, 3:6] = freejoint_angvel

    return qvel


def recalculate_traj_linear_velocity(qpos: np.ndarray, qvel: np.ndarray, frequency: float):
    linear_vel = (qpos[1:, :3] - qpos[:-1, :3]) * frequency

    zero_pad = np.zeros((1, 3), dtype=linear_vel.dtype)
    linear_vel = np.concatenate([linear_vel, zero_pad], axis=0)

    qvel[:, :3] = linear_vel

    return qvel


def recalculate_traj_joint_velocity(qpos: np.ndarray, qvel: np.ndarray, frequency: float):
    joint_pos = qpos[:, 7:]
    joint_vel = (joint_pos[1:] - joint_pos[:-1]) * frequency
    qvel[:-1, 6:] = joint_vel

    return qvel


def npz2onnx2(mocap_path: str, qpos_output_path: str):
    data = np.load(mocap_path)
    qpos = data["qpos"]
    qvel = data["qvel"]

    interpolate_interval = 50

    new_qpos = np.zeros(
        (qpos.shape[0] + interpolate_interval * 2, qpos.shape[1]),
        dtype=qpos.dtype,
    )
    new_qvel = np.zeros(
        (qvel.shape[0] + interpolate_interval * 2, qvel.shape[1]),
        dtype=qvel.dtype,
    )
    new_qpos[:interpolate_interval] = np.linspace(DEFAULT_QPOS, qpos[0], interpolate_interval)
    new_qvel[:interpolate_interval] = np.linspace(np.zeros(qvel.shape[1]), qvel[0], interpolate_interval)
    new_qpos[qpos.shape[0] + interpolate_interval :] = np.linspace(qpos[-1], DEFAULT_QPOS, interpolate_interval)
    new_qvel[qpos.shape[0] + interpolate_interval :] = np.linspace(
        qvel[-1], np.zeros(qvel.shape[1]), interpolate_interval
    )

    new_qpos[:interpolate_interval, :7] = qpos[0, :7]
    new_qvel[:interpolate_interval, :6] = qvel[0, :6]
    new_qpos[qpos.shape[0] + interpolate_interval :, :7] = qpos[-1, :7]
    new_qvel[qpos.shape[0] + interpolate_interval :, :6] = qvel[-1, :6]

    new_qpos[interpolate_interval : interpolate_interval + qpos.shape[0]] = qpos
    new_qvel[interpolate_interval : interpolate_interval + qvel.shape[0]] = qvel

    qpos = new_qpos
    qvel = new_qvel

    qvel = recalculate_traj_angular_velocity(qpos, qvel, 50)
    qvel = recalculate_traj_linear_velocity(qpos, qvel, 50)
    qvel = recalculate_traj_joint_velocity(qpos, qvel, 50)

    model = mujoco.MjModel.from_xml_path("storage/assets/unitree_g1/scene_mjx_flat_terrain.xml")
    data = mujoco.MjData(model)
    feet_all_site_id = np.array([model.site(name).id for name in FEET_ALL_SITES])

    feet_height = []
    root_height = []
    for i in tqdm(range(qpos.shape[0])):
        data.qpos[:] = qpos[i].copy()
        data.qvel[:] = qvel[i].copy()
        mujoco.mj_forward(model, data)
        time.sleep(0.01)
        feet_height.append(data.site_xpos[feet_all_site_id, 2].copy())
        root_height.append(data.qpos[2:3].copy())
    feet_height = np.array(feet_height)
    root_height = np.array(root_height)

    def create_constant_model(data_list, output_name_list):
        assert isinstance(data_list, list) and isinstance(output_name_list, list), "input must be list"
        assert len(data_list) == len(output_name_list), "data_list and output_name_list must have the same length"

        nodes = []
        outputs = []
        for i, (data, output_name) in enumerate(zip(data_list, output_name_list)):
            tensor = helper.make_tensor(
                name=output_name,
                data_type=TensorProto.FLOAT,
                dims=data.shape,
                vals=data.flatten().tolist(),
            )
            output = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, data.shape)
            constant_node = helper.make_node(
                "Constant", inputs=[], outputs=[output_name], name=f"Constant_Node_{i}", value=tensor
            )
            nodes.append(constant_node)
            outputs.append(output)

        graph = helper.make_graph(nodes=nodes, name="ConstantOutputModel", inputs=[], outputs=outputs)
        model = helper.make_model(graph, producer_name="npz2onnx", opset_imports=[helper.make_opsetid("", 13)])

        if len(data_list) > 0:
            model.metadata_props.append(StringStringEntryProto(key="total_steps", value=str(data_list[0].shape[0])))

        return model
    
    onnx_data_model = create_constant_model(
        [qpos, qvel, feet_height, root_height], ["qpos", "qvel", "feet_height", "root_height"]
    )

    onnx.save(onnx_data_model, qpos_output_path)
    print(f"ONNX model saved to {qpos_output_path}")


if __name__ == "__main__":
    input_dir = "storage/data/mocap/lafan1/UnitreeG1"
    output_dir = "deploy/storage/data"

    os.makedirs(output_dir, exist_ok=True)

    motion_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".npz")])

    if len(motion_files) == 0:
        print(f"No npz motion files found in {input_dir}")
    else:
        print(f"Found {len(motion_files)} motions in {input_dir}")

    for motion_file in motion_files:
        mocap_path = os.path.join(input_dir, motion_file)
        motion_name = os.path.splitext(motion_file)[0]
        motion_output_dir = os.path.join(output_dir, motion_name)
        os.makedirs(motion_output_dir, exist_ok=True)
        qpos_output_path = os.path.join(motion_output_dir, "ref_data.onnx")
        npz2onnx2(mocap_path, qpos_output_path)
