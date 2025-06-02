import cv2
import numpy as np


def resize_image(image, intp, resolution=64):
    """Resize an image to the target size using INTER_AREA interpolation."""
    target_size = (resolution, resolution)
    return cv2.resize(image, target_size, interpolation=intp)


def sample_random_robot_pos(size):
    """Sample random robot positions"""
    chunk_size = int(0.75 * size)
    robot_x = np.concatenate(
        [
            np.random.uniform(-0.3, 0.3, chunk_size),
            np.random.uniform(-0.1, 0.25, size - chunk_size),
        ]
    )
    robot_y = np.concatenate(
        [
            np.random.uniform(-0.15, 0.0, chunk_size),
            np.random.uniform(-0.3, -0.15, size - chunk_size),
        ]
    )
    robot_z = np.random.uniform(0.50, 0.575, size)
    return np.array([robot_x, robot_y, robot_z]).T


def sample_random_robot_orn(size):
    """Sample random robot orientations"""
    chunk_size = int(0.8 * size)
    robot_roll = np.concatenate(
        [
            np.random.uniform(3, 3.14, chunk_size),
            np.random.uniform(-3.14, -3, size - chunk_size),
        ]
    )
    robot_pitch = np.random.normal(-0.1, 0.1, size)
    robot_yaw = np.random.normal(1.57, 0.1, size)
    return np.array([robot_roll, robot_pitch, robot_yaw]).T


def sample_random_block_pos(size):
    """Sample random block positions"""
    chunk_size = int(0.75 * size)

    block_x = np.concatenate(
        [
            np.random.uniform(-0.05, 0.3, chunk_size),  # Right to the button
            np.random.uniform(-0.3, -0.2, size - chunk_size),  # Left to the button
        ]
    )
    block_y = np.random.uniform(-0.12, -0.05, size)
    block_z = np.random.uniform(0.46, 0.46, size)
    return np.array([block_x, block_y, block_z]).T


def sample_random_block_orn(size):
    """Sample random block orientations"""
    rand_roll = np.random.choice([-3.14, -1.57, 0, 1.57, 3.14], p=[0.1, 0.1, 0.5, 0.2, 0.1], size=size)
    rand_pitch = np.zeros(size)
    chunk_size = int(0.5 * size)
    rand_yaw = np.concatenate(
        [
            np.random.uniform(-1.75, -1.4, chunk_size),
            np.random.uniform(1.4, 1.75, size - chunk_size),
        ]
    )
    return np.array([rand_roll, rand_pitch, rand_yaw]).T


def sample_random_scene_obs(scene_obs, skill_name, size, rand_block_pos, rand_block_orn):
    # Block positions and orientations
    if "pink" in skill_name:
        scene_obs[:, [-6, -5, -4]] = rand_block_pos
        scene_obs[:, [-3, -2, -1]] = rand_block_orn
    elif "blue" in skill_name:
        scene_obs[:, [-12, -11, -10]] = rand_block_pos
        scene_obs[:, [-9, -8, -7]] = rand_block_orn
    elif "red" in skill_name:
        scene_obs[:, [-18, -17, -16]] = rand_block_pos
        scene_obs[:, [-15, -14, -13]] = rand_block_orn

    # Slider positions
    if "slider" not in skill_name:
        # randomly open/close the slider
        scene_obs[:, 0] = np.random.choice([0, 0.25, 0.28], size=size, p=[0.5, 0.1, 0.4])
    else:
        if "right" in skill_name:
            scene_obs[:, 0] = np.random.choice([0.25, 0.26, 0.28], size=size, p=[0.2, 0.2, 0.6])
        elif "left" in skill_name:
            scene_obs[:, 0] = np.random.choice([0, 0.02], size=size, p=[0.8, 0.2])

    # Drawer positions
    if "drawer" not in skill_name:
        # randomly open/close the drawer
        scene_obs[:, 1] = np.random.choice([0, 0.19, 0.22], size=size, p=[0.2, 0.4, 0.4])
    else:
        if "open" in skill_name:
            scene_obs[:, 1] = 0.0
        elif "close" in skill_name:
            scene_obs[:, 1] = np.random.uniform(0.19, 0.22, size)

    # Light and switch positions
    if "lightbulb" not in skill_name:
        # randomly turn on the lightbulb
        scene_obs[:, 4] = np.random.choice([0, 1], size=size, p=[0.5, 0.5])
        scene_obs[scene_obs[:, 4] == 0, 3] = 0
        scene_obs[scene_obs[:, 4] == 1, 3] = 0.088
    else:
        if "turn_on" in skill_name:
            scene_obs[:, 4] = 0
            scene_obs[:, 3] = 0
        elif "turn_off" in skill_name:
            scene_obs[:, 4] = 1
            scene_obs[:, 3] = 0.088

    # LED
    if "led" not in skill_name:
        # randonmly turn on the led
        scene_obs[:, 5] = np.random.choice([0, 1], size=size, p=[0.5, 0.5])
    else:
        if "turn_on" in skill_name:
            scene_obs[:, 5] = 0
        elif "turn_off" in skill_name:
            scene_obs[:, 5] = 1

    # When the drawer is open, place the blocks inside
    if sum(scene_obs[:, 1]) > 0:
        drawer_x = np.random.uniform(0.15, 0.25, size)
        drawer_y = np.random.uniform(-0.3, -0.25, size)
        drawer_z = np.random.uniform(0.37, 0.37, size)

        drawer_xyz = np.array([drawer_x, drawer_y, drawer_z]).T

        pink_indices = [-6, -5, -4]
        blue_indices = [-12, -11, -10]
        red_indices = [-18, -17, -16]
        option1 = None
        option2 = None

        if "block" in skill_name:
            if "pink" in skill_name:
                option1 = blue_indices
                option2 = red_indices
            elif "blue" in skill_name:
                option1 = pink_indices
                option2 = red_indices
            elif "red" in skill_name:
                option1 = pink_indices
                option2 = blue_indices

            # Randomly choose between the two options
            open_filter = scene_obs[:, 1] > 0
            option1_filter = np.copy(open_filter)
            option2_filter = np.copy(open_filter)
            # Create two filters for the two options and randomly choose between them
            block_filter = np.random.choice([True, False], size=sum(open_filter))
            option1_filter[open_filter] = block_filter
            option2_filter[open_filter] = ~block_filter
            scene_obs[option1_filter, option1[0]] = drawer_xyz[option1_filter, 0]
            scene_obs[option1_filter, option1[1]] = drawer_xyz[option1_filter, 1]
            scene_obs[option1_filter, option1[2]] = drawer_xyz[option1_filter, 2]

            scene_obs[option2_filter, option2[0]] = drawer_xyz[option2_filter, 0]
            scene_obs[option2_filter, option2[1]] = drawer_xyz[option2_filter, 1]
            scene_obs[option2_filter, option2[2]] = drawer_xyz[option2_filter, 2]
        else:
            # If the skill is not block related, we get one extra option
            option1 = pink_indices
            option2 = blue_indices
            option3 = red_indices

            # Randomly choose between the three options
            open_filter = scene_obs[:, 1] > 0
            option1_filter = np.copy(open_filter)
            option2_filter = np.copy(open_filter)
            option3_filter = np.copy(open_filter)
            # Create two filters for the two options and randomly choose between them
            block_filter = np.random.choice([1, 2, 3], size=sum(open_filter))
            option1_filter[open_filter] = block_filter == 1
            option2_filter[open_filter] = block_filter == 2
            option3_filter[open_filter] = block_filter == 3

            scene_obs[option1_filter, option1[0]] = drawer_xyz[option1_filter, 0]
            scene_obs[option1_filter, option1[1]] = drawer_xyz[option1_filter, 1]
            scene_obs[option1_filter, option1[2]] = drawer_xyz[option1_filter, 2]

            scene_obs[option2_filter, option2[0]] = drawer_xyz[option2_filter, 0]
            scene_obs[option2_filter, option2[1]] = drawer_xyz[option2_filter, 1]
            scene_obs[option2_filter, option2[2]] = drawer_xyz[option2_filter, 2]

            scene_obs[option3_filter, option3[0]] = drawer_xyz[option3_filter, 0]
            scene_obs[option3_filter, option3[1]] = drawer_xyz[option3_filter, 1]
            scene_obs[option3_filter, option3[2]] = drawer_xyz[option3_filter, 2]
    return scene_obs


def replace_euler_with_rot6d(euler_to_rot6d, obs, type="robot"):
    """Replace the euler angles in robot and scene obs with rotation_6d"""
    if type == "robot":
        obs = np.concatenate(
            [
                obs[:3],
                euler_to_rot6d.forward(obs[3:6]),
                obs[6:],
            ]
        )
    elif type == "scene":
        # red, blue, pink blocks respectively
        indices = [[9, 10, 11], [15, 16, 17], [21, 22, 23]]
        rotation_6ds = []
        for idx in indices:
            euler_angles = obs[idx]
            rotation_6d = euler_to_rot6d.forward(euler_angles)
            rotation_6ds.append(rotation_6d)
        obs = np.concatenate(
            [
                obs[:9],
                rotation_6ds[0],
                obs[12:15],
                rotation_6ds[1],
                obs[18:21],
                rotation_6ds[2],
            ]
        )
    return obs


def replace_rot6d_with_euler(euler_to_rot6d, obs, type="robot"):
    """Replace the euler angles in robot and scene obs with rotation_6d"""
    if type == "robot":
        obs = np.concatenate(
            [
                obs[:3],
                euler_to_rot6d.inverse(obs[3:9]),
                obs[9:],
            ]
        )
    elif type == "scene":
        # red, blue, pink blocks respectively
        indices = [
            [9, 10, 11, 12, 13, 14],
            [18, 19, 20, 21, 22, 23],
            [27, 28, 29, 30, 31, 32],
        ]
        euler_angles_arr = []
        for idx in indices:
            rotation_6d = obs[idx]
            euler_angles = euler_to_rot6d.inverse(rotation_6d)
            euler_angles_arr.append(euler_angles)
        obs = np.concatenate(
            [
                obs[:9],
                euler_angles_arr[0],
                obs[15:18],
                euler_angles_arr[1],
                obs[24:27],
                euler_angles_arr[2],
            ]
        )
    return obs
