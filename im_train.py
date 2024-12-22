import os
os.environ['MUJOCO_GL'] = 'egl'
import robomimic
from robosuite import load_composite_controller_config
import torch
from torch.utils.data import DataLoader
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.test_utils as TestUtils
import robomimic.utils.train_utils as TrainUtils
from robomimic.utils.dataset import SequenceDataset
from robomimic.config import config_factory
from robomimic.algo import algo_factory
import robosuite as suite

import numpy as np
import torch
from torch.utils.data import DataLoader

from robomimic.algo import RolloutPolicy
from robomimic.utils.train_utils import run_rollout
import imageio

import sys
import os
sys.path.append('./robomimic/')

import os
import json
import h5py
import numpy as np

import robomimic
import robomimic.utils.file_utils as FileUtils
from robomimic.envs.env_base import EnvType
from robomimic.envs.env_robosuite import EnvRobosuite 
from robomimic import DATASET_REGISTRY

WS_DIR = "/home/wanghaoran/robomimic"
download_folder = WS_DIR + "/robomimic_data/"
os.makedirs(download_folder, exist_ok=True)

task = "lift"
dataset_type = "ph"
hdf5_type = "low_dim"

dataset_path = os.path.join(download_folder, "low_dim_v141.hdf5")
if not os.path.exists(dataset_path):
    FileUtils.download_url(
        url=DATASET_REGISTRY[task][dataset_type][hdf5_type]["url"],
        download_dir=download_folder,
    )

controller_config = load_composite_controller_config(controller="BASIC")

# robosuite_env = suite.make(
#     env_name="Lift",
#     robots="Panda",
#     controller_configs=controller_config,
#     has_renderer=False,
#     has_offscreen_renderer=True,
#     use_camera_obs=False,
#     use_object_obs=True,
#     horizon=400,
#     control_freq=20,
# )

env = EnvRobosuite(
    env_name="Lift",
    controller_configs=controller_config,
    has_renderer=False,
    has_offscreen_renderer=False,
    render_camera=None,
    control_freq=20,
    horizon=400,
    use_object_obs=False,
    use_camera_obs=False,
    robots="Panda",
)

def get_example_model(dataset_path, device):
    """
    Use a default config to construct a BC model.
    """

    # default BC config
    config = config_factory(algo_name="bc")

    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(config)

    # read dataset to get some metadata for constructing model
    # all_obs_keys determines what observations we will feed to the policy
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=dataset_path,
        all_obs_keys=sorted((
            "robot0_eef_pos",  # robot end effector position
            "robot0_eef_quat",   # robot end effector rotation (in quaternion)
            "robot0_gripper_qpos",   # parallel gripper joint position
            "object",  # object information
        )),
    )

    # make BC model
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )
    return model

device = TorchUtils.get_torch_device(try_to_use_cuda=True)
model = get_example_model(dataset_path, device=device)

print(model)

def get_data_loader(dataset_path):
    """
    Get a data loader to sample batches of data.
    Args:
        dataset_path (str): path to the dataset hdf5
    """
    dataset = SequenceDataset(
        hdf5_path=dataset_path,
        obs_keys=(                      # observations we want to appear in batches
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "object",
        ),
        dataset_keys=(                  # can optionally specify more keys here if they should appear in batches
            "actions",
            "rewards",
            "dones",
        ),
        load_next_obs=True,
        frame_stack=1,
        seq_length=10,                  # length-10 temporal sequences
        pad_frame_stack=True,
        pad_seq_length=True,            # pad last obs per trajectory to ensure all sequences are sampled
        get_pad_mask=False,
        goal_mode=None,
        hdf5_cache_mode="all",          # cache dataset in memory to avoid repeated file i/o
        hdf5_use_swmr=True,
        hdf5_normalize_obs=False,
        filter_by_attribute=None,       # can optionally provide a filter key here
    )
    print("\n============= Created Dataset =============")
    print(dataset)
    print("")

    data_loader = DataLoader(
        dataset=dataset,
        sampler=None,       # no custom sampling logic (uniform sampling)
        batch_size=100,     # batches of size 100
        shuffle=True,
        num_workers=0,
        drop_last=True      # don't provide last batch in dataset pass if it's less than 100 in size
    )
    return data_loader


def run_train_loop(model, data_loader, num_epochs=100, gradient_steps_per_epoch=100):
    """
    Note: this is a stripped down version of @TrainUtils.run_epoch and the train loop
    in the train function in train.py. Logging and evaluation rollouts were removed.
    Args:
        model (Algo instance): instance of Algo class to use for training
        data_loader (torch.utils.data.DataLoader instance): torch DataLoader for
            sampling batches
    """
    model.set_train()

    for epoch in range(1, num_epochs + 1): # epoch numbers start at 1
        data_loader_iter = iter(data_loader)
        losses = []
        for _ in range(gradient_steps_per_epoch):
            try:
                batch = next(data_loader_iter)
            except StopIteration:
                data_loader_iter = iter(data_loader)
                batch = next(data_loader_iter)

            input_batch = model.process_batch_for_training(batch)
            info = model.train_on_batch(batch=input_batch, epoch=epoch, validate=False)
            step_log = model.log_info(info)
            losses.append(step_log["Loss"])

        model.on_epoch_end(epoch)

        print("Train Epoch {}: Loss {}".format(epoch, np.mean(losses)))

data_loader = get_data_loader(dataset_path=dataset_path)
run_train_loop(model=model, data_loader=data_loader, num_epochs=100, gradient_steps_per_epoch=100)

# create simulation environment
env = EnvRobosuite(
    env_name="Lift",
    controller_configs=controller_config,
    has_renderer=False,
    has_offscreen_renderer=True,
    render_camera="frontview",
    control_freq=20,
    horizon=400,
    use_object_obs=True,
    use_camera_obs=True,       
    robots="Panda",
)

from robosuite.utils.binding_utils import MjRenderContextOffscreen

sim = env.env.sim
render_context = MjRenderContextOffscreen(sim, device_id=-1)
sim.add_render_context(render_context)
env.env.sim = sim

model.set_eval()
policy = RolloutPolicy(model)

# create a video writer
video_path = "rollout.mp4"
video_writer = imageio.get_writer(video_path, fps=20)

done = False
total_reward = 0

obs = env.get_observation()
steps = 0
while (not done) and steps < 1000:
    print(f'Evaluating at step {steps}')
    steps += 1
    action = policy(obs)
    obs, reward, done, info = env.step(action)
    total_reward += reward

    frame = env.render(mode="rgb_array", height=480, width=640)
    if frame is not None:
        video_writer.append_data(frame)
video_writer.close()

print(f"Total Reward: {total_reward}")

# from base64 import b64encode

# mp4 = open(video_path, "rb").read()
# data_url = "data:video/mp4;base64," + b64encode(mp4).decode()