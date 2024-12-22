import os
import torch
import numpy as np
import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.config import config_factory
from robomimic.scripts.train import train
from robomimic.utils.log_utils import DataLogger, PrintLogger, log_warning
from robomimic.config.config import Config
import wandb
from datetime import datetime
import sys
import urllib.request
from tqdm import tqdm
from robomimic import DATASET_REGISTRY
from robosuite import load_composite_controller_config
import robosuite as suite
# Available datasets and robots
AVAILABLE_DATASETS = {
    "lift": {
        "description": "Basic object lifting task",
        "versions": ["low_dim.hdf5", "image.hdf5", "ph.hdf5"]
    },
    "can": {
        "description": "Can manipulation task",
        "versions": ["low_dim.hdf5", "image.hdf5", "ph.hdf5"]
    },
    "square": {
        "description": "Square block manipulation",
        "versions": ["low_dim.hdf5", "image.hdf5", "ph.hdf5"]
    },
    "transport": {
        "description": "Object transport task",
        "versions": ["low_dim.hdf5", "image.hdf5", "ph.hdf5"]
    },
    "tool": {
        "description": "Tool manipulation task",
        "versions": ["low_dim.hdf5", "image.hdf5", "ph.hdf5"]
    }
}

AVAILABLE_ROBOTS = {
    "Panda": "Franka Emika Panda - 7-DOF collaborative robot",
    "Sawyer": "Rethink Sawyer - 7-DOF robot",
    "IIWA": "KUKA IIWA - 7-DOF robot",
    "Jaco": "Kinova Jaco - 6-DOF robot",
    "UR5e": "Universal Robots UR5e - 6-DOF robot"
}

# Add after other global variables
WANDB_CONFIG = {
    "project": "robomimic_training",  # Your wandb project name
    "entity": None,                   # Your wandb username/entity
}

class PrintLogger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log_file = open(log_file, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    # Add this method to handle wandb's terminal checks
    def isatty(self):
        return self.terminal.isatty()

def print_available_options():
    """Print available datasets and robots"""
    PrintLogger.print("Available Datasets:", color='green')
    for dataset, info in AVAILABLE_DATASETS.items():
        PrintLogger.print(f"  - {dataset}: {info['description']}")
        PrintLogger.print(f"    Versions: {', '.join(info['versions'])}")
    
    PrintLogger.print("\nAvailable Robots:", color='green')
    for robot, desc in AVAILABLE_ROBOTS.items():
        PrintLogger.print(f"  - {robot}: {desc}")

def create_training_config(
    algo_name="bcq",
    dataset_type="lift",
    robot_name="Panda",
    use_images=False,
    run_name=None
):
    """Create training configuration"""
    config = config_factory(algo_name)
    
    # 使用 with config.unlocked() 来确保可以修改配置
    with config.unlocked():
        # Basic training settings
        config.train.batch_size = 128
        config.train.num_epochs = 500
        
        # Experiment settings
        config.experiment.name = f"{algo_name}_{dataset_type}_{robot_name}"
        config.experiment.validate = True
        
        # Logging settings
        config.experiment.logging.terminal_output_to_txt = True
        config.experiment.logging.log_tb = True
        config.experiment.logging.log_frequency = 100
        
        # Model architecture
        if algo_name == "bc_rnn":
            config.algo.rnn.enabled = True
            config.algo.rnn.horizon = 10
            config.algo.rnn.hidden_dim = 400
            config.algo.rnn.rnn_type = "LSTM"
            config.algo.rnn.num_layers = 2
        
        # Observation space
        if use_images:
            config.observation.modalities.obs.rgb = [
                "agentview_image",
                "robot0_eye_in_hand"
            ]
            config.observation.modalities.obs.low_dim = []
        else:
            config.observation.modalities.obs.low_dim = [
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
                "robot0_joint_pos",
                "robot0_joint_vel",
                "object"
            ]
        
        # Robot configuration
        config.environment.robot_name = robot_name
        config.environment.control_freq = 20
        config.environment.horizon = 400
        
        # 确保所有必需的配置键都存在
        if not hasattr(config.experiment, "validation"):
            config.experiment.validation = Config()
        config.experiment.validation.enabled = True
        config.experiment.validation.frequency = 50  # 每50个epoch验证一次
        
        # 确保存在训练数据路径配置
        if not hasattr(config, "train"):
            config.train = Config()
        config.train.data = None  # 这个会在main函数中设置
        
    return config

def evaluate_policy(
    model_path,
    config,
    num_episodes=10,
    video_path=None,
    video_skip=5,
):
    """
    Evaluate trained policy and optionally save videos
    """
    from robomimic.algorithms.algorithm import Algorithm
    from robomimic.envs.env_base import EnvBase
    
    # Create environment
    env = EnvBase.create(config.environment)
    
    # Load policy
    policy = Algorithm.load(model_path, env=env)
    
    # Evaluate
    PrintLogger.print("Running evaluation episodes...", color='blue')
    videos = []
    stats = []
    
    for ep in range(num_episodes):
        obs = env.reset()
        video_frames = []
        episode_reward = 0
        done = False
        
        while not done:
            # Get action from policy
            action = policy(obs)
            
            # Step environment
            next_obs, reward, done, info = env.step(action)
            episode_reward += reward
            
            # Record video frame if needed
            if video_path is not None and ep < 5:  # Save first 5 episodes
                if len(video_frames) % video_skip == 0:
                    video_frames.append(env.render(mode="rgb_array"))
            
            obs = next_obs
        
        stats.append(episode_reward)
        
        # Save video if frames were collected
        if video_frames:
            video_name = os.path.join(video_path, f"episode_{ep}.mp4")
            save_videos(video_frames, video_name)
            videos.append(video_name)
    
    return videos, stats

def download_dataset(dataset_type, hdf5_type="low_dim"):
    """
    下载数据集到本地
    
    Args:
        dataset_type (str): 数据集类型，如 "lift"
        hdf5_type (str): 数据类型，"low_dim" 或 "image"
    """
    # 创建数据集目录
    dataset_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "datasets",
        dataset_type
    )
    os.makedirs(dataset_dir, exist_ok=True)
    
    # 获取下载链接
    if dataset_type not in DATASET_REGISTRY:
        raise ValueError(f"Dataset type {dataset_type} not found in registry!")
    
    dataset_info = DATASET_REGISTRY[dataset_type]["ph"]  # ph 是默认的数据集类型
    if hdf5_type not in dataset_info:
        raise ValueError(f"HDF5 type {hdf5_type} not found for dataset {dataset_type}!")
    
    url = dataset_info[hdf5_type]["url"]
    filename = os.path.join(dataset_dir, f"{hdf5_type}.hdf5")
    
    # 如果文件已存在，跳过下载
    if os.path.exists(filename):
        print(f"Dataset already exists at {filename}")
        return filename
    
    # 下载数据集
    print(f"Downloading dataset from {url} to {filename}")
    
    def show_progress(block_num, block_size, total_size):
        pbar.update(block_size)
    
    with tqdm(unit='B', unit_scale=True, desc=f"Downloading {dataset_type}") as pbar:
        urllib.request.urlretrieve(url, filename, reporthook=show_progress)
    
    return filename

def main(
    dataset_type="lift",
    robot_name="Panda",
    algo_name="bcq",
    use_images=False,
    project_name="robomimic_training",
    run_name="lift_task_experiment_001"
):
    """
    Training script for imitation learning.
    """
    # Create log directory
    log_dir = os.path.join("logs", f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(log_dir, exist_ok=True)

    # Set up logging
    log_file = os.path.join(log_dir, "log.txt")
    print_logger = PrintLogger(log_file=log_file)
    sys.stdout = print_logger
    sys.stderr = print_logger

    # Create configuration
    config = create_training_config(
        algo_name=algo_name,
        dataset_type=dataset_type,
        robot_name=robot_name,
        use_images=use_images,
        run_name=run_name
    )

    # 设置并下载数据集
    hdf5_type = "image" if use_images else "low_dim"
    try:
        dataset_path = download_dataset(dataset_type, hdf5_type)
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        raise
    
    # Set the dataset path in config
    config.train.data = dataset_path

    # Set output directory
    config.train.output_dir = os.path.join(log_dir, "trained_models")
    
    # Initialize data logger
    data_logger = DataLogger(
        log_dir=log_dir,
        config=config,
        log_tb=True,
    )

    try:
        # 创建控制器配置
        controller_config = load_composite_controller_config(controller="BASIC")
        
        # 添加到环境配置中
        config.env.controller_configs = controller_config

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        train(config, device=device)
        
    except Exception as e:
        log_warning(f"Training failed with error: {str(e)}")
        raise e
    
    finally:
        data_logger.close()

if __name__ == "__main__":
    # Add argument parsing for command line usage
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", type=str, default="lift", choices=DATASET_REGISTRY.keys())
    parser.add_argument("--robot_name", type=str, default="Panda", choices=AVAILABLE_ROBOTS.keys())
    parser.add_argument("--algo_name", type=str, default="bcq")
    parser.add_argument("--use_images", action="store_true")
    parser.add_argument("--run_name", type=str, default="experiment")
    
    args = parser.parse_args()
    
    main(
        dataset_type=args.dataset_type,
        robot_name=args.robot_name,
        algo_name=args.algo_name,
        use_images=args.use_images,
        run_name=args.run_name
    ) 