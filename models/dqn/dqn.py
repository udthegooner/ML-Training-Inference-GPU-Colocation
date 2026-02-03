# Ignore UserWarnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import argparse, random, numpy as np, torch, time, os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.buffers import ReplayBuffer

from core.performanceIterator import PerformanceIterator

# --- Setup & Config ---
DEVICE = 'cpu'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# --- Model Architecture ---

class QNetwork(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 64, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(512 * 3 * 3, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, envs.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x / 255.0)

# --- RL Helpers ---

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env.action_space.seed(seed)
        return env
    return thunk

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

# --- Data Processing (Environment Iterator) ---

def env_generator(envs, model, device, mode, seed, epsilon_info=None):
    """
    Yields data batches for the runBenchmark loop.
    In RL, 'data' consists of the current observation and actions.
    """
    obs, _ = envs.reset(seed=seed)
    while True:
        if mode == 'training':
            epsilon = linear_schedule(epsilon_info['start'], epsilon_info['end'], 
                                     epsilon_info['duration'], epsilon_info['step'])
            epsilon_info['step'] += 1
            if random.random() < epsilon:
                actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            else:
                q_values = model(torch.Tensor(obs).to(device))
                actions = torch.argmax(q_values, dim=1).cpu().numpy()
        else:
            # Inference: Pure Greedy
            q_values = model(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
            
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        yield obs, next_obs, actions, rewards, terminations, truncations, infos
        obs = next_obs

# --- Benchmarking ---

def runBenchmark(mode, q_network, iterator, numSteps, target_network=None, rb=None, optimizer=None, args=None):
    q_network.to(DEVICE)
    if target_network: target_network.to(DEVICE)
    
    q_network.train() if mode == 'training' else q_network.eval()
    
    total_loss = 0.0
    start_time = time.time()
    
    for i in range(numSteps):
        try:
            # RL Batch: obs, next_obs, actions, rewards, terms, truncs, infos
            obs, next_obs, actions, rewards, terminations, truncations, infos = next(iterator)
        except StopIteration:
            break

        if mode == 'training':
            # Handle Replay Buffer
            real_next_obs = next_obs.copy()
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs[idx] = infos["final_observation"][idx]
            rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

            # Training Logic
            if i > args.learning_starts and i % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Update Target Network
            if i % args.target_network_frequency == 0:
                for target_param, q_param in zip(target_network.parameters(), q_network.parameters()):
                    target_param.data.copy_(args.tau * q_param.data + (1.0 - args.tau) * target_param.data)

        if mode == 'training' and i % 500 == 0 and i > 0:
            avg_loss = total_loss / (i / args.train_frequency) if i > args.learning_starts else 0
            print(f'Step: {i}, {mode} loss: {avg_loss:.4f}')

    total_time = time.time() - start_time
    print(f'Total {mode} time for {numSteps} steps: {total_time:.2f} seconds')
    print(f'Average {mode} time per step: {total_time/numSteps:.4f} seconds')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DQN Atari Benchmark Workload')
    parser.add_argument('--gpuIdx', type=int, default=0, help='Index of GPU to use')
    parser.add_argument('--alpha', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_steps', type=int, default=10000, help='Total timesteps')
    parser.add_argument('--job_type', choices=['training', 'inference'], required=True)
    parser.add_argument('--log_file', type=str, default='dqn.log')
    parser.add_argument('--enable_perf_log', action='store_true')
    parser.add_argument('--model_path', type=str, default=None)
    
    # DQN Specific
    parser.add_argument('--env_id', type=str, default="BreakoutNoFrameskip-v4")
    parser.add_argument('--buffer_size', type=int, default=100000)
    parser.add_argument('--learning_starts', type=int, default=1000)
    parser.add_argument('--train_frequency', type=int, default=4)
    parser.add_argument('--target_network_frequency', type=int, default=1000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=1.0)
    
    args = parser.parse_args()

    DEVICE = torch.device(f'cuda:{args.gpuIdx}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    seed = 42
    set_seed(seed)

    # Env Setup
    run_name = f"{args.env_id}__{int(time.time())}"
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, seed, 0, False, run_name)])
    # Model Setup
    q_network = QNetwork(envs).to(DEVICE)
    
    # TODO: check these params from main.py
    epsilon_info = {'start': 1.0, 'end': 0.01, 'duration': 0.1 * args.num_steps, 'step': 0}
    
    base_iterator = env_generator(envs, q_network, DEVICE, args.job_type, seed, epsilon_info)

    if args.enable_perf_log:
        iterator = PerformanceIterator(base_iterator, None, None, None, args.log_file)
    else:
        iterator = base_iterator

    if args.job_type == 'training':
        target_network = QNetwork(envs).to(DEVICE)
        target_network.load_state_dict(q_network.state_dict())
        optimizer = optim.Adam(q_network.parameters(), lr=args.alpha)
        rb = ReplayBuffer(args.buffer_size, envs.single_observation_space, envs.single_action_space, DEVICE, optimize_memory_usage=True)
        
        runBenchmark('training', q_network, iterator, args.num_steps, target_network, rb, optimizer, args)

        if args.model_path:
            torch.save(q_network.state_dict(), args.model_path)
            print(f"Model saved to {args.model_path}")
    else:
        if args.model_path:
            q_network.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
        runBenchmark('inference', q_network, iterator, args.num_steps, args=args)

    envs.close()