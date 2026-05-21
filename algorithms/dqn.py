"""
algorithms/dqn.py
─────────────────────────────────────────────────────────────────────────────
FIX A5: File was floating as dqn_controller.py / dqn.py in the root.
rl_trainer.py imports from algorithms.dqn — this file lives here.
"""
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
        )

    def forward(self, x):
        return self.network(x)


class ReplayBuffer:

    def __init__(self, capacity=2000):
        self.buffer   = []
        self.capacity = capacity
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class RLMetaController:

    def __init__(self, state_dim=4, action_dim=4):
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.device     = torch.device("cpu")

        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.memory    = ReplayBuffer()

        self.batch_size    = 32
        self.gamma         = 0.95
        self.epsilon       = 1.0
        self.epsilon_min   = 0.05
        self.epsilon_decay = 0.995

        self.weights_path    = os.path.expanduser("~/thesis/rl/dqn_weights.pth")
        self.prev_avg_speed  = 0.0
        self.load_checkpoint()

    def _compute_reward(self, traci_handle):
        """
        Composite reward signal for E³-Hybrid DQN meta-controller.
        Three components: speed improvement, ERT improvement, stuck-vehicle reduction.
        """
        vehicles = traci_handle.vehicle.getIDList()
        avg_speed = (
            sum(traci_handle.vehicle.getSpeed(v) for v in vehicles)
            / max(len(vehicles), 1)
        )
        speed_reward = avg_speed - self.prev_avg_speed
        self.prev_avg_speed = avg_speed

        ert_reward = 0.0
        emergency_vehs = [
            v for v in vehicles
            if traci_handle.vehicle.getTypeID(v) == "emergency"
        ]
        if emergency_vehs:
            avg_ert = (
                sum(traci_handle.vehicle.getAccumulatedWaitingTime(v)
                    for v in emergency_vehs)
                / len(emergency_vehs)
            )
            ert_reward = -avg_ert / 100.0

        stuck_threshold_s = 120.0
        stuck_count = sum(
            1 for v in vehicles
            if traci_handle.vehicle.getWaitingTime(v) > stuck_threshold_s
        )
        stuck_ratio   = stuck_count / max(len(vehicles), 1)
        stuck_penalty = -stuck_ratio * 2.0

        return (0.5 * speed_reward
                + 0.3 * ert_reward
                + 0.2 * stuck_penalty)

    def select_action(self, state, freeze_policy=False):
        if not freeze_policy and random.random() <= self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.policy_net(t).argmax().item()

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = (
            zip(*transitions)
        )

        state_t      = torch.FloatTensor(batch_state).to(self.device)
        action_t     = torch.LongTensor(batch_action).unsqueeze(1).to(self.device)
        reward_t     = torch.FloatTensor(batch_reward).unsqueeze(1).to(self.device)
        next_state_t = torch.FloatTensor(batch_next_state).to(self.device)
        done_t       = torch.FloatTensor(batch_done).unsqueeze(1).to(self.device)

        current_q = self.policy_net(state_t).gather(1, action_t)

        with torch.no_grad():
            max_next_q = self.target_net(next_state_t).max(1)[0].unsqueeze(1)
            target_q   = reward_t + (self.gamma * max_next_q * (1 - done_t))

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_checkpoint(self):
        os.makedirs(os.path.dirname(self.weights_path), exist_ok=True)
        torch.save(
            {"policy_state_dict": self.policy_net.state_dict(),
             "epsilon":           self.epsilon},
            self.weights_path,
        )
        print(f"[RL] Checkpoint saved to {self.weights_path}.")

    def load_checkpoint(self):
        if os.path.exists(self.weights_path):
            ckpt = torch.load(self.weights_path, map_location=self.device)
            self.policy_net.load_state_dict(ckpt["policy_state_dict"])
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.epsilon = ckpt.get("epsilon", self.epsilon_min)
            print(f"[RL] Weights loaded. Epsilon={self.epsilon:.2f}")
        else:
            print("[RL] No checkpoint found. Fresh network initialized.")
