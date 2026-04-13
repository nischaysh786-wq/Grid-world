"""
env.py
Gymnasium-compatible wrapper with compact flat observations.
Each agent gets a 10-dim vector:
  [row/G, col/G, d_green_r/G, d_green_c/G, d_red_r/G, d_red_c/G,
   obs_up, obs_dn, obs_left, obs_right]
This is fast to compute, trains quickly on CPU, and still captures
all spatially-relevant information the agent needs.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gridworld import GridWorld, NUM_AGENTS, GRID_SIZE, ACTION_DELTAS

OBS_DIM = 10


class MultiAgentGridWorldEnv(gym.Env):
    """Multi-agent env with compact flat observations (MLP-friendly)."""

    def __init__(
        self,
        grid_size:       int = GRID_SIZE,
        num_green_goals: int = 12,
        num_red_goals:   int = 6,
        num_obstacles:   int = 50,
        max_steps:       int = 1000,
    ):
        super().__init__()
        self.g_size    = grid_size
        self.num_green = num_green_goals
        self.num_red   = num_red_goals
        self.num_obs   = num_obstacles
        self.max_steps = max_steps
        self.num_agents = NUM_AGENTS

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )

        self.gw = GridWorld(grid_size, num_green_goals, num_red_goals, num_obstacles)
        self.step_count = 0
        self._prev_cum  = np.zeros(NUM_AGENTS, dtype=np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.gw         = GridWorld(self.g_size, self.num_green, self.num_red, self.num_obs)
        self.step_count = 0
        self._prev_cum  = np.zeros(NUM_AGENTS, dtype=np.float32)
        return self._get_obs(), {}

    def _obs_for(self, agent) -> np.ndarray:
        G  = self.g_size
        r, c = agent.pos

        # Nearest green goal direction
        best_dg = (0.0, 0.0)
        min_d   = G * 2.0
        for gr, gc in self.gw.green_goals:
            d = abs(gr - r) + abs(gc - c)
            if d < min_d:
                min_d   = d
                best_dg = ((gr - r) / G, (gc - c) / G)

        # Nearest red goal direction
        best_dr = (0.0, 0.0)
        min_d   = G * 2.0
        for rr, rc in self.gw.red_goals:
            d = abs(rr - r) + abs(rc - c)
            if d < min_d:
                min_d   = d
                best_dr = ((rr - r) / G, (rc - c) / G)

        # Immediate obstacle detection
        def blocked(dr, dc):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < G and 0 <= nc < G):
                return 1.0
            return 1.0 if (nr, nc) in self.gw.obstacles else 0.0

        return np.array([
            r / G, c / G,
            best_dg[0], best_dg[1],
            best_dr[0], best_dr[1],
            blocked(-1, 0), blocked(1, 0), blocked(0, -1), blocked(0, 1),
        ], dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        return np.stack([self._obs_for(a) for a in self.gw.agents])  # (N, 10)

    def get_action_masks(self) -> np.ndarray:
        """Return (N, 5) binary action mask."""
        masks = np.zeros((NUM_AGENTS, 5), dtype=np.int32)
        for i, a in enumerate(self.gw.agents):
            for act in self.gw.get_valid_actions(a.pos):
                masks[i, act] = 1
        return masks

    # ──────────────────────────────────────────────────────────────────────────
    def step(self, actions: np.ndarray):
        logs, _         = self.gw.step(actions.tolist())
        self.step_count += 1

        obs              = self._get_obs()
        cum              = np.array([a.cumulative_reward for a in self.gw.agents], dtype=np.float32)
        rewards          = cum - self._prev_cum
        self._prev_cum   = cum

        terminated = np.zeros(NUM_AGENTS, dtype=bool)
        truncated  = np.full(NUM_AGENTS, self.step_count >= self.max_steps, dtype=bool)
        info       = {"logs": logs}

        return obs, rewards, terminated, truncated, info
