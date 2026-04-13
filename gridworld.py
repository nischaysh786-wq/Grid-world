"""
gridworld.py
Core multi-agent GridWorld environment.
25x25 grid, 9 agents, green/red goals, obstacles, configurable parameters.
"""

import numpy as np
import random
from collections import deque

# ─── DEFAULT CONFIGURATION ────────────────────────────────────────────────────
GRID_SIZE      = 25
NUM_AGENTS     = 9
NUM_GREEN_GOALS = 12
NUM_RED_GOALS   = 6
NUM_OBSTACLES   = 50
MAX_STEPS       = 1000

# ─── ACTIONS ──────────────────────────────────────────────────────────────────
# 0:Stay  1:Up  2:Down  3:Left  4:Right
ACTION_DELTAS = {0:(0,0), 1:(-1,0), 2:(1,0), 3:(0,-1), 4:(0,1)}
ACTION_NAMES  = {0:"STAY", 1:"UP", 2:"DOWN", 3:"LEFT", 4:"RIGHT"}

# ─── AGENT COLORS ─────────────────────────────────────────────────────────────
AGENT_COLOR_NAMES = [
    "Blue", "Yellow", "Cyan", "Magenta",
    "White", "Orange", "Purple", "Pink", "Brown"
]
# Hex colors for web UI
AGENT_HEX_COLORS = [
    "#3498db", "#f1c40f", "#00e5ff", "#e056fd",
    "#ffffff", "#e67e22", "#9b59b6", "#fd79a8", "#c0392b"
]


class Agent:
    """Represents a single autonomous agent in the grid."""

    def __init__(self, agent_id: int, position: tuple):
        self.id         = agent_id
        self.pos        = position
        self.color_name = AGENT_COLOR_NAMES[agent_id - 1]
        self.hex_color  = AGENT_HEX_COLORS[agent_id - 1]
        # All agents target green goals (one-hot: green=1, red=0)
        self.goal_signal: list = [1.0, 0.0]

        # Metrics
        self.cumulative_reward = 0.0
        self.correct_goals     = 0
        self.incorrect_goals   = 0
        self.steps_taken       = 0
        self.last_action       = 0
        self.last_reward       = 0.0
        self.last_action_desc  = "Normal"


class GridWorld:
    """
    25×25 grid environment with:
    - Obstacles (static, impassable)
    - Green goals (+1.0 reward), Red goals (-1.0 penalty)
    - 9 autonomous agents with unique colors
    - Goals respawn after collection, agents never terminate
    """

    def __init__(
        self,
        grid_size:       int = GRID_SIZE,
        num_green_goals: int = NUM_GREEN_GOALS,
        num_red_goals:   int = NUM_RED_GOALS,
        num_obstacles:   int = NUM_OBSTACLES,
    ):
        self.size            = grid_size
        self.num_green_goals = num_green_goals
        self.num_red_goals   = num_red_goals
        self.num_obstacles   = num_obstacles

        self.obstacles:   set = set()
        self.green_goals: set = set()
        self.red_goals:   set = set()
        self.agents:      list = []

        self._initialize_environment()

    # ──────────────────────────────────────────────────────────────────────────
    def _occupied(self) -> set:
        occupied = set(self.obstacles) | self.green_goals | self.red_goals
        occupied |= {a.pos for a in self.agents}
        return occupied

    def _get_empty_pos(self) -> tuple:
        occupied = self._occupied()
        while True:
            r = random.randint(0, self.size - 1)
            c = random.randint(0, self.size - 1)
            if (r, c) not in occupied:
                return (r, c)

    def _initialize_environment(self):
        random.seed(42)
        np.random.seed(42)

        for _ in range(self.num_obstacles):
            self.obstacles.add(self._get_empty_pos())

        for i in range(1, NUM_AGENTS + 1):
            self.agents.append(Agent(i, self._get_empty_pos()))

        for _ in range(self.num_green_goals):
            self.green_goals.add(self._get_empty_pos())

        for _ in range(self.num_red_goals):
            self.red_goals.add(self._get_empty_pos())

    # ──────────────────────────────────────────────────────────────────────────
    def reset(self):
        self.obstacles   = set()
        self.green_goals = set()
        self.red_goals   = set()
        self.agents      = []
        self._initialize_environment()

    # ──────────────────────────────────────────────────────────────────────────
    def get_valid_actions(self, pos: tuple) -> list:
        """Return list of valid action indices from *pos*."""
        valid = [0]  # Stay always valid
        for a in [1, 2, 3, 4]:
            dr, dc = ACTION_DELTAS[a]
            nr, nc = pos[0] + dr, pos[1] + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                if (nr, nc) not in self.obstacles:
                    valid.append(a)
        return valid

    # ──────────────────────────────────────────────────────────────────────────
    def get_action_mask(self, pos: tuple) -> list:
        """Return binary mask of length 5 (1=valid, 0=invalid)."""
        valid = self.get_valid_actions(pos)
        return [1 if a in valid else 0 for a in range(5)]

    # ──────────────────────────────────────────────────────────────────────────
    def rule_based_policy(self, agent: Agent) -> int:
        """BFS shortest path to nearest green goal (fallback)."""
        queue   = deque([(agent.pos, [])])
        visited = {agent.pos}

        while queue:
            curr, path = queue.popleft()
            if curr in self.green_goals:
                return path[0] if path else 0
            for a in [1, 2, 3, 4]:
                dr, dc = ACTION_DELTAS[a]
                nb = (curr[0]+dr, curr[1]+dc)
                if (0 <= nb[0] < self.size and 0 <= nb[1] < self.size
                        and nb not in self.obstacles
                        and nb not in self.red_goals
                        and nb not in visited):
                    visited.add(nb)
                    queue.append((nb, path + [a]))

        return random.choice(self.get_valid_actions(agent.pos))

    # ──────────────────────────────────────────────────────────────────────────
    def step(self, actions: list = None) -> tuple:
        """
        Execute one step for all agents.

        Args:
            actions: list[int] length NUM_AGENTS. If None, uses rule_based_policy.

        Returns:
            (logs: list[dict], step_rewards: list[float])
        """
        logs = []
        step_rewards = []

        for i, agent in enumerate(self.agents):
            agent.steps_taken += 1
            action = actions[i] if actions is not None else self.rule_based_policy(agent)

            # Apply action masking – silently stay if invalid
            dr, dc = ACTION_DELTAS[action]
            nr, nc = agent.pos[0] + dr, agent.pos[1] + dc
            if 0 <= nr < self.size and 0 <= nc < self.size and (nr, nc) not in self.obstacles:
                agent.pos = (nr, nc)
            else:
                action = 0

            # Rewards
            reward      = -0.01
            action_desc = "Normal"

            if agent.pos in self.green_goals:
                reward      += 1.0
                action_desc  = "Correct Goal"
                agent.correct_goals += 1
                self.green_goals.discard(agent.pos)
                self.green_goals.add(self._get_empty_pos())

            elif agent.pos in self.red_goals:
                reward      += -1.0
                action_desc  = "Incorrect Goal"
                agent.incorrect_goals += 1
                self.red_goals.discard(agent.pos)
                self.red_goals.add(self._get_empty_pos())
                # Reposition agent away from penalty tile
                agent.pos = self._get_empty_pos()

            agent.cumulative_reward  += reward
            agent.last_action         = action
            agent.last_reward         = reward
            agent.last_action_desc    = action_desc

            step_rewards.append(reward)
            logs.append({
                "agent_id":    agent.id,
                "color":       agent.color_name,
                "hex_color":   agent.hex_color,
                "action":      ACTION_NAMES[action],
                "pos":         agent.pos,
                "reward":      round(reward, 2),
                "action_desc": action_desc,
            })

        return logs, step_rewards

    # ──────────────────────────────────────────────────────────────────────────
    def get_state_dict(self) -> dict:
        """Return full serializable state for the web UI."""
        return {
            "grid_size":   self.size,
            "obstacles":   list(self.obstacles),
            "green_goals": list(self.green_goals),
            "red_goals":   list(self.red_goals),
            "agents": [
                {
                    "id":           a.id,
                    "color":        a.color_name,
                    "hex":          a.hex_color,
                    "pos":          list(a.pos),
                    "goal_signal":  a.goal_signal,
                    "cum_reward":   round(a.cumulative_reward, 2),
                    "correct":      a.correct_goals,
                    "incorrect":    a.incorrect_goals,
                    "steps":        a.steps_taken,
                    "last_action":  ACTION_NAMES[a.last_action],
                    "last_reward":  round(a.last_reward, 2),
                    "action_desc":  a.last_action_desc,
                }
                for a in self.agents
            ],
        }
