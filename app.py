"""
app.py
Flask server for the Multi-Agent GridWorld Simulation.
Serves the web UI at http://localhost:8000.

Endpoints:
  GET  /                  → index.html
  POST /api/reset         → reset environment with optional config
  POST /api/step          → execute N steps, return state + logs
  GET  /api/state         → current state (no step)
  GET  /api/logs/csv      → download full log history as CSV
"""

import csv
import io
import os
import time
import threading
import numpy as np
import torch
from typing import Optional

from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS

from gridworld import GridWorld, GRID_SIZE, NUM_GREEN_GOALS, NUM_RED_GOALS, NUM_OBSTACLES, MAX_STEPS
from env import MultiAgentGridWorldEnv
from ppo import ActorCritic

# ─── APP SETUP ────────────────────────────────────────────────────────────────
app    = Flask(__name__, template_folder="templates")
CORS(app)

CHECKPOINT = "ppo_cnn_agent.pth"

# ─── GLOBAL SIMULATION STATE ──────────────────────────────────────────────────
sim_lock        = threading.Lock()
sim_env: Optional[MultiAgentGridWorldEnv] = None
sim_model: Optional[ActorCritic]           = None
sim_obs: Optional[np.ndarray]              = None
sim_step: int                          = 0
sim_log_history: list                  = []   # list of step-log dicts
sim_config: dict                       = {
    "grid_size":       GRID_SIZE,
    "num_green_goals": NUM_GREEN_GOALS,
    "num_red_goals":   NUM_RED_GOALS,
    "num_obstacles":   NUM_OBSTACLES,
    "max_steps":       MAX_STEPS,
    "use_model":       True,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── HELPERS ──────────────────────────────────────────────────────────────────
def load_model(grid_size: int = GRID_SIZE) -> Optional[ActorCritic]:
    if not os.path.exists(CHECKPOINT):
        return None
    from env import OBS_DIM
    m = ActorCritic(obs_dim=OBS_DIM, act_dim=5, grid_size=grid_size).to(device)
    m.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    m.eval()
    return m


def build_env(cfg: dict) -> tuple:
    env = MultiAgentGridWorldEnv(
        grid_size       = cfg["grid_size"],
        num_green_goals = cfg["num_green_goals"],
        num_red_goals   = cfg["num_red_goals"],
        num_obstacles   = cfg["num_obstacles"],
        max_steps       = cfg["max_steps"],
    )
    obs, _ = env.reset()
    return env, obs


def get_actions(obs: np.ndarray, env: MultiAgentGridWorldEnv) -> np.ndarray:
    """Get actions from MLP model with action masking, or rule-based fallback."""
    if sim_model is not None:
        obs_t  = torch.tensor(obs, dtype=torch.float32).to(device)  # (N, OBS_DIM)
        masks  = torch.tensor(env.get_action_masks(), dtype=torch.float32, device=device)
        with torch.no_grad():
            action, _, _, _ = sim_model.get_action_and_value(obs_t, masks=masks)
        return action.cpu().numpy()
    # Fallback: rule-based BFS
    return np.array([env.gw.rule_based_policy(a) for a in env.gw.agents])


# ─── INITIALISE ON STARTUP ────────────────────────────────────────────────────
def init_simulation():
    global sim_env, sim_model, sim_obs, sim_step, sim_log_history
    sim_env, sim_obs = build_env(sim_config)
    sim_model        = load_model(sim_config["grid_size"])
    sim_step         = 0
    sim_log_history  = []
    if sim_model:
        print(f"[App] Loaded PPO model from {CHECKPOINT}")
    else:
        print("[App] No trained model found – using rule-based BFS policy")


# ─── ROUTES ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/reset", methods=["POST"])
def api_reset():
    global sim_env, sim_model, sim_obs, sim_step, sim_log_history, sim_config
    body = request.get_json(silent=True) or {}
    with sim_lock:
        sim_config.update({k: v for k, v in body.items() if k in sim_config})
        sim_env, sim_obs = build_env(sim_config)
        sim_model        = load_model(sim_config["grid_size"])
        sim_step         = 0
        sim_log_history  = []
    return jsonify({"ok": True, "state": sim_env.gw.get_state_dict(), "step": sim_step})


@app.route("/api/step", methods=["POST"])
def api_step():
    global sim_obs, sim_step, sim_log_history
    body       = request.get_json(silent=True) or {}
    n_steps    = int(body.get("n", 1))
    delay      = float(body.get("delay", 0.0))

    results = []
    with sim_lock:
        for _ in range(n_steps):
            if sim_step >= sim_config["max_steps"]:
                break
            actions            = get_actions(sim_obs, sim_env)
            obs, rewards, _, _, info = sim_env.step(actions)
            sim_obs            = obs
            sim_step          += 1
            logs               = info["logs"]
            sim_log_history.append({"step": sim_step, "logs": logs})
            results.append({"step": sim_step, "logs": logs})
            if delay > 0:
                time.sleep(delay)

    state = sim_env.gw.get_state_dict()
    return jsonify({
        "step":    sim_step,
        "state":   state,
        "results": results,
        "done":    sim_step >= sim_config["max_steps"],
    })


@app.route("/api/state")
def api_state():
    with sim_lock:
        return jsonify({
            "step":  sim_step,
            "state": sim_env.gw.get_state_dict() if sim_env else {},
            "done":  sim_step >= sim_config["max_steps"],
        })


@app.route("/api/logs/csv")
def api_logs_csv():
    """Download all step logs as a CSV file."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Step", "Agent ID", "Color", "Action", "Row", "Col",
                     "Reward", "Status"])
    with sim_lock:
        for entry in sim_log_history:
            step = entry["step"]
            for lg in entry["logs"]:
                writer.writerow([
                    step, lg["agent_id"], lg["color"],
                    lg["action"], lg["pos"][0], lg["pos"][1],
                    lg["reward"], lg["action_desc"],
                ])
    output = buf.getvalue()
    return Response(
        output,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=gridworld_logs.csv"},
    )


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_simulation()
    print("[App] Serving at http://localhost:8000")
    app.run(host="0.0.0.0", port=8000, debug=False)
