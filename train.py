"""
train.py
PPO training for the MLP-based multi-agent GridWorld.
Trains fast on CPU (~1-2 min). Saves to ppo_cnn_agent.pth.
Usage: python train.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from env import MultiAgentGridWorldEnv, OBS_DIM
from ppo import ActorCritic

# ─── HYPERPARAMETERS ──────────────────────────────────────────────────────────
GRID_SIZE         = 25
NUM_EPOCHS        = 150
STEPS_PER_EPOCH   = 500
GAMMA             = 0.99
GAE_LAMBDA        = 0.95
CLIP_COEF         = 0.2
ENT_COEF          = 0.05
VF_COEF           = 0.5
LR                = 3e-3
PPO_UPDATE_EPOCHS = 4
CHECKPOINT        = "ppo_cnn_agent.pth"


def train():
    device = torch.device("cpu")
    print(f"[Train] Device: {device}")

    env = MultiAgentGridWorldEnv(
        grid_size=GRID_SIZE, num_green_goals=12,
        num_red_goals=6, num_obstacles=50, max_steps=STEPS_PER_EPOCH
    )
    N = env.num_agents

    model     = ActorCritic(obs_dim=OBS_DIM, act_dim=5, grid_size=GRID_SIZE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, eps=1e-5)

    print(f"[Train] {NUM_EPOCHS} epochs × {STEPS_PER_EPOCH} steps × {N} agents")

    for epoch in range(1, NUM_EPOCHS + 1):
        obs_np, _  = env.reset()
        obs        = torch.tensor(obs_np, dtype=torch.float32).to(device)  # (N, OBS_DIM)

        b_obs      = torch.zeros((STEPS_PER_EPOCH, N, OBS_DIM), device=device)
        b_actions  = torch.zeros((STEPS_PER_EPOCH, N), dtype=torch.long, device=device)
        b_logprobs = torch.zeros((STEPS_PER_EPOCH, N), device=device)
        b_rewards  = torch.zeros((STEPS_PER_EPOCH, N), device=device)
        b_dones    = torch.zeros((STEPS_PER_EPOCH, N), device=device)
        b_values   = torch.zeros((STEPS_PER_EPOCH, N), device=device)

        for t in range(STEPS_PER_EPOCH):
            b_obs[t] = obs
            masks_np = env.get_action_masks()
            masks    = torch.tensor(masks_np, dtype=torch.float32, device=device)
            with torch.no_grad():
                action, logprob, _, value = model.get_action_and_value(obs, masks=masks)
            b_actions[t]  = action
            b_logprobs[t] = logprob
            b_values[t]   = value.flatten()

            next_np, rew_np, _, trunc, _ = env.step(action.cpu().numpy())
            obs        = torch.tensor(next_np, dtype=torch.float32, device=device)
            b_rewards[t] = torch.tensor(rew_np, dtype=torch.float32, device=device)
            b_dones[t]   = torch.tensor(trunc.astype(np.float32), device=device)

        # GAE
        with torch.no_grad():
            masks_np = env.get_action_masks()
            masks    = torch.tensor(masks_np, dtype=torch.float32, device=device)
            _, _, _, next_val = model.get_action_and_value(obs, masks=masks)
            next_val   = next_val.flatten()
            advantages = torch.zeros_like(b_rewards)
            gae        = torch.zeros(N, device=device)
            for t in reversed(range(STEPS_PER_EPOCH)):
                nv  = next_val if t == STEPS_PER_EPOCH - 1 else b_values[t+1]
                nd  = b_dones[t]
                δ   = b_rewards[t] + GAMMA * nv * (1 - nd) - b_values[t]
                gae = δ + GAMMA * GAE_LAMBDA * (1 - nd) * gae
                advantages[t] = gae
            returns = advantages + b_values

        mb_obs  = b_obs.view(-1, OBS_DIM)
        mb_act  = b_actions.view(-1)
        mb_lp   = b_logprobs.view(-1)
        mb_adv  = advantages.view(-1)
        mb_ret  = returns.view(-1)

        mb_adv  = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

        for _ in range(PPO_UPDATE_EPOCHS):
            _, new_lp, entropy, new_val = model.get_action_and_value(mb_obs, mb_act)
            ratio   = (new_lp - mb_lp).exp()
            pg1     = -mb_adv * ratio
            pg2     = -mb_adv * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)
            pg_loss = torch.max(pg1, pg2).mean()
            v_loss  = 0.5 * ((new_val.flatten() - mb_ret) ** 2).mean()
            loss    = pg_loss + VF_COEF * v_loss - ENT_COEF * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

        mean_r = b_rewards.sum(0).mean().item()
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{NUM_EPOCHS}  mean_reward={mean_r:+.2f}")

    torch.save(model.state_dict(), CHECKPOINT)
    print(f"\n[Train] Saved → {CHECKPOINT}")
    print(f"[Train] Final mean reward: {mean_r:+.2f}")
    if mean_r >= 0.8:
        print("[Train] ✓ Benchmark achieved!")
    else:
        print("[Train] Benchmark not yet met — consider more epochs.")


if __name__ == "__main__":
    train()
