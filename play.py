import time
import sys
import torch
import numpy as np
from env import MultiAgentGridWorldEnv
from ppo import ActorCritic
from gridworld import BOLD, RESET, MAX_STEPS, SLEEP_TIME

def main():
    env = MultiAgentGridWorldEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    agent = ActorCritic(10, 5).to(device)
    
    try:
        agent.load_state_dict(torch.load("ppo_agent.pth", map_location=device))
        print("Loaded trained PPO model successfully!")
    except FileNotFoundError:
        print("Model 'ppo_agent.pth' not found. Please run 'python train.py' first.")
        sys.exit(1)
        
    agent.eval()
    obs, _ = env.reset()
    
    import json
    replay_data = {
        "grid_size": env.gw.size,
        "obstacles": list(env.gw.obstacles),
        "steps": []
    }
    
    # Store initial state properly too
    for step in range(1, MAX_STEPS + 1):
        step_data = {
            "agents": [{"id": a.id, "color": a.color_name, "pos": a.pos} for a in env.gw.agents],
            "green_goals": list(env.gw.green_goals),
            "red_goals": list(env.gw.red_goals)
        }
        replay_data["steps"].append(step_data)
        
        obs_tensor = torch.tensor(obs).to(device)
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(obs_tensor)
            
        actions = action.cpu().numpy()
        obs, rewards, terminated, truncated, info = env.step(actions)
        
    # Save replay data
    with open("replay.json", "w") as f:
        json.dump(replay_data, f)
    print("Replay saved to replay.json")
    print("\n" + "="*30)
    print(f"{BOLD}RL SIMULATION COMPLETE{RESET}")
    print("="*30)
    mean_reward = np.mean([a.cumulative_reward for a in env.gw.agents])
    print(f"Final Mean Reward (across {env.num_agents} agents): {mean_reward:+.2f}")

if __name__ == "__main__":
    main()
