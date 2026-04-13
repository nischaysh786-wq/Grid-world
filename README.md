# 🧠 GridWorld Reinforcement Learning with PPO

A comprehensive **Reinforcement Learning** project that implements an intelligent agent capable of autonomously navigating a **GridWorld** environment using the **Proximal Policy Optimization (PPO)** algorithm. The project also features a **Flask-based web interface** for interactive visualization and gameplay.

---

## 🚀 Features

- 🎮 **Custom GridWorld Environment**
  - Configurable grid size with obstacles, goals, and penalties.
  - Deterministic and extensible environment design.
  - Suitable for experimentation and educational purposes.

- 🤖 **PPO Reinforcement Learning Agent**
  - Implementation of the **Proximal Policy Optimization (PPO)** algorithm.
  - Supports both standard and CNN-based policies.
  - Includes pre-trained models for quick testing and demonstration.

- 🌐 **Interactive Web Interface**
  - Built using **Flask**.
  - Visualizes agent movements in real time.
  - User-friendly browser-based interaction.

- 📊 **Training & Replay**
  - Complete training pipeline for developing intelligent agents.
  - Replay functionality for analyzing agent behavior.

---

## 📁 Project Structure
Grid-world/
│
├── app.py # Flask web application
├── env.py / gridworld.py # Environment logic
├── ppo.py # PPO algorithm implementation
├── train.py # Script to train the PPO agent
├── play.py # Run the trained agent
├── ppo_agent.pth # Pre-trained PPO model
├── ppo_cnn_agent.pth # CNN-based PPO model
├── replay.json # Stored gameplay or training replay
├── templates/
│ └── index.html # Web interface template
├── requirements.txt # Python dependencies
└── README.md # Project documentation

---

## 🛠️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/nischaysh786-wq/Grid-world.git
cd Grid-world
2️⃣ Create a Virtual Environment
python3 -m venv venv
source venv/bin/activate      # On macOS/Linux
# venv\Scripts\activate       # On Windows
3️⃣ Install Dependencies
pip install -r requirements.txt
If requirements.txt is not available, generate it using:
pip freeze > requirements.txt
🧠 Training the Agent
Train a new PPO agent from scratch:
python train.py
The trained model weights will be saved as .pth files.
🎮 Running the Agent (Command Line)
To run the trained agent in the GridWorld environment:
python play.py
🌐 Running the Web Application
Start the Flask-based web interface:
python app.py
Then open your browser and navigate to:
http://localhost:8000
This interface allows you to visualize the GridWorld and observe the agent’s behavior in real time.
📊 Model Files
File	Description
ppo_agent.pth	Pre-trained PPO model
ppo_cnn_agent.pth	CNN-based PPO model
replay.json	Stored gameplay or training replay
🧩 Technologies Used
Python 3.9+
PyTorch – Deep learning framework
Flask – Web application framework
NumPy – Numerical computations
HTML/CSS/JavaScript – Frontend visualization
Reinforcement Learning (PPO) – Autonomous agent training
📈 Use Cases
Academic and educational demonstrations of reinforcement learning.
Research and experimentation with PPO algorithms.
Visualization of intelligent agent behavior in grid-based environments.
AI and game development projects.
🧹 .gitignore Recommendations
Ensure the following entries exist in your .gitignore:
venv/
env/
.venv/
__pycache__/
*.py[cod]
.DS_Store
*.log
.vscode/
.idea/
*.pth
*.pt
🤝 Contributing
Contributions are welcome! If you’d like to improve this project:
Fork the repository.
Create a new branch:
git checkout -b feature/YourFeature
Commit your changes:
git commit -m "Add your feature"
Push to the branch:
git push origin feature/YourFeature
Open a Pull Request.
📄 License
This project is licensed under the MIT License. Feel free to use and modify it for academic and commercial purposes.
👨‍💻 Author
Nischay
GitHub: https://github.com/nischaysh786-wq
