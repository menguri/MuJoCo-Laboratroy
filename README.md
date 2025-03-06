# Reinforcement Learning Experiments in MuJoCo

This repository contains reinforcement learning experiments conducted in the MuJoCo environment, implementing various state-of-the-art algorithms:

- **REINFORCE**
- **A2C**
- **PPO**
- **DDPG**
- **SAC**

---

## 🔧 Quick Setup

Clone the repository and install dependencies:

```bash
git clone <repo_url>
cd <repo_name>
pip install -r requirements.txt
```

Make sure you're using **wandb** for experiment tracking.

---

## 🚀 Training

Run training scripts (`.sh` files) with your desired configuration:

```bash
bash sh_folder/<environment>/a2c_run_experiment.sh
```

Example (`half_cheetah` with `A2C`):

```bash
# Navigate to the project root
cd <repo_root>

# Set the desired configuration file
CONFIG_PATH="../configs/half_cheetah/A2C_half_cheetah.yaml"
EXP_NAME="HalfCheetah_A2C_experiment"

python main.py --config $CONFIG_PATH --exp_name $EXP_NAME
```

---

## 🚀 Simulation & Evaluation

After training, you can load and test trained models interactively in Jupyter Notebook using the scripts provided in the `simulation` folder.

---

## 📌 Additional Information

- This project uses **Weights & Biases (wandb)** for experiment tracking. Make sure to create an account and log in (`wandb login`).
- Install required packages with:

```bash
pip install -r requirements.txt
```

---

Feel free to explore and modify experiments as needed!


