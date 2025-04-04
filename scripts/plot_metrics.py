import matplotlib.pyplot as plt
import pandas as pd
import os

from absl import app
from absl import flags
import logging

FLAGS = flags.FLAGS
flags.DEFINE_string('log', None, 'Logged data to be plotted.')
flags.mark_flags_as_required(['log'])

def plot_metrics(log_file, save_dir=None):
    data = pd.read_csv(log_file)
    if not save_dir: save_dir = 'plots'
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(data["Iteration"], data["AverageReward"], label="Average Reward")
    plt.xlabel("Iteration")
    plt.ylabel("Average Reward")
    plt.title("Average Reward Over Iterations")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, "average_reward.png"))
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(data["Iteration"], data["PolicyLoss"], label="Policy Loss", color="orange")
    plt.xlabel("Iteration")
    plt.ylabel("Policy Loss")
    plt.title("Policy Loss Over Iterations")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, "policy_loss.png"))
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(data["Iteration"], data["ValueLoss"], label="Value Loss", color="green")
    plt.xlabel("Iteration")
    plt.ylabel("Value Loss")
    plt.title("Value Loss Over Iterations")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, "value_loss.png"))
    plt.show()

def main(argv):
    plot_metrics(FLAGS.log)

if __name__ == '__main__':
    app.run(main)