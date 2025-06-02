import argparse
import wandb
import gymnasium
from networks import PolicyNet, ValueNet
from reinforce import reinforce
from common import run_episode

def parse_args():
    """The argument parser for the main training script."""
    parser = argparse.ArgumentParser(description='A script implementing REINFORCE on the Cartpole environment.')
    parser.add_argument('--project', type=str, default='DLA2025-Cartpole', help='Wandb project to log to.')
    parser.add_argument('--baseline', type=str, default='none', help='Baseline to use (none, std, val_net)')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor for future rewards')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--visualize', action='store_true', help='Visualize final agent')
    parser.set_defaults(visualize=True)
    args = parser.parse_args()
    return args


# Main entry point.
if __name__ == "__main__":
    # Get command line arguments.
    args = parse_args()

    # Initialize wandb with our configuration parameters.
    run = wandb.init(
        project=args.project,
        config={
            'learning_rate': args.lr,
            'baseline': args.baseline,
            'gamma': args.gamma,
            'num_episodes': args.episodes
        }
    )


    env_name = "LunarLander-v3"  # Default environment

    if env_name == "CartPole-v1":
        env = gymnasium.make('CartPole-v1') # Cartpole environment
        policy = PolicyNet(env)
        value_net = ValueNet(env)
    else:
        env = gymnasium.make('LunarLander-v3') # Lunar Lander environment
        args.episodes = 2000 # More episodes
        policy = PolicyNet(env, n_hidden=2, width=256) # More complex policy network
        value_net = ValueNet(env, n_hidden=2, width=256) # More complex value network


    # Train the agent.
    reinforce(policy, value_net, env, run, lr=args.lr, baseline='val_net', num_episodes=args.episodes, gamma=args.gamma)

    # And optionally run the final agent for a few episodes.
    if args.visualize:
        if env_name == "CartPole-v1":
            env_render = gymnasium.make('CartPole-v1', render_mode='human')
        else:
            env_render = gymnasium.make('LunarLander-v3', render_mode='human')
        for _ in range(10):
            run_episode(env_render, policy)

        # Close the visualization environment.
        env_render.close()

    # Close the Cartpole environment and finish the wandb run.
    env.close()
    run.finish()
