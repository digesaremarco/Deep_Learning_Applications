import torch
import wandb
from networks import save_checkpoint
from common import run_episode, compute_returns

def reinforce(policy, env, run, gamma=0.99, lr=0.02, baseline='std', num_episodes=10):
    """
    A direct, inefficient, and probably buggy implementation of the REINFORCE policy gradient algorithm.
    Checkpoints best model at each iteration to the wandb run directory.

    Args:
        policy: The policy network to be trained.
        env: The environment in which the agent operates.
        run: An object that handles logging and running episodes.
        gamma: The discount factor for future rewards.
        lr: Learning rate for the optimizer.
        baseline: The type of baseline to use ('none', or 'std').
        num_episodes: The number of episodes to train the policy.

    Returns:
        running_rewards: A list of running rewards over episodes.
    """
    # Check for valid baseline (should probably be done elsewhere).
    if baseline not in ['none', 'std']:
        raise ValueError(f'Unknown baseline {baseline}')

    # The only non-vanilla part: we use Adam instead of SGD.
    opt = torch.optim.Adam(policy.parameters(), lr=lr)

    # Track episode rewards in a list.
    running_rewards = [0.0]
    best_return = 0.0

    # The main training loop.
    policy.train()
    for episode in range(num_episodes):
        # New dict for the wandb log for current iteration.
        log = {}

        # Run an episode of the environment, collect everything needed for policy update.
        (observations, actions, log_probs, rewards) = run_episode(env, policy)

        # Compute the discounted reward for every step of the episode. 
        returns = torch.tensor(compute_returns(rewards, gamma), dtype=torch.float32)

        # Keep a running average of total discounted rewards for the whole episode.
        running_rewards.append(0.05 * returns[0].item() + 0.95 * running_rewards[-1])

        # Log some stuff.
        log['episode_length'] = len(returns)
        log['return'] = returns[0]

        # Checkpoint best model.
        if running_rewards[-1] > best_return:
            save_checkpoint('BEST', policy, opt, wandb.run.dir)

        # Basline returns.
        if baseline == 'none':
            base_returns = returns # no baseline
        elif baseline == 'std':
            base_returns = (returns - returns.mean()) / returns.std() # standardized baseline

        # Make an optimization step on the policy network.
        opt.zero_grad()
        policy_loss = (-log_probs * base_returns).mean()
        policy_loss.backward()
        opt.step()

        # Log the current loss and finalize the log for this episode.
        log['policy_loss'] = policy_loss.item()
        run.log(log)

        # Print running reward and (optionally) render an episode after every 100 policy updates.
        if not episode % 100:
            print(f'Running reward @ episode {episode}: {running_rewards[-1]}')

    # Return the running rewards.
    policy.eval()
    return running_rewards
