import torch
import wandb
from networks import save_checkpoint
from common import run_episode, compute_returns

def reinforce(policy, value_net, env, run, gamma=0.99, lr=0.02, baseline='std', num_episodes=10):
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
    if baseline not in ['none', 'std', 'val_net']:
        raise ValueError(f'Unknown baseline {baseline}')

    # The only non-vanilla part: we use Adam instead of SGD.
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    #opt_value = torch.optim.Adam(value_net.parameters(), lr=lr)
    opt_value = torch.optim.Adam(value_net.parameters(), lr=lr) if baseline == 'val_net' else None


    # Track episode rewards in a list.
    running_rewards = [0.0]
    best_return = 0.0

    # The main training loop.
    policy.train()
    if value_net:
        value_net.train()
    #value_net.train()
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
        log['episode_length'] = len(returns) # length of the episode (i.e., number of steps), bigger value means longer episode
        log['return'] = returns[0] # return indicates the total discounted reward for the episode, big value means good

        # Checkpoint best model.
        if running_rewards[-1] > best_return:
            save_checkpoint('BEST', policy, opt, wandb.run.dir)

        # Baseline returns.
        if baseline == 'none':
            base_returns = returns # no baseline
        elif baseline == 'std':
            base_returns = (returns - returns.mean()) / returns.std() # standardized baseline
        elif baseline == 'val_net':
            with torch.no_grad():
                value_estimates = value_net(torch.stack(observations)).squeeze()
                base_returns = returns - value_estimates

            # Train value network to fit returns
            opt_value.zero_grad()
            value_estimates_train = value_net(torch.stack(observations)).squeeze()
            value_loss = torch.nn.functional.mse_loss(value_estimates_train, returns) # MSE loss
            value_loss.backward()
            opt_value.step()
            log['value_loss'] = value_loss.item()

        '''elif baseline == 'val_net':
            #base_returns = value_net(torch.stack(observations)).squeeze() # value network baseline
            #with torch.no_grad():
            baseline_values = value_net(torch.stack(observations)).squeeze()
            base_returns = returns - baseline_values'''

        # Make an optimization step on the policy network.
        opt.zero_grad()
        policy_loss = (-log_probs * base_returns).mean()
        policy_loss.backward()
        opt.step()
        # Make an optimization step on the value network.
        #opt_value.zero_grad()
        #value_loss = (returns - value_net(torch.stack(observations)).squeeze()).pow(2).mean() # MSE loss
        #value_loss.backward()
        #opt_value.step()

        # Log the current loss and finalize the log for this episode.
        log['policy_loss'] = policy_loss.item()
        '''if baseline == 'val_net':
            log['value_loss'] = value_loss.item()'''
        run.log(log)

        # Print running reward and (optionally) render an episode after every 100 policy updates.
        if not episode % 100:
            print(f'Running reward @ episode {episode}: {running_rewards[-1]}')

    # Return the running rewards.
    policy.eval()
    if value_net:
        value_net.eval()

    return running_rewards
