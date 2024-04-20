import numpy as np
import random
import matplotlib.pyplot as plt  
import time
from sklearn.metrics import mean_squared_error  

# This class define the Dynamic Programing agent

class DP_agent(object):

    def solve(self, env):
        """
        Solve a given Maze environment using Dynamic Programming, specifically using Value Iteration
        input: env {Maze object} -- Maze to solve
        output:
          - policy {np.array} -- Optimal policy found to solve the given Maze environment
          - V {np.array} -- Corresponding value function
        """

        # Initialize variables
        policy = np.zeros((env.get_state_size(), env.get_action_size()))
        V = np.zeros(env.get_state_size())
        threshold = 1e-4
        delta = threshold * 1.1

        # Value Iteration
        while delta > threshold:
            delta = 0

            for initial_state in range(env.get_state_size()):
                if not env.get_absorbing()[0, initial_state]:
                    v = V[initial_state]
                    Q = np.zeros(env.get_action_size())

                    # Update Q-values using Bellman equation
                    for dest_state in range(env.get_state_size()):
                        Q += env.get_T()[initial_state, dest_state, :] * (
                            env.get_R()[initial_state, dest_state, :] + env.get_gamma() * V[dest_state])

                    # Update value function
                    V[initial_state] = np.max(Q)
                    delta = max(delta, np.abs(v - V[initial_state]))

        # Policy extraction
        for initial_state in range(env.get_state_size()):
            Q = np.zeros(env.get_action_size())

            # Calculate Q-values using final value function
            for dest_state in range(env.get_state_size()):
                Q += env.get_T()[initial_state, dest_state, :] * (env.get_R()
                                                                  [initial_state, dest_state, :] + env.get_gamma() * V[dest_state])

            # Update policy
            policy[initial_state, np.argmax(Q)] = 1

        return policy, V
        
    def solve_with_threshold(self, env, custom_threshold):
        """
        Solve a given Maze environment using Dynamic Programming, specifically using Value Iteration.
        This version of the function allows a custom convergence threshold.

        """

        start_time = time.time()  # Record the start time

        # Initialize variables
        policy = np.zeros((env.get_state_size(), env.get_action_size()))
        V = np.zeros(env.get_state_size())
        threshold = custom_threshold
        delta = threshold * 1.1

        # Value Iteration
        while delta > threshold:
            delta = 0

            for initial_state in range(env.get_state_size()):
                if not env.get_absorbing()[0, initial_state]:
                    v = V[initial_state]
                    Q = np.zeros(env.get_action_size())

                    # Update Q-values using Bellman equation
                    for dest_state in range(env.get_state_size()):
                        Q += env.get_T()[initial_state, dest_state, :] * (env.get_R()[initial_state, dest_state, :] + env.get_gamma() * V[dest_state])

                    # Update value function
                    V[initial_state] = np.max(Q)
                    delta = max(delta, np.abs(v - V[initial_state]))

        # Policy extraction
        for initial_state in range(env.get_state_size()):
            Q = np.zeros(env.get_action_size())

            # Calculate Q-values using final value function
            for dest_state in range(env.get_state_size()):
                Q += env.get_T()[initial_state, dest_state, :] * (env.get_R()[initial_state, dest_state, :] + env.get_gamma() * V[dest_state])

            # Update policy
            policy[initial_state, np.argmax(Q)] = 1

        elapsed_time = time.time() - start_time  # Calculate the elapsed time
        
        return policy, V, elapsed_time
      
    
    def threshold_testing(self, env, threshold_list):
      """
      Solve a given Maze environment using multiple thresholds and plot the results.
      """

      # Initialize variables to hold results
      errors = []
      times = []

      # Solve using the minimum threshold to get V_gold
      min_threshold = min(threshold_list)
      _, V_gold, _ = self.solve_with_threshold(env, min_threshold)

      for threshold in threshold_list:
          # Solve the environment with the current threshold
          _, V_current, elapsed_time = self.solve_with_threshold(env, threshold)

          # Measure the error between V_current and V_gold using mean_squared_error
          error = mean_squared_error(V_current, V_gold)

          # Append results
          errors.append(error)
          times.append(elapsed_time)

      return errors, times
      
    @staticmethod
    def plot_thresholds(threshold_list, errors, times):

      # Plotting
      fig, ax1 = plt.subplots()

      color = 'tab:red'
      ax1.set_xlabel('Threshold')
      ax1.set_ylabel('Mean Squared Error')
      ax1.plot(threshold_list, errors, color=color)
      ax1.tick_params(axis='y')
      ax1.set_xscale("log")  # Set x-axis to logarithmic scale

      ax2 = ax1.twinx()
      color = 'tab:blue'
      ax2.set_ylabel('Time (s)')
      ax2.plot(threshold_list, times, color=color)
      ax2.tick_params(axis='y')

      # Place legend to the right side of the plot
      lines, labels = ax1.get_legend_handles_labels()
      lines2, labels2 = ax2.get_legend_handles_labels()
      ax2.legend(lines + lines2, labels + labels2, loc='center right')

      fig.tight_layout()
      plt.show()



# This class define the Monte-Carlo agent

class MC_agent(object):

    def solve(self, env):
        """
        Solve a given Maze environment using Monte Carlo learning
        input: env {Maze object} -- Maze to solve
        output:
          - policy {np.array} -- Optimal policy found to solve the given Maze environment
          - values {list of np.array} -- List of successive value functions for each episode
          - total_rewards {list of float} -- Corresponding list of successive total non-discounted sum of reward for each episode
        """

        # Initialization
        Q = np.random.rand(env.get_state_size(), env.get_action_size())
        V = np.zeros(env.get_state_size())
        policy = np.zeros((env.get_state_size(), env.get_action_size()))
        values = [V]
        total_rewards = []
        batch_returns = {}  # To store returns for the batch

        # Parameters
        num_episodes = 1000
        epsilon = 0.6  # Start with a high value of epsilon
        epsilon_min = 0.01  # Minimum allowable value for epsilon
        epsilon_decay = 0.999  # Decay rate for epsilon
        batch_size = 100  # Number of episodes in a batch

        for episode_num in range(num_episodes):
            episode = []
            state = env.reset()[1]
            done = False
            total_reward = 0

            while not done:
                # Epsilon-greedy policy with decaying epsilon
                probs = np.ones(env.get_action_size()) * \
                    epsilon / env.get_action_size()
                best_action = np.argmax(Q[state])
                probs[best_action] += (1.0 - epsilon)
                action = np.random.choice(
                    np.arange(env.get_action_size()), p=probs)

                _, next_state, reward, done = env.step(action)
                episode.append((state, action, reward))
                total_reward += reward
                state = next_state

            total_rewards.append(total_reward)
            G = 0
            for t in reversed(range(len(episode))):
                state, action, reward = episode[t]
                G = env.get_gamma() * G + reward
                if (state, action) not in [(x[0], x[1]) for x in episode[:t]]:
                    if (state, action) not in batch_returns:
                        batch_returns[(state, action)] = []
                    batch_returns[(state, action)].append(G)

            # Update Q-values and policy after each batch
            if (episode_num + 1) % batch_size == 0:
                for state_action, G_values in batch_returns.items():
                    state, action = state_action
                    Q[state, action] = np.mean(G_values)

                for state in range(env.get_state_size()):
                    policy[state] = np.eye(env.get_action_size())[
                        np.argmax(Q[state, :])]

                V = np.max(Q, axis=1)
                values.append(V)

                batch_returns = {}  # Clear the batch_returns for the next batch

            # Decay epsilon after each episode, but ensure it doesn't go below epsilon_min
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

        return policy, values, total_rewards


    def solve_with_parameters(self, env, **kwargs):
      """
      Solve a given Maze environment using Monte Carlo learning for custom parameters.
      Parameters are passed as keyword arguments.
      """

      # Initialization
      Q = np.random.rand(env.get_state_size(), env.get_action_size())
      V = np.zeros(env.get_state_size())
      policy = np.zeros((env.get_state_size(), env.get_action_size()))
      values = [V]
      total_rewards = []
      batch_returns = {}  # To store returns for the batch

      # Default Parameters
      num_episodes = 1000
      epsilon = 0.4
      epsilon_min = 0.01
      epsilon_decay = 0.99
      batch_size = 50

      # Override default parameters if passed in via kwargs
      if 'initial_epsilon' in kwargs:
          epsilon = kwargs['initial_epsilon']
      if 'decay_rate' in kwargs:
          epsilon_decay = kwargs['decay_rate']
      if 'batch_size' in kwargs:
        batch_size = kwargs['batch_size']

      for episode_num in range(num_episodes):
          episode = []
          state = env.reset()[1]
          done = False
          total_reward = 0

          while not done:
              # Epsilon-greedy policy with decaying epsilon
              probs = np.ones(env.get_action_size()) * epsilon / env.get_action_size()
              best_action = np.argmax(Q[state])
              probs[best_action] += (1.0 - epsilon)
              action = np.random.choice(np.arange(env.get_action_size()), p=probs)

              _, next_state, reward, done = env.step(action)
              episode.append((state, action, reward))
              total_reward += reward
              state = next_state

          total_rewards.append(total_reward)
          G = 0
          for t in reversed(range(len(episode))):
              state, action, reward = episode[t]
              G = env.get_gamma() * G + reward
              if (state, action) not in [(x[0], x[1]) for x in episode[:t]]:
                  if (state, action) not in batch_returns:
                      batch_returns[(state, action)] = []
                  batch_returns[(state, action)].append(G)

          # Update Q-values and policy after each batch
          if (episode_num + 1) % batch_size == 0:
              for state_action, G_values in batch_returns.items():
                  state, action = state_action
                  Q[state, action] = np.mean(G_values)

              for state in range(env.get_state_size()):
                  policy[state] = np.eye(env.get_action_size())[np.argmax(Q[state, :])]

              V = np.max(Q, axis=1)
              values.append(V)

              batch_returns = {}  # Clear the batch_returns for the next batch

          # Decay epsilon after each episode, but ensure it doesn't go below epsilon_min
          epsilon = max(epsilon_min, epsilon * epsilon_decay)

      return policy, values

    def parameter_testing_3D(self, env, initial_epsilons, decay_rates, batch_sizes, V_gold):
        errors = np.zeros(len(batch_sizes))
        total_runs = len(batch_sizes) * 10  # Calculate total number of runs
        current_run = 0  # Initialize current run count

        for i, batch_size in enumerate(batch_sizes):
            temp_errors = np.zeros(10)  # Temporary array to hold the 10 errors for each batch size

            for j in range(10):  # Perform 10 runs
                current_run += 1  # Increment current run count
                progress_percentage = (current_run / total_runs) * 100  # Calculate progress percentage

                _, values_current = self.solve_with_parameters(env, initial_epsilon=0.6, decay_rate=0.999, batch_size=batch_size)
                V_current = values_current[-1]
                error = mean_squared_error(V_current, V_gold)

                temp_errors[j] = error  # Store each error into temp_errors

                print(f"Progress: {progress_percentage:.2f}%")  # Print progress statement

            errors[i] = np.mean(temp_errors)  # Calculate and store the average error for each batch size

        return errors

    def plot_2D_heatmap(values, x_labels, y_labels, xlabel, ylabel, title, color_map='Greens_r'):
        """
        Plots a 2D heatmap.
        """
        fig, ax = plt.subplots()

        # Transpose the values for correct orientation
        values = values.T

        im = ax.imshow(values, cmap=color_map)

        # We want to show all ticks...
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_yticks(np.arange(len(y_labels)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(y_labels)

        # Loop over data dimensions and create text annotations
        for i in range(len(y_labels)):
            for j in range(len(x_labels)):
                text = ax.text(j, i, int(values[i, j]),
                            ha="center", va="center", color="w" if values[i, j] < (values.max() / 2) else "black")

        # Set axis labels and title
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('MSE', rotation=-90, va="bottom")

        plt.show()

    def run_multiple_times(self, env, runs=25, batch_size=50):
        all_rewards = []
        for i in range(runs):
            print(f'Training Run: {i}')
            _, _, rewards = self.solve(env)
            # Only store rewards at batch points for comparison
            all_rewards.append(rewards[::batch_size])
        return all_rewards

    @staticmethod
    def plot_learning_curve(all_rewards, batch_size=50):
        # Convert list of lists into a numpy array
        rewards_matrix = np.array(all_rewards)

        # Calculate mean and standard deviation
        mean_rewards = np.mean(rewards_matrix, axis=0)
        std_rewards = np.std(rewards_matrix, axis=0)

        # Plot
        episodes = np.arange(0, len(mean_rewards) * batch_size, batch_size)
        plt.plot(episodes, mean_rewards, label='Mean Rewards')
        plt.fill_between(episodes,
                        mean_rewards - std_rewards,
                        mean_rewards + std_rewards,
                        color='blue', alpha=0.2)

        plt.xlabel('Episodes')
        plt.ylabel('Total Non-discounted Sum of Reward')
        plt.title(f'Learning Curve with Mean and Standard Deviation across {rewards_matrix.shape[0]} runs')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_mse_vs_batch_size(errors, batch_sizes):
        # Shift each value in batch sizes by -10
        batch_sizes_shifted = [x - 10 for x in batch_sizes]

        plt.figure(figsize=(10,6))
        plt.plot(batch_sizes_shifted, errors, '-o', markersize=8)
        plt.title('MSE vs Batch Size')
        plt.xlabel('Batch Size')
        plt.ylabel('MSE')
        # Removed the grid lines
        plt.show()
    


# This class define the Temporal-Difference agent

class TD_agent(object):

    def solve(self, env):
        """
        Solve a given Maze environment using Temporal Difference learning
        input: env {Maze object} -- Maze to solve
        output:
          - policy {np.array} -- Optimal policy found to solve the given Maze environment
          - values {list of np.array} -- List of successive value functions for each episode
          - total_rewards {list of float} -- Corresponding list of successive total non-discounted sum of reward for each episode
        """

        # Initialisation
        Q = np.random.rand(env.get_state_size(), env.get_action_size())
        V = np.zeros(env.get_state_size())
        policy = np.zeros((env.get_state_size(), env.get_action_size()))
        values = [V]
        total_rewards = []

        # Parameters
        num_episodes = 500
        epsilon = 1.0
        epsilon_decay = 0.9999
        epsilon_min = 0.01
        alpha = 0.2

        for _ in range(num_episodes):
            state = env.reset()[1]
            total_reward = 0
            done = False

            while not done:
                # Epsilon-greedy action selection
                if np.random.rand() < epsilon:
                    action = np.random.choice(env.get_action_size())
                else:
                    action = np.argmax(Q[state, :])

                _, next_state, reward, done = env.step(action)

                # Q-learning update
                best_next_action = np.argmax(Q[next_state, :])
                td_target = reward + env.get_gamma() * \
                    Q[next_state, best_next_action]
                td_error = td_target - Q[state, action]
                Q[state, action] += alpha * td_error

                total_reward += reward
                state = next_state

            total_rewards.append(total_reward)
            V = np.max(Q, axis=1)
            values.append(V)

            # Decay epsilon after each episode, but ensure it doesn't go below epsilon_min
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

            # Update policy
            for state in range(env.get_state_size()):
                policy[state] = np.eye(env.get_action_size())[
                    np.argmax(Q[state, :])]

        return policy, values, total_rewards


    def parameter_testing(self, env, alpha=0.1, epsilon=1.0, epsilon_decay=0.995):
        """
        Solve a given Maze environment using Temporal Difference learning with specified alpha, epsilon, and epsilon_decay.
        """

        # Initialisation 
        Q = np.random.rand(env.get_state_size(), env.get_action_size())
        V = np.zeros(env.get_state_size())
        policy = np.zeros((env.get_state_size(), env.get_action_size()))
        values = [V]
        total_rewards = []

        # Parameters
        num_episodes = 1000
        epsilon_min = 0.01

        for _ in range(num_episodes):
            state = env.reset()[1]
            total_reward = 0
            done = False

            while not done:
                # Epsilon-greedy action selection
                if np.random.rand() < epsilon:
                    action = np.random.choice(env.get_action_size())
                else:
                    action = np.argmax(Q[state, :])

                _, next_state, reward, done  = env.step(action)

                # Q-learning update
                best_next_action = np.argmax(Q[next_state, :])
                td_target = reward + env.get_gamma() * Q[next_state, best_next_action]
                td_error = td_target - Q[state, action]
                Q[state, action] += alpha * td_error

                total_reward += reward
                state = next_state

            total_rewards.append(total_reward)
            V = np.max(Q, axis=1)
            values.append(V)

            # Decay epsilon after each episode, but ensure it doesn't go below epsilon_min
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

            # Update policy
            for state in range(env.get_state_size()):
                policy[state] = np.eye(env.get_action_size())[np.argmax(Q[state, :])]

        return policy, values, total_rewards


    def vary_epsilon(self, env, epsilons, trials=10):
        """Evaluate performance over different exploration parameters."""
        results = {}
        for eps in epsilons:
            total_rewards_accumulated = np.zeros(200)  # Assuming 200 episodes
            for _ in range(trials):
                _, _, total_rewards = self.parameter_testing(env, epsilon=eps)
                total_rewards_accumulated += np.array(total_rewards[:200])  # consider only the first 200 episodes
            results[eps] = total_rewards_accumulated / trials  # average rewards across trials
        return results

    def vary_alpha(self, env, alphas, trials=10):
        """Evaluate performance over different learning rates."""
        results = {}
        for alp in alphas:
            total_rewards_accumulated = np.zeros(200)  # Assuming 200 episodes
            for _ in range(trials):
                _, _, total_rewards = self.parameter_testing(env, alpha=alp)
                total_rewards_accumulated += np.array(total_rewards[:200])  # consider only the first 200 episodes
            results[alp] = total_rewards_accumulated / trials  # average rewards across trials
        return results


    def vary_epsilon_decay(self, env, epsilon_decays, trials=10):
        """Evaluate performance over different epsilon decay rates."""
        results = {}
        for eps_decay in epsilon_decays:
            total_rewards_accumulated = np.zeros(200)  # Assuming 200 episodes
            for _ in range(trials):
                _, _, total_rewards = self.parameter_testing(env, epsilon_decay=eps_decay)
                total_rewards_accumulated += np.array(total_rewards[:200])  # consider only the first 200 episodes
            results[eps_decay] = total_rewards_accumulated / trials  # average rewards across trials
        return results


    def vary_parameters(self, env, epsilons, alphas, constant_epsilon_decay, V_gold):
        """Evaluate performance over different exploration parameters and learning rates with a constant decay rate."""

        # Initialize 2D array to store results
        errors = np.zeros((len(epsilons), len(alphas)))  # Assuming 2D array now

        for i, eps in enumerate(epsilons):
            for j, alp in enumerate(alphas):
                _, values_current, _ = self.parameter_testing(env, epsilon=eps, alpha=alp, epsilon_decay=constant_epsilon_decay)
                # Calculate the last value function and measure the error
                V_current = values_current[-1]
                error = mean_squared_error(V_current, V_gold)

                # Store the error into the 2D array
                errors[i, j] = error

        # Get the coordinates of the minimum error
        min_error_coords = np.unravel_index(np.argmin(errors), errors.shape)

        return errors, min_error_coords

    def plot_2D_heatmap(values, x_labels, y_labels, xlabel, ylabel, title, color_map='Greens_r'):
        """
        Plots a 2D heatmap.
        """
        fig, ax = plt.subplots()

        values = values.T

        im = ax.imshow(values, cmap=color_map)

        # We want to show all ticks...
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_yticks(np.arange(len(y_labels)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(y_labels)

        # Loop over data dimensions and create text annotations
        for i in range(len(y_labels)):
            for j in range(len(x_labels)):
                text = ax.text(j, i, int(values[i, j]),
                            ha="center", va="center", color="w" if values[i, j] < (values.max() / 2) else "black")

        # Set axis labels and title
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        # # Flip the y-axis
        ax.invert_yaxis()

        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('MSE', rotation=-90, va="bottom")

        plt.show()


    @staticmethod
    def plot_results(results, parameter_name, smoothing_window=50):
        """Plot learning curves for different parameter values."""

        plt.figure(figsize=(10, 6))  # Increase the size of the plot

        # Use a colormap
        colors = plt.cm.viridis(np.linspace(0, 1, len(results)))

        for idx, (param_val, rewards) in enumerate(results.items()):
            # Consider only the first 200 episodes
            rewards = rewards[:200]

            # Smooth the curve
            rewards_smoothed = np.convolve(rewards, np.ones(smoothing_window) / smoothing_window, mode='valid')
            plt.plot(rewards_smoothed, label=f"{parameter_name}={round(param_val, 2)}", color=colors[idx])

        plt.xlabel('Episodes')
        plt.ylabel('Total Non-discounted Sum of Reward')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Place legend outside
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add grid
        plt.title(f'Learning Curve varying {parameter_name} (First 200 Episodes)')
        plt.tight_layout()
        plt.show()