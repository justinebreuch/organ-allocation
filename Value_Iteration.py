from collections import deque
import numpy as np
import pandas as pd
import random
import sys
import time
import pyttsx3 #To let me know when the code is done
import heapq

class SequentialOrganAllocationMDP:
    def __init__(self, initial_state, available_organs):
        """
        Initialize the sequential MDP for organ allocation.

        Parameters:
        - initial_state: List of dictionaries representing waitlist candidates.
        - available_organs: Dictionary with available organs by blood type.
        """
        self.initial_state = (
            frozenset({r['id']: (r['age'], r['MELD'], r['blood_type'], r['allocated']) for r in initial_state}.items()),
            frozenset(available_organs.items())  # Convert to frozenset for hashing
        )
        self.value_table = {self.initial_state: 0}  # Value function initialized for the initial state
        self.policy = {self.initial_state: None}  # Policy initialized for the initial state
        self.reward_table = self._precompute_rewards(initial_state)  # Precompute rewards
        self.deltas = []  # To store deltas over iterations

    def _precompute_rewards(self, recipients):
        """
        Precompute rewards for all recipients.

        Parameters:
        - recipients: List of dictionaries representing recipients.

        Returns:
        - A dictionary mapping recipient IDs to precomputed rewards.
        """
        ids = np.array([r['id'] for r in recipients])
        ages = np.array([r['age'] for r in recipients])
        melds = np.array([r['MELD'] for r in recipients])

        rewards = 10 + (1 / (1 + np.maximum(melds, 0))) * 10 - (ages * 0.1)
        reward_table = dict(zip(ids, rewards))

        return reward_table
    
    def calculate_reward(self, recipient):
        """
        Retrieve precomputed reward for the recipient.

        Parameters:
        - recipient: Tuple representing the recipient receiving the organ.

        Returns:
        - Precomputed reward value.
        """
        recipient_id = recipient[0]  # Extract the recipient ID
        return self.reward_table.get(recipient_id, 0)  # Default to 0 if not found

    #@profile
    def generate_next_state(self, state, action):
        """
        Generate the next state based on the current state and action.

        Parameters:
        - state: Tuple (recipients_dict, available_organs_dict).
        - action: Tuple (recipient_id, organ_type).

        Returns:
        - next_state: Updated state after taking the action.
        - reward: Associated reward for the action.
        """

        recipients = dict(state[0])  # Reconstruct recipients dictionary
        available_organs = dict(state[1])  # Reconstruct available organs dictionary
        if action is None:
            return (frozenset(recipients.items()), frozenset(available_organs.items())), 0
        
        recipient_id, organ_type = action
        recipient = recipients.get(recipient_id)
        if recipient is not None and available_organs[organ_type] > 0:
            # Ensure organ compatibility
            if recipient[2] == organ_type:  # Blood type match
                if random.random() < 1.1:  # 90% success
                    recipients[recipient_id] = (recipient[0], recipient[1], recipient[2], 1)
                    available_organs[organ_type] -= 1
                    reward = self.calculate_reward((recipient_id,) + recipient)
                    return (frozenset(recipients.items()), frozenset(available_organs.items())), reward
                else:  # 10% failure
                    recipients.pop(recipient_id)
                    available_organs[organ_type] -= 1
                    reward = -100  # Large penalty for death
                    return (frozenset(recipients.items()), frozenset(available_organs.items())), reward
        return (frozenset(recipients.items()), frozenset(available_organs.items())), 0
        
    #@profile
    #This one is without pruning/sweeping
    '''
    def value_iteration(self, gamma=0.9, epsilon=0.01, threshold_ratio=0.8):
        """
        Perform value iteration with pruning.
        
        Parameters:
        - gamma: Discount factor (default 0.9).
        - epsilon: Convergence threshold (default 0.01).
        - max_states: Maximum number of states to retain in the queue during pruning.
        """
        states_to_explore = deque([self.initial_state])
        visited_states = set()  # Track visited states
        self.deltas = []  # Reset deltas at the start of value iteration
        iteration_count = 0

        while True:
            iteration_count += 1
            delta = 0
            new_states_to_explore = []

            print(f"Iteration {iteration_count}, States to explore: {len(states_to_explore)}")

            while states_to_explore:
                current_state = states_to_explore.popleft()

                # Skip already visited states
                if current_state in visited_states:
                    continue

                visited_states.add(current_state)
                old_value = self.value_table[current_state]
                max_value = float('-inf')
                best_action = None
                no_valid_actions = True

                #recipients, available_organs = current_state
                recipients = dict(current_state[0])  # Reconstruct recipients dictionary
                available_organs = dict(current_state[1])
                for recipient_id, recipient_data in recipients.items():
                    if recipient_data[3] == 0:  # Not yet allocated
                        for organ_type in available_organs.keys():
                            if available_organs[organ_type] > 0:
                                no_valid_actions = False
                                action = (recipient_id, organ_type)
                                next_state, reward = self.generate_next_state(current_state, action)
                                value = reward + gamma * self.value_table.get(next_state, 0)

                                if value > max_value:
                                    max_value = value
                                    best_action = action

                                if next_state not in self.value_table:
                                    self.value_table[next_state] = 0
                                    self.policy[next_state] = None
                                    new_states_to_explore.append((next_state, abs(old_value - max_value)))
              
                if no_valid_actions:
                    continue

                self.value_table[current_state] = max_value
                self.policy[current_state] = best_action
                delta = max(delta, abs(old_value - max_value))

                #Threshold pruning
                if new_states_to_explore:
                    max_value_in_iteration = max(value for _, value in new_states_to_explore)
                    threshold = threshold_ratio * max_value_in_iteration

                    # Retain only states above the threshold
                    new_states_to_explore = [
                        (state, value) for state, value in new_states_to_explore if value >= threshold
                    ]

            print(f"Pruned to {len(new_states_to_explore)} states.")
            states_to_explore = deque(state for state, _ in new_states_to_explore)
            self.deltas.append(delta)
            print(f"Delta: {delta}")

            if delta < epsilon and not states_to_explore:
                break
    '''
    def value_iteration(self, gamma=0.9, epsilon=0.01, threshold_ratio=0.8, top_k_actions=4):
        """
        Perform value iteration with combined threshold-based pruning and action-level pruning.

        Parameters:
        - gamma: Discount factor (default 0.9).
        - epsilon: Convergence threshold (default 0.01).
        - threshold_ratio: Fraction of the max value to retain states (default 0.8).
        - top_k_actions: Number of top actions to retain for each state (default 3).
        """
        states_to_explore = deque([self.initial_state])
        visited_states = set()  # Track visited states
        self.deltas = []  # Reset deltas at the start of value iteration
        iteration_count = 0

        while True:
            iteration_count += 1
            delta = 0
            new_states_to_explore = []

            print(f"Iteration {iteration_count}, States to explore: {len(states_to_explore)}")

            while states_to_explore:
                current_state = states_to_explore.popleft()

                # Skip already visited states
                if current_state in visited_states:
                    continue

                visited_states.add(current_state)
                old_value = self.value_table[current_state]
                max_value = float('-inf')
                best_action = None
                no_valid_actions = True

                # Reconstruct dictionaries from frozenset
                recipients = dict(current_state[0])
                available_organs = dict(current_state[1])

                # Evaluate all actions and sort by reward
                actions_with_values = []
                for recipient_id, recipient_data in recipients.items():
                    if recipient_data[3] == 0:  # Not yet allocated
                        for organ_type in available_organs.keys():
                            if available_organs[organ_type] > 0:
                                no_valid_actions = False
                                action = (recipient_id, organ_type)
                                next_state, reward = self.generate_next_state(current_state, action)
                                value = reward + gamma * self.value_table.get(next_state, 0)
                                actions_with_values.append((action, value, next_state, reward))

                # Keep only the top k actions
                top_actions = sorted(actions_with_values, key=lambda x: x[1], reverse=True)[:top_k_actions]

                for action, value, next_state, reward in top_actions:
                    if value > max_value:
                        max_value = value
                        best_action = action

                    if next_state not in self.value_table:
                        self.value_table[next_state] = 0
                        self.policy[next_state] = None
                        new_states_to_explore.append((next_state, abs(old_value - max_value)))

                if no_valid_actions:
                    continue

                self.value_table[current_state] = max_value
                self.policy[current_state] = best_action
                delta = max(delta, abs(old_value - max_value))

            # Compute the maximum value and threshold
            if new_states_to_explore:
                max_value_in_iteration = max(value for _, value in new_states_to_explore)
                threshold = threshold_ratio * max_value_in_iteration

                # Retain only states above the threshold
                new_states_to_explore = [
                    (state, value) for state, value in new_states_to_explore if value >= threshold
                ]

            print(f"Pruned to {len(new_states_to_explore)} states.")
            states_to_explore = deque(state for state, _ in new_states_to_explore)
            self.deltas.append(delta)
            print(f"Delta: {delta}")

            if delta < epsilon and not states_to_explore:
                break
    import heapq

    def value_iteration(self, gamma=0.9, epsilon=0.01, threshold_ratio=0.8, top_k_actions=3):
        """
        Perform value iteration with prioritized sweeping, threshold-based pruning, and action-level pruning.

        Parameters:
        - gamma: Discount factor (default 0.9).
        - epsilon: Convergence threshold (default 0.01).
        - threshold_ratio: Fraction of the max value to retain states (default 0.8).
        - top_k_actions: Number of top actions to retain for each state (default 3).
        """
        # Priority queue for prioritized sweeping
        priority_queue = []
        visited_states = set()  # Track visited states
        self.deltas = []  # Reset deltas at the start of value iteration
        iteration_count = 0

        # Initialize priority queue with the initial state
        heapq.heappush(priority_queue, (0, self.initial_state))  # Priority is delta (0 initially)

        while priority_queue:
            iteration_count += 1
            delta = 0
            new_states_to_explore = []

            print(f"Iteration {iteration_count}, Priority queue size: {len(priority_queue)}")

            while priority_queue:
                _, current_state = heapq.heappop(priority_queue)  # Get the highest-priority state

                # Skip already visited states
                if current_state in visited_states:
                    continue

                visited_states.add(current_state)
                old_value = self.value_table[current_state]
                max_value = float('-inf')
                best_action = None
                no_valid_actions = True

                # Reconstruct dictionaries from frozenset
                recipients = dict(current_state[0])
                available_organs = dict(current_state[1])

                # Evaluate all actions and sort by reward
                actions_with_values = []
                for recipient_id, recipient_data in recipients.items():
                    if recipient_data[3] == 0:  # Not yet allocated
                        for organ_type in available_organs.keys():
                            if available_organs[organ_type] > 0:
                                no_valid_actions = False
                                action = (recipient_id, organ_type)
                                next_state, reward = self.generate_next_state(current_state, action)
                                value = reward + gamma * self.value_table.get(next_state, 0)
                                actions_with_values.append((action, value, next_state, reward))

                # Keep only the top k actions
                top_actions = sorted(actions_with_values, key=lambda x: x[1], reverse=True)[:top_k_actions]

                for action, value, next_state, reward in top_actions:
                    if value > max_value:
                        max_value = value
                        best_action = action

                    if next_state not in self.value_table:
                        self.value_table[next_state] = 0
                        self.policy[next_state] = None
                        new_states_to_explore.append((next_state, abs(old_value - max_value)))

                if no_valid_actions:
                    continue

                self.value_table[current_state] = max_value
                self.policy[current_state] = best_action
                delta = max(delta, abs(old_value - max_value))

                # Backpropagate priority to predecessors
                for action, _, next_state, _ in top_actions:
                    if next_state not in visited_states:
                        heapq.heappush(priority_queue, (-delta, next_state))

            # Compute the maximum value and threshold
            if new_states_to_explore:
                max_value_in_iteration = max(value for _, value in new_states_to_explore)
                threshold = threshold_ratio * max_value_in_iteration

                # Retain only states above the threshold
                new_states_to_explore = [
                    (state, value) for state, value in new_states_to_explore if value >= threshold
                ]

            print(f"Pruned to {len(new_states_to_explore)} states.")
            self.deltas.append(delta)
            print(f"Delta: {delta}")

            if delta < epsilon and not priority_queue:
                break


    def get_deltas(self):
        """
        Retrieve the deltas recorded during value iteration.

        Returns:
        - List of delta values for each iteration.
        """
        return self.deltas #To use for plotting later
    
    #@profile
    def simulate_with_policy(self, steps=10):
        """
        Simulate the allocation process using the computed policy.

        Parameters:
        - steps: Number of allocation steps to simulate (default 10).

        Returns:
        - Total reward, total deaths, total allocations.
        """
        current_state = self.initial_state
        total_reward = 0
        total_deaths = 0
        total_allocations = 0

        for _ in range(steps):
            action = self.policy.get(current_state, None)
            #print(action)
            if action is None:
                break

            next_state, reward = self.generate_next_state(current_state, action)

            if reward == -100:  # Death penalty
                total_deaths += 1
            elif reward > 0:  # Successful allocation
                total_allocations += 1

            total_reward += reward
            current_state = next_state

        return total_reward, total_deaths, total_allocations

def notify_completion():
    #Audibly notifies me that the code has finished runnning
    engine = pyttsx3.init()
    engine.say("Your code has finished running.")
    engine.runAndWait()

def main():
    
    num_ppl = int(sys.argv[1])
    threshold = float(sys.argv[2])
    
    # Start the timer
    start_time = time.time()
      
    df = pd.read_csv('sampled_waitlist.csv')
    np.random.seed(42)
    df = df.iloc[np.random.choice(np.arange(0, len(df)), size=num_ppl, replace=False)] #Take a small sample 
    initial_state = df.apply(
        lambda row: {
            'id': row.name + 1,  # Generate an 'id' starting from 1
            'age': row['RECIPIENT_AGE'],  # Replace with the actual age column if available
            'MELD': row['INIT_MELD_PELD_LAB_SCORE'],
            'blood_type': row['RECIPIENT_BLOOD_TYPE'],
            'allocated': 0  # Default value
        }, axis=1
    ).tolist()
    print(f"Num ppl: {len(initial_state)}")
    available_organs = {'A': 2, 'A1': 3, 'A1B': 1, 'A2': 1, 'A2B': 1, 'AB': 1, 'B': 2, 'O': 7, 'AB': 1}

    # Initialize and execute the MDP
    mdp_model = SequentialOrganAllocationMDP(initial_state, available_organs)
    mdp_model.value_iteration(gamma=0.9, epsilon=0.01, threshold_ratio = threshold)

    deltas = mdp_model.get_deltas()
    total_reward, total_deaths, total_allocations = mdp_model.simulate_with_policy(steps=len(initial_state))
    print(f"Total reward: {total_reward}, Total deaths: {total_deaths}, Total allocations: {total_allocations}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time/60} min")
    #notify_completion()
    


if __name__ == '__main__':
    main()