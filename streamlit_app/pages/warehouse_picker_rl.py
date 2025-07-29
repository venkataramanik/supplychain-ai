# warehouse_picker_rl.py

import streamlit as st
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import time # For simulation delay

# --- Streamlit Page Configuration ---
st.set_page_config(layout="wide", page_title="🤖 AI for Warehouse Picking Optimization")

st.title("🤖 AI for Warehouse Picking Optimization with Reinforcement Learning")

st.header("🌟 Business Context: Smarter Warehouse Picking")
st.markdown("""
Order picking is one of the most labor-intensive and costly activities in a warehouse. Inefficient picking paths lead to:
-   **Increased Labor Costs:** Workers spending more time walking than picking.
-   **Slower Order Fulfillment:** Delays in getting products to customers.
-   **Higher Operational Expenses:** More energy consumption, wear-and-tear on equipment.

**Reinforcement Learning (RL)** offers a powerful solution by training AI agents to learn optimal strategies for navigating the warehouse and collecting items. Instead of fixed, pre-programmed rules, the RL agent learns through trial and error, adapting to different warehouse layouts, item locations, and order complexities.
""")

st.subheader("💡 How RL Solves This Problem:")
st.markdown("""
1.  **Trial and Error Learning:** The AI agent explores the warehouse, making decisions (move up, down, left, right, pick item).
2.  **Rewards and Penalties:** It receives "rewards" for good actions (e.g., picking an item, completing an order) and "penalties" for bad ones (e.g., taking unnecessary steps).
3.  **Optimal Policy:** Over many trials, the agent learns a "policy" – a set of rules that tells it the best action to take in any given situation to maximize its total reward (which translates to minimizing steps/time in our case).

**Impact:** This leads to significantly reduced travel distances, faster picking times, improved worker productivity, and ultimately, higher warehouse throughput.
""")

st.divider()

st.header("🔬 Simulation: Warehouse Picking with Q-Learning")
st.info("This simulation uses a simplified warehouse grid and a Q-Learning agent to find efficient paths for fulfilling orders. The agent learns which actions to take in different 'states' (location, items remaining) to maximize its rewards (i.e., minimize steps).")

# --- Warehouse Environment Class ---
class WarehouseEnv:
    def __init__(self, size=(10, 10), start_pos=(0, 0)):
        self.rows, self.cols = size
        self.start_pos = start_pos
        self.current_pos = list(start_pos)
        self.items = {}  # {item_id: (row, col)}
        self.items_to_pick = [] # List of item_ids for the current order
        self.picked_items = set()
        self.grid = np.zeros(size) # 0 for empty, 1 for item
        self.action_space = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'PICK']
        # state_space_size[2] represents the max value of the item_mask.
        # This will need to be consistent with the number of items considered in training.
        # For a fixed max of 10 items in training, 2**10 is appropriate for the mask.
        self.state_space_size = (self.rows, self.cols, 2**10)

    def reset(self, item_positions, order_items):
        self.current_pos = list(self.start_pos)
        self.items = item_positions
        self.items_to_pick = list(order_items) # Make a copy
        self.picked_items = set()
        self.grid = np.zeros((self.rows, self.cols))
        for item_id, pos in self.items.items():
            self.grid[pos[0], pos[1]] = 1 # Mark item locations
        return self._get_state()

    def _get_state(self):
        # State includes current position and remaining items (represented by a bitmask for simplicity)
        item_mask = 0
        # For state consistency, ensure items used to form the mask are sorted
        # This assumes a predefined, stable set of items for mask generation.
        # For simplicity in training, we assume item IDs 'A' through 'J' (10 items) for the mask.
        # In a real app, this would need careful design based on max possible items.
        all_possible_mask_items = [chr(ord('A') + i) for i in range(10)] # 'A' to 'J'
        
        for i, item_id in enumerate(all_possible_mask_items):
            # Only include items in the current order for the mask if they are not yet picked
            if item_id in self.items_to_pick and item_id not in self.picked_items:
                item_mask |= (1 << i) # Set bit if item is not yet picked
        
        # Clamp item_mask to avoid issues if state_space_size is too small for many items
        max_item_mask_idx = self.state_space_size[2] - 1
        item_mask = min(item_mask, max_item_mask_idx)

        return (self.current_pos[0], self.current_pos[1], item_mask)

    def step(self, action_idx):
        action = self.action_space[action_idx]
        reward = -1 # Penalty for each step (encourages shortest path)
        done = False
        
        next_pos = list(self.current_pos)

        if action == 'UP':
            next_pos[0] = max(0, self.current_pos[0] - 1)
        elif action == 'DOWN':
            next_pos[0] = min(self.rows - 1, self.current_pos[0] + 1)
        elif action == 'LEFT':
            next_pos[1] = max(0, self.current_pos[1] - 1)
        elif action == 'RIGHT':
            next_pos[1] = min(self.cols - 1, self.current_pos[1] + 1)
        elif action == 'PICK':
            # Check if current position has an item to pick that is in the current order and not yet picked
            # Iterate over a copy because we might modify self.items_to_pick
            for item_id in list(self.items_to_pick): 
                if self.items.get(item_id) == tuple(self.current_pos) and item_id not in self.picked_items:
                    self.picked_items.add(item_id)
                    reward += 100 # Reward for picking an item
                    break # Only pick one item per 'PICK' action, even if multiple are at the same spot
            else: # No item picked at this location (or already picked)
                reward -= 5 # Penalty for trying to pick where there's nothing or it's already picked

        # Update position
        self.current_pos = next_pos
        
        # Check if all items in the order are picked
        if len(self.picked_items) == len(self.items_to_pick) and len(self.items_to_pick) > 0:
            reward += 500 # Large reward for completing the order
            done = True
        elif len(self.picked_items) == len(self.items_to_pick) and len(self.items_to_pick) == 0:
            # Case for empty order, immediately done with small reward
            done = True
            reward = 100 # Still get some reward for empty order if done

        return self._get_state(), reward, done

    def render(self, ax, path=None):
        ax.clear()
        ax.set_xticks(np.arange(-.5, self.cols, 1), minor=True)
        ax.set_yticks(np.arange(-.5, self.rows, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
        ax.set_xlim(-0.5, self.cols - 0.5)
        ax.set_ylim(-0.5, self.rows - 0.5)
        ax.invert_yaxis() # Invert y-axis to have (0,0) at top-left
        ax.set_aspect('equal', adjustable='box')

        # Draw grid
        for r in range(self.rows):
            for c in range(self.cols):
                color = 'white'
                if (r, c) == self.start_pos:
                    color = 'lightblue'
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=color, edgecolor='black', linewidth=0.5))

        # Mark items
        for item_id, (r, c) in self.items.items():
            if item_id in self.items_to_pick and item_id not in self.picked_items:
                ax.text(c, r, f'I{item_id}', color='red', ha='center', va='center', fontsize=9, fontweight='bold')
                ax.add_patch(plt.Circle((c, r), 0.4, color='orange', alpha=0.6))
            elif item_id in self.picked_items:
                 ax.text(c, r, f'I{item_id}', color='green', ha='center', va='center', fontsize=9, fontweight='bold')
                 ax.add_patch(plt.Circle((c, r), 0.4, color='lightgreen', alpha=0.6))
            else: # Items not in current order
                ax.text(c, r, f'I{item_id}', color='gray', ha='center', va='center', fontsize=9)
                ax.add_patch(plt.Circle((c, r), 0.4, color='lightgray', alpha=0.3))


        # Mark picker current position
        ax.text(self.current_pos[1], self.current_pos[0], '🤖', ha='center', va='center', fontsize=18)
        
        # Draw path if provided
        if path:
            xs = [p[1] for p in path]
            ys = [p[0] for p in path]
            ax.plot(xs, ys, 'b-o', alpha=0.7, markersize=5, linewidth=2)


        ax.set_title(f"Warehouse State (Picked: {len(self.picked_items)}/{len(self.items_to_pick)})")
        ax.axis('off')

# --- Q-Learning Function ---
# Now takes hashable parameters (grid_size, etc.)
@st.cache_resource(show_spinner="Training AI Agent (Q-Learning)... This may take a moment.")
def train_q_agent(grid_size, num_episodes=5000, learning_rate=0.1, discount_factor=0.99, epsilon_decay=0.995, min_epsilon=0.01):
    # Create the environment INSIDE the cached function
    env = WarehouseEnv(size=(grid_size, grid_size))

    # Q-table shape uses env properties for consistency
    q_table = np.zeros((env.rows, env.cols, env.state_space_size[2], len(env.action_space)))
    epsilon = 1.0 # Exploration-exploitation trade-off

    st.write(f"Starting Q-Learning training for {num_episodes} episodes...")
    st.write(f"Q-table shape: {q_table.shape}")

    # Use a Streamlit progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    for episode in range(num_episodes):
        # For training, let's use a fixed small set of item IDs for consistency in the item mask
        # but their positions will be randomized. This is a simplification for stable training.
        train_order_items = ['A', 'B'] # Example small order for training consistency

        # Generate item positions relative to the env size
        train_item_positions = {
            'A': (random.randint(0, env.rows-1), random.randint(0, env.cols-1)),
            'B': (random.randint(0, env.rows-1), random.randint(0, env.cols-1)),
            'C': (random.randint(0, env.rows-1), random.randint(0, env.cols-1)),
            'D': (random.randint(0, env.rows-1), random.randint(0, env.cols-1))
        }

        # Reset the environment for a new episode
        state_r, state_c, state_item_mask = env.reset(train_item_positions, train_order_items)
        done = False

        while not done:
            if random.uniform(0, 1) < epsilon:
                action_idx = random.randrange(len(env.action_space)) # Explore action space
            else:
                action_idx = np.argmax(q_table[state_r, state_c, state_item_mask, :]) # Exploit learned values
            
            # Take action and observe new state and reward
            new_state, reward, done = env.step(action_idx)
            new_state_r, new_state_c, new_state_item_mask = new_state

            # Q-learning update rule
            old_value = q_table[state_r, state_c, state_item_mask, action_idx]
            next_max = np.max(q_table[new_state_r, new_state_c, new_state_item_mask, :])

            new_value = (1 - learning_rate) * old_value + learning_rate * (reward + discount_factor * next_max)
            q_table[state_r, state_c, state_item_mask, action_idx] = new_value

            state_r, state_c, state_item_mask = new_state_r, new_state_c, new_state_item_mask

        # Epsilon decay
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        if (episode + 1) % 100 == 0:
            status_text.text(f"Training Episode {episode + 1}/{num_episodes} (Epsilon: {epsilon:.2f})")
            progress_bar.progress((episode + 1) / num_episodes)
    
    st.success("Q-Learning Training Complete!")
    return q_table

# --- Streamlit App Logic ---

# Sidebar for controls
st.sidebar.header("Simulation Settings")
grid_size = st.sidebar.slider("Warehouse Size (N x N)", 5, 20, 10)
num_total_items = st.sidebar.slider("Number of Total Items in Warehouse", 5, 20, 8)
num_episodes = st.sidebar.slider("Q-Learning Episodes (for training)", 1000, 10000, 5000, step=1000)

# Instantiate env for visualization and testing outside the cached function
env_for_testing_and_rendering = WarehouseEnv(size=(grid_size, grid_size))

# Generate fixed item positions for consistency across runs with same settings
# Use st.session_state to persist item positions
if 'item_positions' not in st.session_state or st.sidebar.button("Re-randomize Item Positions"):
    st.session_state.item_positions = {}
    for i in range(num_total_items):
        item_id = chr(ord('A') + i) # A, B, C...
        st.session_state.item_positions[item_id] = (random.randint(0, grid_size - 1), random.randint(0, grid_size - 1))

st.sidebar.write("---")
st.sidebar.write("### Test Order Configuration")
# Select items for the order dynamically from available items
all_item_ids = list(st.session_state.item_positions.keys())
selected_order_items = st.sidebar.multiselect(
    "Select Items for the Picking Order:",
    options=all_item_ids,
    default=all_item_ids[:min(3, len(all_item_ids))] # Default to first 3 items
)
if not selected_order_items:
    st.sidebar.warning("Please select at least one item for the order.")
    selected_order_items = [] # Ensure it's empty if nothing selected

# Call the cached function with hashable parameters
q_table = train_q_agent(grid_size, num_episodes=num_episodes)

st.subheader(f"Warehouse Layout ({grid_size}x{grid_size}) and Item Locations:")

# Visualize initial item placements
fig_initial, ax_initial = plt.subplots(figsize=(grid_size, grid_size))
env_for_testing_and_rendering.reset(st.session_state.item_positions, selected_order_items)
env_for_testing_and_rendering.render(ax_initial)
st.pyplot(fig_initial)


st.subheader(f"Optimal Picking Path for Order: {', '.join(selected_order_items)}")

if not selected_order_items:
    st.warning("No items selected for the order. Please select items in the sidebar to visualize a path.")
else:
    # Use the env_for_testing_and_rendering object for running the learned policy
    env_test = env_for_testing_and_rendering # Alias for clarity
    state_r, state_c, state_item_mask = env_test.reset(st.session_state.item_positions, selected_order_items)
    done = False
    path_taken = [env_test.current_pos]
    total_reward = 0
    steps = 0

    fig_path, ax_path = plt.subplots(figsize=(grid_size, grid_size))
    placeholder = st.empty() # Placeholder for animated plot

    while not done and steps < (grid_size * grid_size * 5): # Add a step limit to prevent infinite loops
        # Ensure item_mask calculation is consistent with training (fixed set of 10 for mask)
        all_possible_mask_items = [chr(ord('A') + i) for i in range(10)] # 'A' to 'J'
        item_mask_val = 0
        for i, item_id in enumerate(all_possible_mask_items):
            if item_id in env_test.items_to_pick and item_id not in env_test.picked_items:
                item_mask_val |= (1 << i)
        
        # Clamp item_mask_val to the max index used in q_table
        max_item_mask_idx = q_table.shape[2] - 1
        item_mask_val = min(item_mask_val, max_item_mask_idx)

        # Get best action from Q-table
        try:
            action_idx = np.argmax(q_table[state_r, state_c, item_mask_val, :])
        except IndexError:
            st.error("Error: The agent encountered an unlearned state. This can happen if the current order's item combination or mask value wasn't sufficiently explored during training, or if `num_total_items` in the sidebar exceeds the `state_space_size` assumption (max 10 for mask).")
            st.warning("Try reducing the number of items in the order or increasing Q-Learning episodes.")
            break # Exit loop if state is out of bounds

        new_state, reward, done = env_test.step(action_idx)
        total_reward += reward
        steps += 1
        path_taken.append(env_test.current_pos)

        # Update plot
        env_test.render(ax_path, path=path_taken)
        placeholder.pyplot(fig_path)
        plt.close(fig_path) # Close the figure to prevent display issues in Streamlit
        time.sleep(0.1) # Simulate movement delay

        state_r, state_c, state_item_mask = new_state
    
    final_message = f"**Order Completed!** Total Steps: {steps}, Total Reward: {total_reward}" if done else f"**Simulation stopped (max steps reached).** Total Steps: {steps}, Total Reward: {total_reward}"
    if len(env_test.picked_items) < len(selected_order_items):
        final_message += f" (Note: Only {len(env_test.picked_items)}/{len(selected_order_items)} items were picked.)"
        st.warning("The agent did not pick all items in the order within the step limit. This might indicate that more training episodes are needed, or the order is too complex for this simplified environment/training.")
    st.success(final_message)


st.divider()

st.header("🚀 Next Steps: Real-World Implementation")
st.markdown("""
This simulation is a simplified example. A real-world RL system for warehouse picking would involve:
-   **More Complex State Representation:** Including aisle numbers, item dimensions, current picker load, time constraints.
-   **Larger Action Space:** Diagonal moves, actions related to equipment (forklifts, AGVs), multi-item picking.
-   **Advanced RL Algorithms:** Deep Q-Networks (DQN), Proximal Policy Optimization (PPO), or Actor-Critic methods for larger, more complex environments.
-   **Integration with WMS/WES:** Real-time data feeds from Warehouse Management/Execution Systems.
-   **Simulation Environment:** Building a highly realistic digital twin of the warehouse for training the RL agents safely and efficiently.
-   **Scalability:** Distributing training across multiple processors or GPUs.

By moving towards such sophisticated AI systems, logistics and manufacturing businesses can achieve unprecedented levels of efficiency and automation in their warehouse operations.
""")
