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
            if
