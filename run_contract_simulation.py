# mvrun_contract_simulation.py

import matplotlib.pyplot as plt
import pandas as pd
import os
import networkx as nx
from contract_interface_mvp import VouchMinimal
from utils import generate_mul_eth_addresses

# --- History Tracking & Plotting ---

history = []
event_counter = 0
all_user_names = set()

def record_step(vm: VouchMinimal, user_map: dict, event_name: str):
    """Records the current state of scores for plotting."""
    global event_counter
    global history
    
    event_counter += 1
    record = {}
    record['event_step'] = event_counter
    record['event_name'] = event_name
    
    # Get score for all known users
    for name, address in user_map.items():
        record[name] = vm.get_score(address)
        
    history.append(record)
    
    # Print current ranks and scores
    print(f"\n--- Event: {event_name} (Step {event_counter}) ---")
    print("User\t| Score\t\t\t| Rank")
    print("-" * 40)
    
    sorted_users = sorted(user_map.items(), key=lambda item: vm.get_score(item[1]), reverse=True)
    
    for name, address in sorted_users:
        score = vm.get_score(address)
        rank = vm.get_rank(address)
        # Format score for better readability
        print(f"{name.ljust(8)}\t| {score:<16}\t| {rank}")


def plot_history(filename="contract_score_evolution.png"):
    """Uses pandas and matplotlib to plot the score evolution."""
    global history
    if not history:
        print("No history to plot.")
        return

    # Create output directory if it doesn't exist
    output_dir = "plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filepath = os.path.join(output_dir, filename)

    # Convert history to pandas DataFrame
    df = pd.DataFrame(history)
    
    # Set event_step as the index for plotting
    df = df.set_index('event_step')
    
    # Get all user columns (exclude 'event_name')
    user_columns = [col for col in df.columns if col != 'event_name']
    
    # Create the plot
    plt.figure(figsize=(15, 8))
    ax = df[user_columns].plot(kind='line', marker='o', linestyle='-')
    
    plt.title('VouchMinimal Score Evolution Over Time')
    plt.xlabel('Event Step')
    plt.ylabel('Score (Raw)')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Add event names as x-ticks
    plt.xticks(
        ticks=df.index, 
        labels=df['event_name'], 
        rotation=90, 
        fontsize=8
    )
    
    # Position legend outside the plot
    plt.legend(title='Users', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
    
    # Save the plot
    plt.savefig(filepath, dpi=300)
    print(f"\nPlot saved to {filepath}")
    plt.close()

# --- Simulation ---

def run_simulation():
    """
    Runs a test simulation of the VouchMinimal contract interface.
    """
    print("--- Starting VouchMinimal Simulation ---")

    # --- 1. Initialization ---
    # We must start with an empty graph for VouchMinimal
    #
    # Note: Your VouchMinimal class adds nodes to the graph as
    # they appear in vouches.
    
    g = nx.DiGraph()
    vm = VouchMinimal(g)

    # Create a map of user names to mock addresses
    user_names = ["Alice", "Bob", "Charlie", "David", "Eve"]
    sybil_names = ["Sybil_1", "Sybil_2", "Sybil_3"]
    all_names = user_names + sybil_names
    
    addresses = generate_mul_eth_addresses(len(all_names))
    user_map = dict(zip(all_names, addresses))

    record_step(vm, user_map, "Initialization")
    
    # --- 2. Vouching Test (Seed Vouches) ---
    # The first 5 vouches are "seed" vouches
    print("\n--- Phase 2: Seed Vouches ---")
    vm.vouch(user_map["Alice"], user_map["Bob"])
    record_step(vm, user_map, "Vouch: A -> B")
    
    vm.vouch(user_map["Alice"], user_map["Charlie"])
    record_step(vm, user_map, "Vouch: A -> C")

    vm.vouch(user_map["Bob"], user_map["Charlie"])
    record_step(vm, user_map, "Vouch: B -> C")
    
    vm.vouch(user_map["David"], user_map["Alice"])
    record_step(vm, user_map, "Vouch: D -> A")
    
    vm.vouch(user_map["David"], user_map["Bob"])
    record_step(vm, user_map, "Vouch: D -> B") # 5th seed vouch

    # --- 3. Normal Vouching ---
    print("\n--- Phase 3: Normal Vouching ---")
    vm.vouch(user_map["Charlie"], user_map["Alice"])
    record_step(vm, user_map, "Vouch: C -> A")

    # --- 4. Unvouching Test ---
    print("\n--- Phase 4: Unvouching ---")
    vm.unvouch(user_map["Alice"], user_map["Charlie"])
    record_step(vm, user_map, "Unvouch: A -x C")
    
    # --- 5. Evolution Test (Sybil Attack) ---
    print("\n--- Phase 5: Sybil Attack Simulation ---")
    # Sybils vouch for Eve
    vm.vouch(user_map["Sybil_1"], user_map["Eve"])
    record_step(vm, user_map, "Vouch: S1 -> Eve")
    
    vm.vouch(user_map["Sybil_2"], user_map["Eve"])
    record_step(vm, user_map, "Vouch: S2 -> Eve")
    
    vm.vouch(user_map["Sybil_3"], user_map["Eve"])
    record_step(vm, user_map, "Vouch: S3 -> Eve")
    
    # Sybils vouch for each other
    vm.vouch(user_map["Sybil_1"], user_map["Sybil_2"])
    record_step(vm, user_map, "Vouch: S1 -> S2")
    
    vm.vouch(user_map["Sybil_2"], user_map["Sybil_3"])
    record_step(vm, user_map, "Vouch: S2 -> S3")
    
    # Eve vouches back
    vm.vouch(user_map["Eve"], user_map["Sybil_1"])
    record_step(vm, user_map, "Vouch: Eve -> S1")
    
    # --- 6. Community Response ---
    print("\n--- Phase 6: Community Response ---")
    # David, a trusted user, vouches for Eve, giving her "real" score
    vm.vouch(user_map["David"], user_map["Eve"])
    record_step(vm, user_map, "Vouch: D -> Eve")
    
    # Alice, another trusted user, vouches for David
    vm.vouch(user_map["Alice"], user_map["David"])
    record_step(vm, user_map, "Vouch: A -> D")
    
    # --- 7. Plotting ---
    print("\n--- Phase 7: Generating Plot ---")
    plot_history("contract_simulation_evolution.png")
    
    print("\n--- Simulation Complete ---")

if __name__ == "__main__":
    run_simulation()
