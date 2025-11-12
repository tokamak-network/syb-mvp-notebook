# mvrun_contract_simulation.py

import matplotlib.pyplot as plt
import pandas as pd
import os
import networkx as nx
from contract_interface_mvp import VouchMinimal
from utils.utils import generate_mul_eth_addresses
from utils.plot_graphs import plot_graph_evolution_with_scores
import random # Import random for NvM simulation

# --- History Tracking & Plotting (No Changes) ---

history = []
graphs_history = []
scores_history = []
event_counter = 0
all_user_names = set()

def reset_globals():
    """Resets global history trackers for a new simulation run."""
    global history, graphs_history, scores_history, event_counter
    history = []
    graphs_history = []
    scores_history = []
    event_counter = 0
    print("\n" + "="*60)
    print("      GLOBALS RESET: Starting new simulation...")
    print("="*60 + "\n")

def record_step(vm: VouchMinimal, user_map: dict, event_name: str):
    """Records the current state of scores for plotting."""
    global event_counter
    global history
    global graphs_history
    global scores_history
    
    event_counter += 1
    record = {}
    record['event_step'] = event_counter
    record['event_name'] = event_name
    
    # Get score for all known users (for line plot)
    for name, address in user_map.items():
        record[name] = vm.get_score(address)
        
    history.append(record)
    
    # --- Add logic for graph evolution plot ---
    # Store a *copy* of the current graph
    graphs_history.append(vm.network.copy())
    
    # Store scores mapped to graph node *indices*
    current_scores_dict = {}
    for node_idx in vm.network.nodes():
        address = vm.idx_to_address.get(node_idx)
        if address:
            current_scores_dict[node_idx] = vm.get_score(address)
        else:
            current_scores_dict[node_idx] = 0.0
    scores_history.append(current_scores_dict)
    # --- End new logic ---

    
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

# --- Simulation Setup (Modified) ---

def setup_simulation(num_real_users: int, num_sybil_users: int):
    """Creates the initial contract and user map based on parameters."""
    g = nx.DiGraph()
    vm = VouchMinimal(g)

    # --- UPDATED LOGIC ---
    # Generate procedural names for scalability
    # Use a base list for the first few to keep simulations comparable
    base_user_names = ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace"]
    
    if num_real_users <= len(base_user_names):
        user_names = base_user_names[:num_real_users]
    else:
        # Use base names and add procedural names
        user_names = base_user_names + [f"User_{i+1}" for i in range(num_real_users - len(base_user_names))]
        
    sybil_names = [f"Sybil_{i+1}" for i in range(num_sybil_users)]
    # --- END UPDATED LOGIC ---
    
    all_names = user_names + sybil_names
    
    addresses = generate_mul_eth_addresses(len(all_names))
    user_map = dict(zip(all_names, addresses))
    
    print(f"--- Simulation Setup ({num_real_users} real, {num_sybil_users} sybil) ---")
    print(f"Real Users: {user_names}")
    print(f"Sybil Users: {sybil_names}")
    
    return vm, user_map


# --- Parameterized Simulation Runner ---

def run_simulation(simulation_name: str, num_real_users: int, num_sybil_users: int, 
                   seed_vouches: list, attack_vouches: list, response_vouches: list):
    """
    Runs a full simulation based on a set of adjustable parameters.
    
    Args:
        simulation_name: Name for plots and logs (e.g., "Star Attack")
        num_real_users: Number of "real" users to create
        num_sybil_users: Number of "sybil" users to create
        seed_vouches: List of (from_name, to_name) tuples for initial graph
        attack_vouches: List of (from_name, to_name) tuples for sybil attack
        response_vouches: List of (from_name, to_name) tuples for community response
    """
    reset_globals()
    print(f"--- Starting VouchMinimal Simulation ({simulation_name}) ---")
    
    vm, user_map = setup_simulation(num_real_users, num_sybil_users)
    
    # --- 1. Initialization ---
    record_step(vm, user_map, "Initialization")
    
    # --- 2. Seed Vouches (from parameters) ---
    print("\n--- Phase 2: Seed Vouches ---")
    for i, (from_name, to_name) in enumerate(seed_vouches):
        if from_name not in user_map or to_name not in user_map:
            print(f"Warning: Skipping seed vouch. Unknown user: {from_name} or {to_name}")
            continue
        try:
            vm.vouch(user_map[from_name], user_map[to_name])
            event_name = f"Seed {i+1}: {from_name}->{to_name}"
            # Special tag for 5th seed vouch
            if (i + 1) == 5:
                event_name += " (5th)"
            record_step(vm, user_map, event_name)
        except ValueError as e:
            print(f"Warning: Seed vouch {from_name}->{to_name} failed: {e}")

    # --- 3. Sybil Attack (from parameters) ---
    print("\n--- Phase 3: Sybil Attack ---")
    for i, (from_name, to_name) in enumerate(attack_vouches):
        if from_name not in user_map or to_name not in user_map:
            print(f"Warning: Skipping attack vouch. Unknown user: {from_name} or {to_name}")
            continue
        try:
            vm.vouch(user_map[from_name], user_map[to_name])
            record_step(vm, user_map, f"Attack {i+1}: {from_name}->{to_name}")
        except ValueError as e:
            print(f"Warning: Attack vouch {from_name}->{to_name} failed: {e}")

    # --- 4. Community Response (from parameters) ---
    print("\n--- Phase 4: Community Response ---")
    for i, (from_name, to_name) in enumerate(response_vouches):
        if from_name not in user_map or to_name not in user_map:
            print(f"Warning: Skipping response vouch. Unknown user: {from_name} or {to_name}")
            continue
        try:
            vm.vouch(user_map[from_name], user_map[to_name])
            record_step(vm, user_map, f"Response {i+1}: {from_name}->{to_name}")
        except ValueError as e:
            print(f"Warning: Response vouch {from_name}->{to_name} failed: {e}")
    
    # --- 5. Plotting ---
    print("\n--- Phase 5: Generating Plots ---")
    
    # Sanitize name for filenames
    file_suffix = simulation_name.lower().replace(" ", "_").replace("/", "").replace("vs", "v")
    
    plot_history(f"contract_simulation_{file_suffix}.png")
    plot_graph_evolution_with_scores(
        graphs_history,
        scores_history,
        f"{simulation_name} Simulation Graph Evolution",
        layout_type="spring",
        filename=f"contract_graph_{file_suffix}.png"
    )
    
    print(f"\n--- {simulation_name} Simulation Complete ---")


# --- Main execution block with ADJUSTABLE PARAMETERS ---

if __name__ == "__main__":
    
    # --- 🅰️ Define Adjustable Parameters for first 3 simulations ---
    
    NUM_REAL_USERS_BASE = 5  # Alice, Bob, Charlie, David, Eve
    NUM_SYBIL_USERS_BASE = 3 # Sybil_1, Sybil_2, Sybil_3
    
    # Common parameters for the "trusted" part of the graph
    SEED_VOUCHES_BASE = [
        ("Alice", "Bob"),
        ("Alice", "Charlie"),
        ("Bob", "Charlie"),
        ("David", "Alice"),
        ("David", "Bob"),     # This is the 5th seed vouch
        ("Charlie", "Alice")
    ]
    
    # Common parameters for the community's reaction
    RESPONSE_VOUCHES_BASE = [
        ("David", "Eve"),
        ("Alice", "David")
    ]
    
    # --- 1️⃣ Simulation 1: Mixed Attack (Original) ---
    ATTACK_MIXED = [
        ("Sybil_1", "Eve"),
        ("Sybil_2", "Eve"),
        ("Sybil_3", "Eve"),
        ("Sybil_1", "Sybil_2"),
        ("Sybil_2", "Sybil_3"),
        ("Eve", "Sybil_1")
    ]
    run_simulation(
        simulation_name="Original Mixed Attack",
        num_real_users=NUM_REAL_USERS_BASE,
        num_sybil_users=NUM_SYBIL_USERS_BASE,
        seed_vouches=SEED_VOUCHES_BASE,
        attack_vouches=ATTACK_MIXED,
        response_vouches=RESPONSE_VOUCHES_BASE
    )

    # --- 2️⃣ Simulation 2: Star Attack ---
    ATTACK_STAR = [
        ("Sybil_1", "Eve"),
        ("Sybil_2", "Eve"),
        ("Sybil_3", "Eve")
    ]
    run_simulation(
        simulation_name="Star Attack",
        num_real_users=NUM_REAL_USERS_BASE,
        num_sybil_users=NUM_SYBIL_USERS_BASE,
        seed_vouches=SEED_VOUCHES_BASE,
        attack_vouches=ATTACK_STAR,
        response_vouches=RESPONSE_VOUCHES_BASE
    )

    # --- 3️⃣ Simulation 3: chain (Chain) Attack ---
    ATTACK_chain = [
        ("Sybil_1", "Sybil_2"),
        ("Sybil_2", "Sybil_3"),
        ("Sybil_3", "Eve")
    ]
    run_simulation(
        simulation_name="chain (Chain) Attack",
        num_real_users=NUM_REAL_USERS_BASE,
        num_sybil_users=NUM_SYBIL_USERS_BASE,
        seed_vouches=SEED_VOUCHES_BASE,
        attack_vouches=ATTACK_chain,
        response_vouches=RESPONSE_VOUCHES_BASE
    )
    
    # --- 4️⃣ Simulation 4: N Real Users vs M Attackers (Star) ---
    
    # --- 🅱️ Define Adjustable Parameters for N vs M ---
    N_REAL_USERS = 10
    M_ATTACKERS = 6
    
    # --- Seed Vouches: Create a "trusted" ring/cycle ---
    # (User_1 -> User_2 -> ... -> User_N -> User_1)
    # Ensure at least 5 vouches to pass the seed phase.
    
    seed_vouches_nxm = []
    user_names_nxm = [f"User_{i+1}" for i in range(N_REAL_USERS)] # Procedural names
    
    if N_REAL_USERS > 0:
        for i in range(N_REAL_USERS):
            from_name = user_names_nxm[i]
            to_name = user_names_nxm[(i + 1) % N_REAL_USERS] # Wraps around
            seed_vouches_nxm.append((from_name, to_name))
            
    # Add extra vouches if N < 5 to ensure we pass the seed phase
    if N_REAL_USERS > 1 and len(seed_vouches_nxm) < 5:
        # Add some random cross-vouches
        extra_vouches_needed = 5 - len(seed_vouches_nxm)
        for _ in range(extra_vouches_needed):
            from_name = random.choice(user_names_nxm)
            to_name = random.choice(user_names_nxm)
            # Ensure it's not a self-vouch and not a duplicate
            if from_name != to_name and (from_name, to_name) not in seed_vouches_nxm:
                 seed_vouches_nxm.append((from_name, to_name))
    
    # --- Attack Vouches: M attackers in a STAR formation targeting User_1 ---
    attack_vouches_nxm = []
    target_user = "User_1"
    for i in range(M_ATTACKERS):
        attacker_name = f"Sybil_{i + 1}"
        attack_vouches_nxm.append((attacker_name, target_user))

    # --- Response Vouches: Other trusted users vouch for the target ---
    response_vouches_nxm = []
    if "User_2" in user_names_nxm:
        response_vouches_nxm.append(("User_2", target_user))
    if "User_3" in user_names_nxm:
        response_vouches_nxm.append(("User_3", target_user))
        
    run_simulation(
        simulation_name=f"{N_REAL_USERS} Users vs {M_ATTACKERS} Attackers (Star)",
        num_real_users=N_REAL_USERS,
        num_sybil_users=M_ATTACKERS,
        seed_vouches=seed_vouches_nxm,
        attack_vouches=attack_vouches_nxm,
        response_vouches=response_vouches_nxm
    )

    # --- 5️⃣ Simulation 5: N Real Users vs M Attackers (Chain) ---
    
    # --- Attack Vouches: M attackers in a chain targeting User_1 ---
    # (Sybil_1 -> Sybil_2 -> ... -> Sybil_M -> User_1)
    
    attack_vouches_nxm_chain = []
    sybil_names_nxm = [f"Sybil_{i+1}" for i in range(M_ATTACKERS)]
    
    if M_ATTACKERS > 0:
        # Create the chain part
        for i in range(M_ATTACKERS - 1):
            from_name = sybil_names_nxm[i]
            to_name = sybil_names_nxm[i+1]
            attack_vouches_nxm_chain.append((from_name, to_name))
        
        # The last sybil attacks the target user
        last_sybil = sybil_names_nxm[-1] # e.g., Sybil_M
        target_user = "User_1"
        attack_vouches_nxm_chain.append((last_sybil, target_user))

    # We re-use the same seed and response vouches from Simulation 4
    run_simulation(
        simulation_name=f"{N_REAL_USERS} Users vs {M_ATTACKERS} Attackers (chain)",
        num_real_users=N_REAL_USERS,
        num_sybil_users=M_ATTACKERS,
        seed_vouches=seed_vouches_nxm, # Same trusted graph
        attack_vouches=attack_vouches_nxm_chain,
        response_vouches=response_vouches_nxm # Same community response
    )

# --- 6️⃣ Simulation 6: N Real Users vs M Attackers (Dandelion Attack) ---
    
    # --- Attack Vouches: M attackers in a Dandelion formation ---
    # 1. "Stem": Sybil_1 attacks User_1
    # 2. "Head": All M Sybils form a fully-connected, double-vouched clique.
    
    attack_vouches_dandelion = []
    sybil_names_dandelion = [f"Sybil_{i+1}" for i in range(M_ATTACKERS)]
    target_user = "User_1"

    if M_ATTACKERS > 0:
        # 1. The "Stem"
        attack_vouches_dandelion.append((sybil_names_dandelion[0], target_user)) # (Sybil_1 -> User_1)
        
        # 2. The "Head" (Clique)
        # Iterate over all unique pairs of attackers
        for i in range(M_ATTACKERS):
            for j in range(i + 1, M_ATTACKERS):
                attacker_a = sybil_names_dandelion[i]
                attacker_b = sybil_names_dandelion[j]
                
                # Add reciprocal vouches
                attack_vouches_dandelion.append((attacker_a, attacker_b))
                attack_vouches_dandelion.append((attacker_b, attacker_a))

    # We re-use the same seed and response vouches from Simulation 4 & 5
    run_simulation(
        simulation_name=f"{N_REAL_USERS} Users vs {M_ATTACKERS} Attackers (Dandelion)",
        num_real_users=N_REAL_USERS,
        num_sybil_users=M_ATTACKERS,
        seed_vouches=seed_vouches_nxm, # Same trusted graph
        attack_vouches=attack_vouches_dandelion,
        response_vouches=response_vouches_nxm # Same community response
    )

    # --- 7️⃣ Simulation 7: Mixed Attack (Malicious Seed 'Eve') ---
    # Eve colludes with Sybils to boost Sybil_1
    
    ATTACK_MIXED_MALICIOUS = [
        ("Eve", "Sybil_1"),      # Malicious seed vouches for target
        ("Sybil_2", "Sybil_1"),    # Other sybils vouch for target
        ("Sybil_3", "Sybil_1"),
        ("Eve", "Sybil_2"),      # Malicious seed boosts other sybils
        ("Sybil_2", "Sybil_3"),    # Sybils vouch internally
        ("Sybil_3", "Eve")       # Sybil vouches back to malicious seed
    ]
    
    run_simulation(
        simulation_name="Mixed Attack (Malicious Eve)",
        num_real_users=NUM_REAL_USERS_BASE,
        num_sybil_users=NUM_SYBIL_USERS_BASE,
        seed_vouches=SEED_VOUCHES_BASE,
        attack_vouches=ATTACK_MIXED_MALICIOUS,
        response_vouches=RESPONSE_VOUCHES_BASE
    )

    # --- 8️⃣ Simulation 8: Star Attack (Malicious Seed 'Eve') ---
    # Eve and Sybils form a star, all vouching for Sybil_1
    
    ATTACK_STAR_MALICIOUS = [
        ("Eve", "Sybil_1"),      # Malicious seed
        ("Sybil_2", "Sybil_1"),    # Sybils
        ("Sybil_3", "Sybil_1")
    ]
    
    run_simulation(
        simulation_name="Star Attack (Malicious Eve)",
        num_real_users=NUM_REAL_USERS_BASE,
        num_sybil_users=NUM_SYBIL_USERS_BASE,
        seed_vouches=SEED_VOUCHES_BASE,
        attack_vouches=ATTACK_STAR_MALICIOUS,
        response_vouches=RESPONSE_VOUCHES_BASE
    )

    # --- 9️⃣ Simulation 9: Chain (Chain) Attack (Malicious Seed 'Eve') ---
    # Eve starts a vouch chain to boost Sybil_1
    
    ATTACK_CHAIN_MALICIOUS = [
        ("Eve", "Sybil_2"),
        ("Sybil_2", "Sybil_3"),
        ("Sybil_3", "Sybil_1")   # Chain ends by vouching for target
    ]
    
    run_simulation(
        simulation_name="Chain Attack (Malicious Eve)",
        num_real_users=NUM_REAL_USERS_BASE,
        num_sybil_users=NUM_SYBIL_USERS_BASE,
        seed_vouches=SEED_VOUCHES_BASE,
        attack_vouches=ATTACK_CHAIN_MALICIOUS,
        response_vouches=RESPONSE_VOUCHES_BASE
    )

    # --- 10️⃣ Simulation 10: N vs M Star Attack (Malicious Seed 'User_N') ---
    
    # --- Attack Vouches: Star formation targeting Sybil_1 ---
    # Malicious 'User_N' and all other sybils vouch for Sybil_1
    
    attack_vouches_nxm_star_malicious = []
    sybil_names_nxm = [f"Sybil_{i+1}" for i in range(M_ATTACKERS)]
    malicious_seed_name = f"User_{N_REAL_USERS}" # e.g., User_20
    attack_target_name = "Sybil_1"

    if M_ATTACKERS > 0 and N_REAL_USERS > 0:
        # 1. Malicious seed vouches for the target
        attack_vouches_nxm_star_malicious.append((malicious_seed_name, attack_target_name))
        
        # 2. All *other* sybils (Sybil_2...Sybil_M) vouch for the target
        for i in range(1, M_ATTACKERS): # Start from index 1 (Sybil_2)
            attacker_name = sybil_names_nxm[i]
            attack_vouches_nxm_star_malicious.append((attacker_name, attack_target_name))

    run_simulation(
        simulation_name=f"{N_REAL_USERS} vs {M_ATTACKERS} Star (Malicious User_{N_REAL_USERS})",
        num_real_users=N_REAL_USERS,
        num_sybil_users=M_ATTACKERS,
        seed_vouches=seed_vouches_nxm,
        attack_vouches=attack_vouches_nxm_star_malicious,
        response_vouches=response_vouches_nxm 
    )

    # --- 11️⃣ Simulation 11: N vs M Chain Attack (Malicious Seed 'User_N') ---

    # --- Attack Vouches: Chain starting from User_N ---
    # (User_N -> Sybil_M -> ... -> Sybil_2 -> Sybil_1)
    
    attack_vouches_nxm_chain_malicious = []
    
    if M_ATTACKERS > 0 and N_REAL_USERS > 0:
        # 1. Malicious seed starts the chain
        attack_vouches_nxm_chain_malicious.append((malicious_seed_name, sybil_names_nxm[-1])) # User_N -> Sybil_M
        
        # 2. Internal Sybil chain
        # (Sybil_M -> Sybil_M-1) ... (Sybil_2 -> Sybil_1)
        for i in range(M_ATTACKERS - 1, 0, -1): # Iterate from M-1 down to 1
            from_name = sybil_names_nxm[i]   # e.g., Sybil_M
            to_name = sybil_names_nxm[i-1] # e.g., Sybil_M-1
            attack_vouches_nxm_chain_malicious.append((from_name, to_name))

    run_simulation(
        simulation_name=f"{N_REAL_USERS} vs {M_ATTACKERS} Chain (Malicious User_{N_REAL_USERS})",
        num_real_users=N_REAL_USERS,
        num_sybil_users=M_ATTACKERS,
        seed_vouches=seed_vouches_nxm,
        attack_vouches=attack_vouches_nxm_chain_malicious,
        response_vouches=response_vouches_nxm 
    )

    # --- 12️⃣ Simulation 12: N vs M Dandelion Attack (Malicious Seed 'User_N') ---
    
    # --- Attack Vouches: Dandelion formation ---
    # 1. "Stem": Malicious User_N vouches for Sybil_1
    # 2. "Head": All M Sybils form a fully-connected, double-vouched clique.
    
    attack_vouches_nxm_dandelion_malicious = []

    if M_ATTACKERS > 0 and N_REAL_USERS > 0:
        # 1. The "Stem" from the trusted graph
        attack_vouches_nxm_dandelion_malicious.append((malicious_seed_name, attack_target_name)) # (User_N -> Sybil_1)
        
        # 2. The "Head" (Clique)
        for i in range(M_ATTACKERS):
            for j in range(i + 1, M_ATTACKERS):
                attacker_a = sybil_names_nxm[i]
                attacker_b = sybil_names_nxm[j]
                attack_vouches_nxm_dandelion_malicious.append((attacker_a, attacker_b))
                attack_vouches_nxm_dandelion_malicious.append((attacker_b, attacker_a))

    run_simulation(
        simulation_name=f"{N_REAL_USERS} vs {M_ATTACKERS} Dandelion (Malicious User_{N_REAL_USERS})",
        num_real_users=N_REAL_USERS,
        num_sybil_users=M_ATTACKERS,
        seed_vouches=seed_vouches_nxm,
        attack_vouches=attack_vouches_nxm_dandelion_malicious,
        response_vouches=response_vouches_nxm 
    )