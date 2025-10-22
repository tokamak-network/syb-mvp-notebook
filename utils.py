import random
import threading
import time
from typing import List, Dict, Optional, Tuple
from contract_interface import TXN_DEPOSIT_NEW_ACCOUNT, TXN_DEPOSIT_EXISTING, TXN_WITHDRAW, TXN_VOUCH, TXN_UNVOUCH

# Fake ethereum address generator
def generate_random_eth_address():
    return f"0x{random.randint(0, 0xffffffffffffffffffffffffffffffffffffffff):040x}"

def generate_mul_eth_addresses(n):
    return [generate_random_eth_address() for _ in range(n)]


def format_address(address: str, prefix_len: int = 6, suffix_len: int = 4) -> str:
    """Format ethereum address for display: 0xab15...1234"""
    if not address or len(address) < prefix_len + suffix_len:
        return address
    return f"{address[:prefix_len]}...{address[-suffix_len:]}"


def get_transaction_type_name(txn_id: int) -> str:
    """Get human-readable transaction type name."""
    txn_names = {
        TXN_DEPOSIT_NEW_ACCOUNT: "CreateAccountDeposit",
        TXN_DEPOSIT_EXISTING: "Deposit",
        TXN_WITHDRAW: "Withdraw",
        TXN_VOUCH: "Vouch",
        TXN_UNVOUCH: "Unvouch"
    }
    return txn_names.get(txn_id, "Unknown")


def format_transaction_display(txn, users: Dict, index: int = None) -> str:
    """
    Format transaction for display in queue/batch.

    Args:
        txn: Transaction object
        users: Dictionary of user information
        index: Optional index number for display

    Returns:
        Formatted string like: "(1) Deposit 0xab15...12 (User 1) 1.5 ETH"
    """
    from contract_interface import TXN_DEPOSIT_NEW_ACCOUNT, TXN_DEPOSIT_EXISTING, TXN_WITHDRAW, TXN_VOUCH, TXN_UNVOUCH

    # Find user info from transaction index
    from_addr = None
    to_addr = None
    from_name = "Unknown"
    to_name = "Unknown"

    # Search for addresses by idx
    for addr, data in users.items():
        user_idx = data['interface'].contract.address_to_idx.get(addr)
        if user_idx == txn.from_idx:
            from_addr = addr
            from_name = data['name']
        if user_idx == txn.to_idx:
            to_addr = addr
            to_name = data['name']

    txn_type = get_transaction_type_name(txn.identifier)
    prefix = f"({index}) " if index is not None else ""

    if txn.identifier in (TXN_DEPOSIT_NEW_ACCOUNT, TXN_DEPOSIT_EXISTING):
        amount_eth = txn.amount / 10**18
        from_display = format_address(from_addr) if from_addr else "Unknown"
        return f"{prefix}{txn_type} {from_display} ({from_name}) {amount_eth:.4f} ETH"

    elif txn.identifier == TXN_WITHDRAW:
        amount_eth = txn.amount / 10**18
        from_display = format_address(from_addr) if from_addr else "Unknown"
        return f"{prefix}{txn_type} {from_display} ({from_name}) {amount_eth:.4f} ETH"

    elif txn.identifier in (TXN_VOUCH, TXN_UNVOUCH):
        from_display = format_address(from_addr) if from_addr else "Unknown"
        to_display = format_address(to_addr) if to_addr else "Unknown"
        return f"{prefix}{txn_type} {from_display} ({from_name}) >> {to_display} ({to_name})"

    return f"{prefix}Unknown Transaction"


class TransactionSimulator:
    """Manages background simulation of random transactions."""

    def __init__(self, contract, users: Dict, connected_user_addr: str, interval: int = 10):
        """
        Initialize transaction simulator.

        Args:
            contract: SYBContract instance
            users: Dictionary of user information
            connected_user_addr: Address of the connected user (to exclude from simulation)
            interval: Seconds between simulated transactions
        """
        self.contract = contract
        self.users = users
        self.connected_user_addr = connected_user_addr
        self.interval = interval
        self.running = False
        self.thread = None

        # Get list of other users (exclude connected user)
        self.other_users = [
            (addr, data) for addr, data in users.items()
            if addr != connected_user_addr
        ]

    def _simulation_loop(self):
        """Main simulation loop running in background thread."""
        while self.running:
            time.sleep(self.interval)
            if self.running:  # Check again after sleep
                try:
                    self._generate_random_transaction()
                except Exception as e:
                    print(f"Simulation error: {e}")

    def _generate_random_transaction(self):
        """Generate a random transaction from a random user."""
        if not self.other_users:
            return

        # Select random user
        user_addr, user_data = random.choice(self.other_users)
        user_interface = user_data['interface']

        # Select random transaction type
        txn_types = ['deposit', 'withdraw', 'vouch', 'unvouch']
        txn_type = random.choice(txn_types)

        try:
            if txn_type == 'deposit':
                # Random deposit between 0.1 and 2.0 ETH
                amount = random.uniform(0.1, 2.0)
                user_interface.deposit(int(amount * 10**18))

            elif txn_type == 'withdraw':
                # Try to withdraw random amount (up to 50% of balance)
                balance = user_interface.get_my_balance()
                if balance > 0:
                    max_withdraw = balance * 0.5
                    amount = random.uniform(0.1 * 10**18, min(max_withdraw, 1.0 * 10**18))
                    user_interface.withdraw(int(amount))

            elif txn_type == 'vouch':
                # Vouch for a random other user
                possible_targets = [
                    addr for addr in self.users.keys()
                    if addr != user_addr and addr != self.connected_user_addr
                ]
                if possible_targets:
                    target = random.choice(possible_targets)
                    # Check if already vouched
                    vouches = self.contract.vouches.get(user_addr, {})
                    if not vouches.get(target, False):
                        user_interface.vouch(target)

            elif txn_type == 'unvouch':
                # Unvouch a random existing vouch
                vouches = self.contract.vouches.get(user_addr, {})
                active_vouches = [addr for addr, is_vouched in vouches.items() if is_vouched]
                if active_vouches:
                    target = random.choice(active_vouches)
                    user_interface.unvouch(target)

        except Exception as e:
            # Silently ignore transaction errors (e.g., insufficient balance)
            pass

    def start(self):
        """Start the simulation thread."""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._simulation_loop, daemon=True)
            self.thread.start()

    def stop(self):
        """Stop the simulation thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
