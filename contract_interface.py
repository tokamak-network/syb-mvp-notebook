from typing import Dict, List, Optional
from dataclasses import dataclass
from network import Network

# Transaction Types
TXN_DEPOSIT_NEW_ACCOUNT = 1
TXN_DEPOSIT_EXISTING = 2
TXN_WITHDRAW = 3
TXN_VOUCH = 4
TXN_UNVOUCH = 5


@dataclass
class Transaction:
    """Represents a transaction in the queue."""
    identifier: int
    from_idx: int
    to_idx: int
    amount: int


@dataclass
class AccountInfo:
    """Account information stored in contract."""
    balance: int
    idx: int


@dataclass
class ScoreSnapshot:
    """Score snapshot for historical tracking."""
    score: int
    batch_num: int


@dataclass
class BatchResult:
    """Result of processing a batch."""
    batch_num: int
    transactions_processed: List[Transaction]
    scores: List[float]


class SYBContract:
    """
    SYB (Sybil Resistance) Contract Interface Simulation.

    Simulates a blockchain contract that implements sybil resistance through social
    vouching and algorithmic reputation scoring. Users can deposit/withdraw funds,
    vouch/unvouch for other users, and the system calculates reputation scores.
    """

    def __init__(self, scoring_algorithm: str = 'pagerank'):
        """Initialize the SYB contract with specified scoring algorithm."""
        self.batch_size = 5
        self.last_idx = 0
        self.last_forged_batch = 0
        self.account_info: Dict[str, AccountInfo] = {}
        self.vouches: Dict[str, Dict[str, bool]] = {}
        self.score_snapshots: Dict[str, ScoreSnapshot] = {}
        self.unprocessed_txns: List[Transaction] = []
        self.batches: Dict[int, BatchResult] = {}  # Store batch history
        self.network = Network()
        self.scoring_algorithm = scoring_algorithm
        self.address_to_idx: Dict[str, int] = {}
        self.idx_to_address: Dict[int, str] = {}

    def _get_or_create_account_idx(self, address: str) -> int:
        """Get or create internal index for an address."""
        if address not in self.address_to_idx:
            idx = self.last_idx
            self.address_to_idx[address] = idx
            self.idx_to_address[idx] = address
            self.network.add_node(idx, balance=0.0)
            self.account_info[address] = AccountInfo(balance=0, idx=idx)
            self.last_idx += 1
            return idx
        return self.address_to_idx[address]

    def _add_tx(self, identifier: int, from_addr: str, to_addr: Optional[str], amount: int):
        """Add a transaction to the queue."""
        from_idx = self._get_or_create_account_idx(from_addr)
        to_idx = self._get_or_create_account_idx(to_addr) if to_addr else 0
        self.unprocessed_txns.append(Transaction(identifier, from_idx, to_idx, amount))

    def deposit(self, depositor: str, amount: int):
        """Deposit funds to the contract."""
        if amount <= 0:
            raise ValueError("Deposit amount must be positive.")

        has_existing_balance = (
            depositor in self.account_info and
            self.account_info[depositor].balance > 0
        )
        transaction_type = TXN_DEPOSIT_EXISTING if has_existing_balance else TXN_DEPOSIT_NEW_ACCOUNT

        self._add_tx(transaction_type, depositor, None, amount)

    def withdraw(self, withdrawer: str, amount: int):
        """Withdraw funds from the contract."""
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive.")
        if withdrawer not in self.account_info or self.account_info[withdrawer].balance < amount:
            raise ValueError("Insufficient balance for withdrawal.")
        self._add_tx(TXN_WITHDRAW, withdrawer, None, amount)

    def vouch(self, voucher: str, vouchee: str):
        """Create a trust relationship between two users."""
        if voucher == vouchee:
            raise ValueError("Cannot vouch for yourself.")
        self._add_tx(TXN_VOUCH, voucher, vouchee, 0)

    def unvouch(self, unvoucher: str, unvouchee: str):
        """Remove a trust relationship between two users."""
        self._add_tx(TXN_UNVOUCH, unvoucher, unvouchee, 0)

    def _process_transaction(self, txn: Transaction):
        """Execute a single transaction, updating contract state."""
        from_addr = self.idx_to_address.get(txn.from_idx)
        to_addr = self.idx_to_address.get(txn.to_idx)

        if not from_addr:
            return  # Invalid transaction

        if txn.identifier in (TXN_DEPOSIT_NEW_ACCOUNT, TXN_DEPOSIT_EXISTING):
            # Update account balance and network node balance
            self.account_info[from_addr].balance += txn.amount
            balance_in_eth = self.account_info[from_addr].balance / 10**18
            self.network.set_balance(txn.from_idx, balance_in_eth)

        elif txn.identifier == TXN_WITHDRAW:
            # Update account balance and network node balance
            self.account_info[from_addr].balance -= txn.amount
            balance_in_eth = self.account_info[from_addr].balance / 10**18
            self.network.set_balance(txn.from_idx, balance_in_eth)

        elif txn.identifier == TXN_VOUCH and to_addr:
            # Create bidirectional trust relationship
            self._ensure_vouch_relationship_exists(from_addr, to_addr)
            self.vouches[from_addr][to_addr] = True
            self.vouches[to_addr][from_addr] = True
            self.network.add_edge(txn.from_idx, txn.to_idx)

        elif txn.identifier == TXN_UNVOUCH and to_addr:
            # Remove trust relationship
            if from_addr in self.vouches:
                self.vouches[from_addr][to_addr] = False
            if to_addr in self.vouches:
                self.vouches[to_addr][from_addr] = False
            if self.network.graph.has_edge(txn.from_idx, txn.to_idx):
                self.network.remove_edge(txn.from_idx, txn.to_idx)

    def _ensure_vouch_relationship_exists(self, addr1: str, addr2: str):
        """Ensure vouch dictionaries exist for both addresses."""
        if addr1 not in self.vouches:
            self.vouches[addr1] = {}
        if addr2 not in self.vouches:
            self.vouches[addr2] = {}

    def forge_batch(self) -> Optional[BatchResult]:
        """Process a batch of pending transactions and update reputation scores."""
        pending_count = len(self.unprocessed_txns)
        if pending_count < self.batch_size:
            return None

        # Extract transactions for this batch
        batch_transactions = self.unprocessed_txns[:self.batch_size]
        self.unprocessed_txns = self.unprocessed_txns[self.batch_size:]

        # Process each transaction in the batch
        for transaction in batch_transactions:
            self._process_transaction(transaction)

        # Update batch counter and compute new scores
        self.last_forged_batch += 1
        new_scores = self.network.compute_score(self.scoring_algorithm)

        # Store score snapshots for each user
        for node_index, node_id in enumerate(self.network._node_order):
            user_address = self.idx_to_address.get(node_id)
            if user_address and node_index < len(new_scores):
                scaled_score = int(new_scores[node_index] * 2**32)
                self.score_snapshots[user_address] = ScoreSnapshot(scaled_score, self.last_forged_batch)

        # Create and store batch result
        batch_result = BatchResult(self.last_forged_batch, batch_transactions, new_scores)
        self.batches[self.last_forged_batch] = batch_result

        return batch_result

    def get_account_info(self, address: str) -> Optional[AccountInfo]:
        """Get account information for an address."""
        return self.account_info.get(address)

    def get_score_snapshot(self, address: str) -> Optional[ScoreSnapshot]:
        """Get the most recent reputation score for a user."""
        return self.score_snapshots.get(address)

    def get_pending_transactions(self) -> List[Transaction]:
        """Get the list of unprocessed transactions."""
        return self.unprocessed_txns

class SYBUser:
    """User interface for interacting with the SYB contract."""

    def __init__(self, contract: SYBContract, user_address: str):
        """Create a user interface for contract interactions."""
        self.contract = contract
        self.user_address = user_address

    def deposit(self, amount: int):
        """Deposit funds on behalf of this user."""
        self.contract.deposit(self.user_address, amount)

    def withdraw(self, amount: int):
        """Withdraw funds on behalf of this user."""
        self.contract.withdraw(self.user_address, amount)

    def vouch(self, target_address: str):
        """Vouch for another user on behalf of this user."""
        self.contract.vouch(self.user_address, target_address)

    def unvouch(self, target_address: str):
        """Remove vouch for another user on behalf of this user."""
        self.contract.unvouch(self.user_address, target_address)

    def get_my_balance(self) -> int:
        """Get this user's current balance in wei."""
        account_info = self.contract.get_account_info(self.user_address)
        return account_info.balance if account_info else 0

    def get_my_score(self) -> float:
        """Get this user's current reputation score."""
        score_snapshot = self.contract.get_score_snapshot(self.user_address)
        return score_snapshot.score / 2**32 if score_snapshot else 0.0