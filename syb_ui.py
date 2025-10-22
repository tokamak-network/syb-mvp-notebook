"""SYB Network User Interface Components"""

import ipywidgets as widgets
from IPython.display import display, clear_output
from contract_interface import SYBContract, SYBUser
from network import Network
from utils import generate_mul_eth_addresses

def create_random_network(num_users=8, random_name='erdos_renyi', balance_range=(0.0, 0.1)) -> tuple:
    """Create a random network, initialize contract, and set up user interfaces."""
    print(f"📊 Creating '{random_name}' network with {num_users} users...")
    network = Network.init_random(n_nodes=num_users, random_name=random_name, balance_range=balance_range)
    contract = SYBContract(scoring_algorithm='random_walk')
    contract.network = network

    initial_addresses = generate_mul_eth_addresses(num_users)
    user_names = [f"User {i}" for i in range(num_users)]

    print("\n💰 Funding user accounts from network balances...")
    for i, addr in enumerate(initial_addresses):
        balance_eth = network.balance_list[i]
        deposit_amount = int(balance_eth * 10**18)
        contract.deposit(addr, deposit_amount)

    print("\n🤝 Creating vouching network from existing graph...")
    for u, v in network.graph.edges():
        addr1, addr2 = initial_addresses[u], initial_addresses[v]
        contract.vouch(addr1, addr2)

    print("\n⚡ Processing initial batch to establish scores...")
    try:
        contract.forge_batch()
        print("✅ Initial batch processed successfully!")
    except Exception as e:
        print(f"⚠️ Initial batch processing failed: {e}")

    users = {}
    for i, addr in enumerate(initial_addresses):
        users[addr] = {
            'interface': SYBUser(contract, addr),
            'name': user_names[i],
            'address': addr
        }
    
    return contract, users

class SYBUserInterface:
    """Main UI class for SYB Network interaction."""

    def __init__(self, contract: SYBContract, users: dict):
        """Initialize the SYB UI interface."""
        self.contract = contract
        self.users = users
        self.first_user_addr = list(users.keys())[0]
        self.first_user = users[self.first_user_addr]
        self.user_interface = self.first_user['interface']
        
        self._create_widgets()
        self._create_interface()
        self._connect_events()

    def _create_widgets(self):
        """Create UI widgets."""
        self.deposit_amount = widgets.FloatText(value=1.0, description='Amount (ETH):')
        self.withdraw_amount = widgets.FloatText(value=0.5, description='Amount (ETH):')
        self.vouch_target = widgets.Dropdown(options=self._get_other_users(), description='Target User:')
        self.unvouch_target = widgets.Dropdown(options=[], description='Target User:')
        
        self.deposit_btn = widgets.Button(description='💰 Deposit', button_style='primary')
        self.withdraw_btn = widgets.Button(description='💸 Withdraw', button_style='warning')
        self.vouch_btn = widgets.Button(description='👍 Vouch', button_style='success')
        self.unvouch_btn = widgets.Button(description='👎 Unvouch', button_style='danger')
        self.balance_btn = widgets.Button(description='📊 Balance', button_style='info')
        self.network_btn = widgets.Button(description='🌐 Network', button_style='info')
        self.forge_batch_btn = widgets.Button(description='⚡ Forge Batch', button_style='success')

        self.deposit_submit = widgets.Button(description='Submit', button_style='success')
        self.withdraw_submit = widgets.Button(description='Submit', button_style='success')
        self.vouch_submit = widgets.Button(description='Submit', button_style='success')
        self.unvouch_submit = widgets.Button(description='Submit', button_style='success')
        
        self.output_area = widgets.Output()

    def _get_other_users(self):
        """Get list of other users for dropdowns."""
        return [(f"{data['name']} ({addr[:10]}...)", addr)
                for addr, data in self.users.items() if addr != self.first_user_addr]

    def _create_interface(self):
        """Create main interface layout."""
        self.deposit_section = widgets.VBox([self.deposit_amount, self.deposit_submit], layout=widgets.Layout(display='none'))
        self.withdraw_section = widgets.VBox([self.withdraw_amount, self.withdraw_submit], layout=widgets.Layout(display='none'))
        self.vouch_section = widgets.VBox([self.vouch_target, self.vouch_submit], layout=widgets.Layout(display='none'))
        self.unvouch_section = widgets.VBox([self.unvouch_target, self.unvouch_submit], layout=widgets.Layout(display='none'))
        
        button_row1 = widgets.HBox([self.deposit_btn, self.withdraw_btn, self.vouch_btn, self.unvouch_btn])
        button_row2 = widgets.HBox([self.balance_btn, self.network_btn, self.forge_batch_btn])
        
        self.contract_interface = widgets.VBox([
            widgets.HTML(value="<h3>📋 SYB Contract Functions</h3>"),
            button_row1, button_row2,
            self.deposit_section, self.withdraw_section, self.vouch_section, self.unvouch_section,
            widgets.HTML(value="<h4>📤 Transaction Results</h4>"),
            self.output_area
        ], layout=widgets.Layout(border='2px solid #2E86AB', padding='15px'))

    def _connect_events(self):
        """Connect button events to handlers."""
        self.deposit_btn.on_click(self._show_deposit_input)
        self.withdraw_btn.on_click(self._show_withdraw_input)
        self.vouch_btn.on_click(self._show_vouch_input)
        self.unvouch_btn.on_click(self._show_unvouch_input)
        
        self.deposit_submit.on_click(self._handle_deposit)
        self.withdraw_submit.on_click(self._handle_withdraw)
        self.vouch_submit.on_click(self._handle_vouch)
        self.unvouch_submit.on_click(self._handle_unvouch)
        
        self.balance_btn.on_click(self._handle_balance)
        self.network_btn.on_click(self._handle_network)
        self.forge_batch_btn.on_click(self._handle_forge_batch)

    def toggle_inputs(self, show_section=None):
        """Show one input section and hide others."""
        for section in [self.deposit_section, self.withdraw_section, self.vouch_section, self.unvouch_section]:
            section.layout.display = 'block' if section == show_section else 'none'

    def _show_deposit_input(self, b): self.toggle_inputs(self.deposit_section)
    def _show_withdraw_input(self, b): self.toggle_inputs(self.withdraw_section)
    def _show_vouch_input(self, b): self.toggle_inputs(self.vouch_section)
    def _show_unvouch_input(self, b):
        self.update_unvouch_options()
        self.toggle_inputs(self.unvouch_section)

    def update_unvouch_options(self):
        """Update unvouch options based on current vouches."""
        vouches = self.contract.vouches.get(self.first_user_addr, {})
        self.unvouch_target.options = [(f"{self.users[addr]['name']} ({addr[:10]}...)", addr)
                                     for addr, is_vouched in vouches.items() if is_vouched]

    def _handle_deposit(self, b):
        with self.output_area:
            clear_output(wait=True)
            try:
                self.user_interface.deposit(int(self.deposit_amount.value * 10**18))
                print(f"✅ Deposited {self.deposit_amount.value:.2f} ETH.")
                # Don't refresh balance here - user can click balance button
            except Exception as e: print(f"❌ Deposit failed: {e}")

    def _handle_withdraw(self, b):
        with self.output_area:
            clear_output(wait=True)
            try:
                self.user_interface.withdraw(int(self.withdraw_amount.value * 10**18))
                print(f"✅ Withdrew {self.withdraw_amount.value:.2f} ETH.")
                # Don't refresh balance here - user can click balance button
            except Exception as e: print(f"❌ Withdraw failed: {e}")

    def _handle_vouch(self, b):
        with self.output_area:
            clear_output(wait=True)
            try:
                self.user_interface.vouch(self.vouch_target.value)
                print(f"✅ Vouched for {self.users[self.vouch_target.value]['name']}.")
            except Exception as e: print(f"❌ Vouch failed: {e}")

    def _handle_unvouch(self, b):
        with self.output_area:
            clear_output(wait=True)
            try:
                self.user_interface.unvouch(self.unvouch_target.value)
                print(f"✅ Unvouched {self.users[self.unvouch_target.value]['name']}.")
            except Exception as e: print(f"❌ Unvouch failed: {e}")

    def _handle_balance(self, b):
        # Clear previous output if called from button click
        if b:
            self.output_area.clear_output(wait=True)

        try:
            balance = self.user_interface.get_my_balance() / 10**18
            score = self.user_interface.get_my_score()
            with self.output_area:
                print(f"💰 Current Balance: {balance:.4f} ETH")
                print(f"🏆 Current Score: {score:.4f}")
                print(f"📍 Address: {self.first_user_addr}")
        except Exception as e:
            with self.output_area:
                print(f"❌ Error getting balance: {e}")

    def _handle_network(self, b):
        with self.output_area:
            clear_output(wait=True)
            print("🌐 Displaying Network Graph...")
            try:
                scores = self.contract.network.compute_score('pagerank')
                self.contract.network.display_graph(scores=scores, title="Current Network State")
            except Exception as e:
                print(f"❌ Error displaying network: {e}")

    def _handle_forge_batch(self, b):
        with self.output_area:
            clear_output(wait=True)
            print("⚡ Forging batch...")
            try:
                result = self.contract.forge_batch()
                if result:
                    print(f"✅ Batch {result.batch_num} forged with {len(result.transactions_processed)} txns.")
                    self._handle_network(None) # Refresh network view
                else:
                    print("⏳ Not enough transactions for a new batch.")
            except Exception as e: print(f"❌ Error forging batch: {e}")

    def display_network_status(self):
        """Display network status summary."""
        try:
            # Get network statistics
            num_users = len(self.users)
            scoring_algorithm = self.contract.scoring_algorithm
            pending_txns = len(self.contract.unprocessed_txns)

            # Calculate total deposits
            total_deposits = 0
            for user_data in self.users.values():
                try:
                    balance = user_data['interface'].get_my_balance() / 10**18
                    total_deposits += balance
                except:
                    pass

            # Get average score
            scores = []
            for user_data in self.users.values():
                try:
                    score = user_data['interface'].get_my_score()
                    scores.append(score)
                except:
                    pass
            avg_score = sum(scores) / len(scores) if scores else 0

            print("🌐 Network Status Summary")
            print("=" * 40)
            print(f"👥 Total Users: {num_users}")
            print(f"💰 Total Deposits: {total_deposits:.4f} ETH")
            print(f"📊 Scoring Algorithm: {scoring_algorithm}")
            print(f"⏳ Pending Transactions: {pending_txns}")
            print(f"📈 Average Score: {avg_score:.4f}")
            print(f"🔄 Last Batch: {self.contract.last_forged_batch}")

        except Exception as e:
            print(f"❌ Error getting network status: {e}")

    def display(self):
        """Display main contract interface."""
        print(f"🔐 Connected as: {self.first_user['name']} ({self.first_user_addr[:12]}...)")
        display(self.contract_interface)
