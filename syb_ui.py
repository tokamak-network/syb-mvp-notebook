"""SYB Network User Interface Components"""

import ipywidgets as widgets
from IPython.display import display, clear_output
import threading
import asyncio
from contract_interface import SYBContract, SYBUser
from network import Network
from utils import (
    generate_mul_eth_addresses,
    TransactionSimulator,
    format_transaction_display
)

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

        # Simulation state
        self.simulator = None
        self.batch_processing = False
        self.current_batch_transactions = []
        self.last_batch_scores = None  # Store scores from last batch

        self._create_widgets()
        self._create_interface()
        self._connect_events()

        # Hook contract methods to trigger UI updates
        self._hook_contract_updates()

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

        self.deposit_submit = widgets.Button(description='Submit', button_style='success')
        self.withdraw_submit = widgets.Button(description='Submit', button_style='success')
        self.vouch_submit = widgets.Button(description='Submit', button_style='success')
        self.unvouch_submit = widgets.Button(description='Submit', button_style='success')

        # Queue and batch display widgets
        self.queue_display = widgets.HTML(value="<div style='padding: 10px;'>Queue empty</div>")
        self.batch_display = widgets.HTML(value="<div style='padding: 10px;'>No batch processing</div>")

        # Status display widgets
        self.last_batch_display = widgets.HTML(value="<div style='padding: 10px;'>No batches forged yet</div>")
        self.current_status_display = widgets.HTML(value="<div style='padding: 10px;'>Loading...</div>")

        # Network graph outputs - side by side
        self.last_batch_graph_output = widgets.Output()
        self.current_graph_output = widgets.Output()

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

        button_row = widgets.HBox([self.deposit_btn, self.withdraw_btn, self.vouch_btn, self.unvouch_btn, self.balance_btn])

        # Top row: Queue and Batch displays
        queue_box = widgets.VBox([
            widgets.HTML(value="<h4 style='margin: 0; padding: 5px; background: #f0f0f0;'>📋 Transaction Queue</h4>"),
            self.queue_display
        ], layout=widgets.Layout(border='2px solid #4CAF50', padding='0px', width='48%'))

        batch_box = widgets.VBox([
            widgets.HTML(value="<h4 style='margin: 0; padding: 5px; background: #f0f0f0;'>⚡ Batch Forging</h4>"),
            self.batch_display
        ], layout=widgets.Layout(border='2px solid #FF9800', padding='0px', width='48%'))

        top_row = widgets.HBox([queue_box, batch_box], layout=widgets.Layout(justify_content='space-between'))

        # Middle row: Last Batch and Current Status
        last_batch_box = widgets.VBox([
            widgets.HTML(value="<h4 style='margin: 0; padding: 5px; background: #f0f0f0;'>📊 Last Batch Status</h4>"),
            self.last_batch_display
        ], layout=widgets.Layout(border='2px solid #2196F3', padding='0px', width='48%'))

        current_status_box = widgets.VBox([
            widgets.HTML(value="<h4 style='margin: 0; padding: 5px; background: #f0f0f0;'>📈 Current Status</h4>"),
            self.current_status_display
        ], layout=widgets.Layout(border='2px solid #9C27B0', padding='0px', width='48%'))

        middle_row = widgets.HBox([last_batch_box, current_status_box], layout=widgets.Layout(justify_content='space-between'))

        # User action interface
        user_actions = widgets.VBox([
            widgets.HTML(value="<h3>👤 User Actions</h3>"),
            button_row,
            self.deposit_section, self.withdraw_section, self.vouch_section, self.unvouch_section,
            self.output_area
        ], layout=widgets.Layout(border='2px solid #2E86AB', padding='15px'))

        # Graph row: Last Batch vs Current
        last_graph_box = widgets.VBox([
            widgets.HTML(value="<h4 style='margin: 0; padding: 5px; background: #f0f0f0; text-align: center;'>📊 Last Batch Network</h4>"),
            self.last_batch_graph_output
        ], layout=widgets.Layout(border='2px solid #2196F3', padding='0px', width='48%'))

        current_graph_box = widgets.VBox([
            widgets.HTML(value="<h4 style='margin: 0; padding: 5px; background: #f0f0f0; text-align: center;'>🌐 Current Network</h4>"),
            self.current_graph_output
        ], layout=widgets.Layout(border='2px solid #4CAF50', padding='0px', width='48%'))

        graph_row = widgets.HBox([last_graph_box, current_graph_box], layout=widgets.Layout(justify_content='space-between'))

        # Main layout
        self.contract_interface = widgets.VBox([
            top_row,
            middle_row,
            user_actions,
            widgets.HTML(value="<h3 style='margin-top: 20px;'>🔄 Network Comparison</h3>"),
            graph_row
        ])

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

    def _hook_contract_updates(self):
        """Hook into contract methods to trigger UI updates."""
        # Store original methods
        original_deposit = self.contract.deposit
        original_withdraw = self.contract.withdraw
        original_vouch = self.contract.vouch
        original_unvouch = self.contract.unvouch

        # Wrap methods to trigger updates
        def wrapped_deposit(*args, **kwargs):
            result = original_deposit(*args, **kwargs)
            self._check_and_update_ui()
            return result

        def wrapped_withdraw(*args, **kwargs):
            result = original_withdraw(*args, **kwargs)
            self._check_and_update_ui()
            return result

        def wrapped_vouch(*args, **kwargs):
            result = original_vouch(*args, **kwargs)
            self._check_and_update_ui()
            return result

        def wrapped_unvouch(*args, **kwargs):
            result = original_unvouch(*args, **kwargs)
            self._check_and_update_ui()
            return result

        # Replace methods
        self.contract.deposit = wrapped_deposit
        self.contract.withdraw = wrapped_withdraw
        self.contract.vouch = wrapped_vouch
        self.contract.unvouch = wrapped_unvouch

    def _check_and_update_ui(self):
        """Check if UI update and batch processing are needed."""
        try:
            # Update displays
            self._update_queue_display()
            self._update_current_status()
            self._update_last_batch_status()

            # Check if we need to auto-forge a batch
            if not self.batch_processing and len(self.contract.unprocessed_txns) >= self.contract.batch_size:
                # Schedule batch processing with a delay to allow UI to update
                self._schedule_batch_processing()
        except Exception:
            pass

    def _schedule_batch_processing(self):
        """Schedule batch processing with a small delay to show animation."""
        if self.batch_processing:
            return

        # Mark as processing and show animation immediately
        self.batch_processing = True
        self.current_batch_transactions = list(self.contract.unprocessed_txns)
        self._update_batch_display(processing=True)

        # Schedule actual batch processing after a tiny delay (allows UI to update)
        timer = threading.Timer(0.1, self._forge_batch_delayed)
        timer.daemon = True
        timer.start()

    def _forge_batch_delayed(self):
        """Execute batch forging after delay (runs in background thread)."""
        try:
            # Save current scores as "last batch" before forging
            self.last_batch_scores = self.contract.network.compute_score('pagerank')

            # Process the batch
            result = self.contract.forge_batch()

            # Schedule UI updates on main thread
            def update_ui_after_batch():
                try:
                    if result:
                        self._update_network_graph()
                        self._update_last_batch_status()

                    # Clear batch display
                    self.current_batch_transactions = []
                    self._update_batch_display(processing=False)
                    self._update_queue_display()
                    self._update_current_status()
                except Exception:
                    pass
                finally:
                    self.batch_processing = False

            # Try to schedule on event loop
            try:
                loop = asyncio.get_event_loop()
                loop.call_soon_threadsafe(update_ui_after_batch)
            except:
                # Fallback: just update directly
                update_ui_after_batch()

        except Exception as e:
            try:
                self.batch_display.value = f"<div style='padding: 10px; color: red;'>Batch error: {e}</div>"
            except:
                pass
            self.batch_processing = False

    def _update_queue_display(self):
        """Update the transaction queue display."""
        try:
            txns = self.contract.unprocessed_txns
            if not txns:
                self.queue_display.value = "<div style='padding: 10px; color: #666;'>Queue empty</div>"
            else:
                lines = []
                for i, txn in enumerate(txns, 1):
                    txn_str = format_transaction_display(txn, self.users, index=i)
                    lines.append(f"<div style='padding: 3px; font-family: monospace; font-size: 12px;'>{txn_str}</div>")
                content = "".join(lines)
                self.queue_display.value = f"<div style='padding: 5px; max-height: 200px; overflow-y: auto;'>{content}</div>"
        except Exception as e:
            self.queue_display.value = f"<div style='padding: 10px; color: red;'>Error: {e}</div>"

    def _update_batch_display(self, processing=False):
        """Update the batch forging display."""
        try:
            if processing and self.current_batch_transactions:
                lines = []
                for i, txn in enumerate(self.current_batch_transactions, 1):
                    txn_str = format_transaction_display(txn, self.users, index=i)
                    lines.append(f"<div style='padding: 3px; font-family: monospace; font-size: 12px;'>{txn_str}</div>")
                content = "".join(lines)
                spinner = "<div style='text-align: center; padding: 10px;'>⚡ Processing batch...</div>"
                self.batch_display.value = f"<div style='padding: 5px; max-height: 200px; overflow-y: auto;'>{spinner}{content}</div>"
            else:
                self.batch_display.value = "<div style='padding: 10px; color: #666;'>No batch processing</div>"
        except Exception as e:
            self.batch_display.value = f"<div style='padding: 10px; color: red;'>Error: {e}</div>"

    def _update_current_status(self):
        """Update the current status display."""
        try:
            num_users = len(self.users)
            pending_txns = len(self.contract.unprocessed_txns)
            batch_size = self.contract.batch_size

            # Calculate total deposits
            total_deposits = 0
            for user_data in self.users.values():
                try:
                    balance = user_data['interface'].get_my_balance() / 10**18
                    total_deposits += balance
                except:
                    pass

            status_html = f"""
            <div style='padding: 10px; font-size: 13px;'>
                <div style='padding: 3px;'><strong>👥 Total Users:</strong> {num_users}</div>
                <div style='padding: 3px;'><strong>💰 Total Deposits:</strong> {total_deposits:.4f} ETH</div>
                <div style='padding: 3px;'><strong>⏳ Pending Txns:</strong> {pending_txns} / {batch_size}</div>
                <div style='padding: 3px;'><strong>📊 Algorithm:</strong> {self.contract.scoring_algorithm}</div>
            </div>
            """
            self.current_status_display.value = status_html
        except Exception as e:
            self.current_status_display.value = f"<div style='padding: 10px; color: red;'>Error: {e}</div>"

    def _update_last_batch_status(self):
        """Update the last batch status display."""
        try:
            last_batch = self.contract.last_forged_batch
            if last_batch is None or last_batch == 0:
                self.last_batch_display.value = "<div style='padding: 10px; color: #666;'>No batches forged yet</div>"
            else:
                # Get batch info
                batch_info = self.contract.batches.get(last_batch)
                if batch_info:
                    num_txns = len(batch_info.transactions_processed)
                    self.last_batch_display.value = f"""
                    <div style='padding: 10px; font-size: 13px;'>
                        <div style='padding: 3px;'><strong>📦 Batch #:</strong> {last_batch}</div>
                        <div style='padding: 3px;'><strong>✅ Transactions:</strong> {num_txns}</div>
                        <div style='padding: 3px;'><strong>⏰ Status:</strong> Completed</div>
                    </div>
                    """
                else:
                    self.last_batch_display.value = f"<div style='padding: 10px;'>Batch #{last_batch} completed</div>"
        except Exception:
            # Don't show error, just show basic info
            last_batch = self.contract.last_forged_batch
            if last_batch and last_batch > 0:
                self.last_batch_display.value = f"<div style='padding: 10px;'>Batch #{last_batch} completed</div>"
            else:
                self.last_batch_display.value = "<div style='padding: 10px; color: #666;'>No batches forged yet</div>"


    def _update_network_graph(self):
        """Update both network graph displays (last batch vs current)."""
        try:
            # Update current network graph
            with self.current_graph_output:
                clear_output(wait=True)
                current_scores = self.contract.network.compute_score('pagerank')
                self.contract.network.display_graph(scores=current_scores, title="Current State")

            # Update last batch network graph
            with self.last_batch_graph_output:
                clear_output(wait=True)
                if self.last_batch_scores is not None:
                    self.contract.network.display_graph(scores=self.last_batch_scores, title="Last Batch State")
                else:
                    print("No previous batch data available")
        except Exception:
            pass

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

    def stop(self):
        """Stop the simulator."""
        # Stop simulator
        if self.simulator:
            self.simulator.stop()

        print("✅ Simulator stopped")

    def resume(self):
        """Resume the transaction simulator."""
        print("🎬 Resuming transaction simulator (3s interval)...")

        self.simulator = TransactionSimulator(
            self.contract,
            self.users,
            self.first_user_addr,
            interval=3,
            on_new_user_callback=self._on_new_user_created
        )
        self.simulator.start()
        print("✅ Simulator resumed! Transactions will be generated automatically.")

    def _on_new_user_created(self, new_addr: str, user_name: str):
        """Callback when a new user is created via CreateAccountDeposit."""
        try:
            # Update vouch and unvouch dropdowns
            self.vouch_target.options = self._get_other_users()

            # Display notification in output area
            with self.output_area:
                print(f"👤 New user joined: {user_name} ({new_addr[:12]}...)")
        except Exception as e:
            pass  # Silently ignore errors

    def display(self):
        """Display main contract interface."""
        print(f"🔐 Connected as: {self.first_user['name']} ({self.first_user_addr[:12]}...)")

        # Initialize displays
        self._update_queue_display()
        self._update_current_status()
        self._update_last_batch_status()
        self._update_network_graph()

        # Display the interface
        display(self.contract_interface)

        # Start transaction simulator (it will trigger UI updates via hooks)
        print("🎬 Starting transaction simulator (3s interval)...")
        self.simulator = TransactionSimulator(
            self.contract,
            self.users,
            self.first_user_addr,
            interval=3,
            on_new_user_callback=self._on_new_user_created
        )
        self.simulator.start()
        print("✅ Simulator started! UI updates automatically on each transaction.")
