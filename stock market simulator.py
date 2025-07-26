import sys
import numpy as np
import collections
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QLabel, QRadioButton, QGroupBox, QFormLayout,
    QSlider
)
from PyQt6.QtCore import QTimer, Qt
import pyqtgraph as pg

# --- Configuration ---
INITIAL_PRICE = 100.0
SMA_PERIOD = 20
SCROLLING_VIEW_TICKS = 100
MINIMUM_PRICE = 1e-6
INT32_MAX = 2147483647
MARKET_SHOCK_CHANCE = 0.005
MARKET_SHOCK_MAGNITUDE = 0.05

PLAYER_INITIAL_CASH = 50000

RETAIL_CAPITAL_MEAN, RETAIL_CAPITAL_STD = 10000, 2000
DAY_TRADER_CAPITAL_MEAN, DAY_TRADER_CAPITAL_STD = 50000, 10000
INSTITUTIONAL_CAPITAL_MEAN, INSTITUTIONAL_CAPITAL_STD = 10000000, 2000000

BEHAVIOR_PROFILES = {
    'retail': {'order_chance': 0.1, 'market_order_pct': 0.7, 'limit_order_ttl': 50},
    'day_trader': {'order_chance': 0.8, 'market_order_pct': 0.5, 'limit_order_ttl': 100},
    'institutional': {'order_chance': 0.25, 'market_order_pct': 0.2, 'limit_order_ttl': 500}
}

TRADER_PERSONALITIES = ['value_investor', 'momentum_trader', 'mean_reversion', 'noise_trader']

# --- Market Simulation Engine ---
class Market:
    def __init__(self):
        self.reset(initial_shares=100000)

    @property
    def total_shares(self):
        if self.num_traders > 0:
            return self.shares.sum() + self.player_shares
        return self.player_shares

    def _initialize_traders(self, trader_counts):
        self.num_traders = sum(trader_counts.values())
        if self.num_traders == 0:
            self.trader_ids = np.array([], dtype=int)
            self.cash = np.array([], dtype=np.float64)
            self.shares = np.array([], dtype=np.int64)
            self.order_chance = np.array([])
            self.market_order_pct = np.array([])
            self.limit_order_ttl = np.array([], dtype=int)
            self.personality = np.array([], dtype=object)
            return

        self.trader_ids = np.arange(self.num_traders)
        self.cash = np.zeros(self.num_traders, dtype=np.float64)
        self.shares = np.zeros(self.num_traders, dtype=np.int64)
        self.order_chance = np.zeros(self.num_traders)
        self.market_order_pct = np.zeros(self.num_traders)
        self.limit_order_ttl = np.zeros(self.num_traders, dtype=int)
        self.personality = np.random.choice(TRADER_PERSONALITIES, self.num_traders, p=[0.25, 0.25, 0.25, 0.25])

        start_idx = 0
        for trader_type, count in trader_counts.items():
            if count == 0: continue
            end_idx = start_idx + count
            mask = slice(start_idx, end_idx)
            
            if trader_type == 'retail': mean_cap, std_cap = RETAIL_CAPITAL_MEAN, RETAIL_CAPITAL_STD
            elif trader_type == 'day_trader': mean_cap, std_cap = DAY_TRADER_CAPITAL_MEAN, DAY_TRADER_CAPITAL_STD
            else: mean_cap, std_cap = INSTITUTIONAL_CAPITAL_MEAN, INSTITUTIONAL_CAPITAL_STD
            
            self.cash[mask] = np.random.normal(mean_cap, std_cap, count)
            self.shares[mask] = 0

            profile = BEHAVIOR_PROFILES[trader_type]
            self.order_chance[mask] = profile['order_chance']
            self.market_order_pct[mask] = profile['market_order_pct']
            self.limit_order_ttl[mask] = profile['limit_order_ttl']

    def reset(self, initial_shares):
        print("--- SIMULATION RESET ---")
        self.time = 0
        self.current_price = INITIAL_PRICE
        self.fundamental_value = INITIAL_PRICE
        self.graph_view_mode = 'Full'
        self.trader_counts = {'retail': 0, 'day_trader': 0, 'institutional': 0}
        
        self.initial_share_supply = initial_shares
        self.shares_seeded = False

        self._initialize_traders(self.trader_counts)

        self.price_history = collections.deque(maxlen=100000)
        self.volume_history = collections.deque(maxlen=100000)
        self.sma_history = collections.deque(maxlen=100000)
        
        self.price_history.append(INITIAL_PRICE)
        self.volume_history.append(0)
        self.sma_history.append(INITIAL_PRICE)
        self.sma_queue = collections.deque([INITIAL_PRICE] * SMA_PERIOD, maxlen=SMA_PERIOD)

        self.bids = []
        self.asks = []

        self.player_cash = PLAYER_INITIAL_CASH
        self.player_shares = 0
        print(f"Initial State: Price=${self.current_price}, Supply={initial_shares}")

    def add_traders(self, count, trader_type):
        if count <= 0: return
        print(f"Adding {count} '{trader_type}' traders...")
        
        if trader_type == 'retail': mean_cap, std_cap = RETAIL_CAPITAL_MEAN, RETAIL_CAPITAL_STD
        elif trader_type == 'day_trader': mean_cap, std_cap = DAY_TRADER_CAPITAL_MEAN, DAY_TRADER_CAPITAL_STD
        else: mean_cap, std_cap = INSTITUTIONAL_CAPITAL_MEAN, INSTITUTIONAL_CAPITAL_STD
        
        new_cash = np.random.normal(mean_cap, std_cap, count)
        new_shares = np.zeros(count, dtype=np.int64)
        
        if not self.shares_seeded and self.initial_share_supply > 0:
            print(f"Seeding {self.initial_share_supply} shares to new traders...")
            total_cash = new_cash.sum()
            if total_cash > 0:
                share_dist = (new_cash / total_cash * self.initial_share_supply).astype(int)
                if len(share_dist) > 0:
                    share_dist[0] += self.initial_share_supply - share_dist.sum()
                new_shares = share_dist
            self.shares_seeded = True
        
        profile = BEHAVIOR_PROFILES[trader_type]
        new_order_chance = np.full(count, profile['order_chance'])
        new_market_pct = np.full(count, profile['market_order_pct'])
        new_ttl = np.full(count, profile['limit_order_ttl'])
        new_personalities = np.random.choice(TRADER_PERSONALITIES, count)

        self.cash = np.concatenate([self.cash, new_cash]) if self.num_traders > 0 else new_cash
        self.shares = np.concatenate([self.shares, new_shares]) if self.num_traders > 0 else new_shares
        self.order_chance = np.concatenate([self.order_chance, new_order_chance]) if self.num_traders > 0 else new_order_chance
        self.market_order_pct = np.concatenate([self.market_order_pct, new_market_pct]) if self.num_traders > 0 else new_market_pct
        self.limit_order_ttl = np.concatenate([self.limit_order_ttl, new_ttl]) if self.num_traders > 0 else new_ttl
        self.personality = np.concatenate([self.personality, new_personalities]) if self.num_traders > 0 else new_personalities
        
        self.trader_counts[trader_type] += count
        self.num_traders += count
        self.trader_ids = np.arange(self.num_traders)
        print(f"Total traders now: {self.num_traders}")

    def perform_split(self, factor=2):
        print(f"--- STOCK SPLIT ({factor}:1) ---")
        self.shares = (self.shares * factor).astype(np.int64)
        self.player_shares = int(self.player_shares * factor)
        self.current_price /= factor
        self.fundamental_value /= factor
        
        for order in self.bids + self.asks:
            order['qty'] = int(order['qty'] * factor)
            if order['price'] is not None: order['price'] /= factor

        self.price_history = collections.deque([p / factor for p in self.price_history], maxlen=self.price_history.maxlen)
        self.sma_history = collections.deque([s / factor for s in self.sma_history], maxlen=self.sma_history.maxlen)
        self.sma_queue = collections.deque([q / factor for q in self.sma_queue], maxlen=SMA_PERIOD)

    def perform_reverse_split(self, factor=2):
        print(f"--- REVERSE SPLIT (1:{factor}) ---")
        self.shares = (self.shares // factor).astype(np.int64)
        self.player_shares = int(self.player_shares // factor)
        self.current_price *= factor
        self.fundamental_value *= factor

        for order in self.bids + self.asks:
            order['qty'] = int(order['qty'] // factor)
            if order['price'] is not None: order['price'] *= factor
        
        self.bids = [o for o in self.bids if o['qty'] > 0]
        self.asks = [o for o in self.asks if o['qty'] > 0]

        self.price_history = collections.deque([p * factor for p in self.price_history], maxlen=self.price_history.maxlen)
        self.sma_history = collections.deque([s * factor for s in self.sma_history], maxlen=self.sma_history.maxlen)
        self.sma_queue = collections.deque([q * factor for q in self.sma_queue], maxlen=SMA_PERIOD)

    def place_player_order(self, side, qty, price=None):
        if qty <= 0: return
        order = {'id': 'player', 'qty': qty, 'price': price}
        if side == 'buy':
            required_cash = qty * (price if price is not None else self.current_price * 1.05)
            if self.player_cash >= required_cash: self.bids.append(order)
            else: print("Player has insufficient cash for this buy order.")
        elif side == 'sell':
            if self.player_shares >= qty: self.asks.append(order)
            else: print("Player has insufficient shares for this sell order.")

    def _execute_trade(self, buyer_id, seller_id, qty, price):
        cost = qty * price
        if buyer_id == 'player': self.player_cash -= cost; self.player_shares += qty
        else: self.cash[buyer_id] -= cost; self.shares[buyer_id] += qty
        if seller_id == 'player': self.player_cash += cost; self.player_shares -= qty
        else: self.cash[seller_id] += cost; self.shares[seller_id] -= qty
    
    def _generate_trader_orders(self):
        if self.num_traders == 0: return
        
        sma = self.sma_history[-1]
        momentum_sentiment = (self.current_price - sma) / (sma + 1e-9)
        value_sentiment = (self.fundamental_value - self.current_price) / (self.current_price + 1e-9)
        reversion_sentiment = (sma - self.current_price) / (self.current_price + 1e-9)

        rand_vals = np.random.rand(2, self.num_traders)
        active_mask = rand_vals[0] < self.order_chance
        if not np.any(active_mask): return

        active_ids = self.trader_ids[active_mask]
        active_personalities = self.personality[active_mask]
        
        buy_prob = np.full(len(active_ids), 0.5)

        buy_prob[active_personalities == 'value_investor'] += value_sentiment * 0.5
        buy_prob[active_personalities == 'momentum_trader'] += momentum_sentiment * 0.5
        buy_prob[active_personalities == 'mean_reversion'] += reversion_sentiment * 0.5
        
        buy_prob = np.clip(buy_prob, 0.01, 0.99)

        is_buy_decision = rand_vals[1, active_mask] < buy_prob
        
        for i, trader_id in enumerate(active_ids):
            is_buy = is_buy_decision[i]
            
            capital = self.cash[trader_id] + self.shares[trader_id] * self.current_price
            order_size_pct = np.random.uniform(0.01, 0.05)
            order_capital = capital * order_size_pct
            quantity = int(order_capital / (self.current_price + 1e-9))
            if quantity == 0: continue

            is_market = np.random.rand() < self.market_order_pct[trader_id]
            price = None
            if not is_market:
                spread = (self.current_price * 1.01) - (self.current_price * 0.99)
                if is_buy: price = self.current_price * 0.99 + np.random.uniform(0, spread/2)
                else: price = self.current_price * 1.01 - np.random.uniform(0, spread/2)
                price = round(price, 2)

            order = {'id': trader_id, 'qty': quantity, 'price': price}
            if price is not None:
                order['expires_at'] = self.time + self.limit_order_ttl[trader_id]

            if is_buy:
                if self.cash[trader_id] >= order_capital: self.bids.append(order)
            else:
                if self.shares[trader_id] >= quantity: self.asks.append(order)

    def _expire_orders(self):
        self.bids = [o for o in self.bids if 'expires_at' not in o or o['expires_at'] > self.time]
        self.asks = [o for o in self.asks if 'expires_at' not in o or o['expires_at'] > self.time]

    def _match_orders(self):
        volume_this_tick = 0
        market_bids = [o for o in self.bids if o['price'] is None]
        limit_bids = sorted([o for o in self.bids if o['price'] is not None], key=lambda x: x['price'], reverse=True)
        market_asks = [o for o in self.asks if o['price'] is None]
        limit_asks = sorted([o for o in self.asks if o['price'] is not None], key=lambda x: x['price'])

        while limit_bids and limit_asks and limit_bids[0]['price'] >= limit_asks[0]['price']:
            bid, ask = limit_bids[0], limit_asks[0]
            trade_price = (bid['price'] + ask['price']) / 2.0
            trade_qty = min(bid['qty'], ask['qty'])
            self._execute_trade(bid['id'], ask['id'], trade_qty, trade_price)
            volume_this_tick += trade_qty
            self.current_price = max(trade_price, MINIMUM_PRICE)
            bid['qty'] -= trade_qty; ask['qty'] -= trade_qty
            if bid['qty'] == 0: limit_bids.pop(0)
            if ask['qty'] == 0: limit_asks.pop(0)

        for bid in market_bids:
            while bid['qty'] > 0 and limit_asks:
                ask = limit_asks[0]; trade_price = ask['price']
                trade_qty = min(bid['qty'], ask['qty'])
                self._execute_trade(bid['id'], ask['id'], trade_qty, trade_price)
                volume_this_tick += trade_qty
                self.current_price = max(trade_price, MINIMUM_PRICE)
                bid['qty'] -= trade_qty; ask['qty'] -= trade_qty
                if ask['qty'] == 0: limit_asks.pop(0)
        
        for ask in market_asks:
            while ask['qty'] > 0 and limit_bids:
                bid = limit_bids[0]; trade_price = bid['price']
                trade_qty = min(ask['qty'], bid['qty'])
                self._execute_trade(bid['id'], ask['id'], trade_qty, trade_price)
                volume_this_tick += trade_qty
                self.current_price = max(trade_price, MINIMUM_PRICE)
                ask['qty'] -= trade_qty; bid['qty'] -= trade_qty
                if bid['qty'] == 0: limit_bids.pop(0)

        self.bids = [b for b in market_bids if b['qty'] > 0] + limit_bids
        self.asks = [a for a in market_asks if a['qty'] > 0] + limit_asks
        
        bid_volume = sum(o['qty'] for o in self.bids)
        ask_volume = sum(o['qty'] for o in self.asks)
        imbalance = bid_volume - ask_volume
        price_nudge = (imbalance / (bid_volume + ask_volume + 1)) * self.current_price * 0.001
        self.current_price = max(self.current_price + price_nudge, MINIMUM_PRICE)

        return volume_this_tick
    
    def _simulate_market_events(self):
        if np.random.rand() < MARKET_SHOCK_CHANCE:
            shock = np.random.normal(0, MARKET_SHOCK_MAGNITUDE)
            self.fundamental_value *= (1 + shock)
            print(f"--- MARKET SHOCK: Fundamental value changed by {shock:.2%} ---")

    def step(self):
        self.time += 1
        self.fundamental_value *= (1 + np.random.normal(0, 0.0005))
        self._expire_orders()
        self._simulate_market_events()
        self._generate_trader_orders()
        volume = self._match_orders()
        self.price_history.append(self.current_price)
        self.volume_history.append(volume)
        self.sma_queue.append(self.current_price)
        self.sma_history.append(sum(self.sma_queue) / len(self.sma_queue))

# --- PyQt6 UI Application ---
class SimulatorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("High-Performance Stock Simulator")
        self.setGeometry(100, 100, 1200, 800)

        self.market = Market()
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.price_plot = pg.PlotWidget()
        self.volume_plot = pg.PlotWidget()
        main_layout.addWidget(self.price_plot)
        main_layout.addWidget(self.volume_plot, stretch=1)
        main_layout.setStretchFactor(self.price_plot, 3)

        self.price_plot.setLabel('left', 'Price', units='$')
        self.price_plot.addLegend()
        self.volume_plot.setLabel('left', 'Volume')
        self.volume_plot.setLabel('bottom', 'Time (ticks)')
        self.volume_plot.setXLink(self.price_plot)

        self.price_curve = self.price_plot.plot(pen='c', name='Price')
        self.sma_curve = self.price_plot.plot(pen=pg.mkPen('y', style=Qt.PenStyle.DashLine), name=f'{SMA_PERIOD}-tick SMA')
        self.volume_bars = pg.BarGraphItem(x=[], height=[], width=0.8, brush='w')
        self.volume_plot.addItem(self.volume_bars)

        controls_layout = QHBoxLayout()
        main_layout.addLayout(controls_layout)

        trader_group = QGroupBox("Add Traders")
        player_group = QGroupBox("Player Controls")
        info_group = QGroupBox("Market Controls")
        time_group = QGroupBox("Time Controls")
        view_group = QGroupBox("Graph View")

        trader_layout = QFormLayout(trader_group)
        self.trader_type_radios = {
            'retail': QRadioButton("Retail"),
            'day_trader': QRadioButton("Day Trader"),
            'institutional': QRadioButton("Institutional")
        }
        self.trader_type_radios['retail'].setChecked(True)
        for radio in self.trader_type_radios.values():
            trader_layout.addRow(radio)
        self.trader_count_input = QLineEdit("100")
        trader_layout.addRow("Amount:", self.trader_count_input)
        add_trader_btn = QPushButton("Add")
        add_trader_btn.clicked.connect(self.on_add_traders)
        trader_layout.addRow(add_trader_btn)

        player_layout = QFormLayout(player_group)
        self.qty_input = QLineEdit("10")
        self.price_input = QLineEdit()
        self.price_input.setPlaceholderText("Market")
        self.price_input.setEnabled(True)
        buy_btn = QPushButton("Buy")
        sell_btn = QPushButton("Sell")
        buy_btn.clicked.connect(self.on_buy)
        sell_btn.clicked.connect(self.on_sell)
        player_layout.addRow("Qty:", self.qty_input)
        player_layout.addRow("Price:", self.price_input)
        player_layout.addRow(buy_btn, sell_btn)
        self.player_status_label = QLabel("...")
        player_layout.addRow(self.player_status_label)

        info_layout = QFormLayout(info_group)
        self.initial_shares_input = QLineEdit("100000")
        reset_btn = QPushButton("Set & Reset")
        reset_btn.clicked.connect(self.on_reset)
        split_btn = QPushButton("Split 2:1")
        rev_split_btn = QPushButton("R-Split 1:2")
        split_btn.clicked.connect(lambda: self.market.perform_split(2))
        rev_split_btn.clicked.connect(lambda: self.market.perform_reverse_split(2))
        self.trader_counts_label = QLabel("...")
        self.market_cap_label = QLabel("...")
        self.order_book_label = QLabel("...")
        info_layout.addRow("Initial Shares:", self.initial_shares_input)
        info_layout.addRow(reset_btn)
        info_layout.addRow(split_btn, rev_split_btn)
        info_layout.addRow(self.trader_counts_label)
        info_layout.addRow(self.market_cap_label)
        info_layout.addRow(self.order_book_label)
        
        time_layout = QFormLayout(time_group)
        self.pause_play_btn = QPushButton("Pause")
        self.pause_play_btn.clicked.connect(self.toggle_pause)
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(1, 500)
        self.speed_slider.setValue(50)
        self.speed_slider.setInvertedAppearance(True)
        self.speed_label = QLabel(f"Speed: {self.speed_slider.value()}ms")
        self.speed_slider.valueChanged.connect(self.change_tick_speed)
        time_layout.addRow(self.pause_play_btn)
        time_layout.addRow("Tick Speed:", self.speed_slider)
        time_layout.addRow(self.speed_label)

        view_layout = QVBoxLayout(view_group)
        self.view_full_radio = QRadioButton("Full")
        self.view_scroll_radio = QRadioButton("Scrolling")
        self.view_full_radio.setChecked(True)
        self.view_full_radio.toggled.connect(lambda: self.on_view_change('Full'))
        self.view_scroll_radio.toggled.connect(lambda: self.on_view_change('Scrolling'))
        view_layout.addWidget(self.view_full_radio)
        view_layout.addWidget(self.view_scroll_radio)

        controls_layout.addWidget(trader_group)
        controls_layout.addWidget(player_group)
        controls_layout.addWidget(info_group)
        controls_layout.addWidget(time_group)
        controls_layout.addWidget(view_group)

        self.timer = QTimer()
        self.timer.setInterval(self.speed_slider.value())
        self.timer.timeout.connect(self.update)
        self.timer.start()

    def on_add_traders(self):
        try:
            count = int(self.trader_count_input.text())
            ttype = [k for k, v in self.trader_type_radios.items() if v.isChecked()][0]
            self.market.add_traders(count, ttype)
        except ValueError:
            print("Invalid trader count")

    def on_buy(self):
        try:
            qty = int(self.qty_input.text())
            price = float(self.price_input.text()) if self.price_input.text() else None
            self.market.place_player_order('buy', qty, price)
        except ValueError:
            print("Invalid quantity or price")

    def on_sell(self):
        try:
            qty = int(self.qty_input.text())
            price = float(self.price_input.text()) if self.price_input.text() else None
            self.market.place_player_order('sell', qty, price)
        except ValueError:
            print("Invalid quantity or price")
            
    def on_reset(self):
        try:
            initial_shares = int(self.initial_shares_input.text())
            self.market.reset(initial_shares)
            self.price_plot.setXRange(0, 100)
            self.price_plot.setYRange(INITIAL_PRICE * 0.9, INITIAL_PRICE * 1.1)
        except ValueError:
            print("Invalid initial shares value.")

    def on_view_change(self, mode):
        self.market.graph_view_mode = mode

    def toggle_pause(self):
        if self.timer.isActive():
            self.timer.stop()
            self.pause_play_btn.setText("Play")
        else:
            self.timer.start()
            self.pause_play_btn.setText("Pause")

    def change_tick_speed(self):
        speed = self.speed_slider.value()
        self.timer.setInterval(speed)
        self.speed_label.setText(f"Speed: {speed}ms")

    def update(self):
        self.market.step()
        x_data = np.arange(self.market.time + 1)
        self.price_curve.setData(x_data, list(self.market.price_history))
        self.sma_curve.setData(x_data, list(self.market.sma_history))
        self.volume_bars.setOpts(x=x_data, height=list(self.market.volume_history))

        if self.market.graph_view_mode == 'Scrolling':
            self.price_plot.setXRange(max(0, self.market.time - SCROLLING_VIEW_TICKS), self.market.time + 5)
        
        player_val = self.market.player_cash + self.market.player_shares * self.market.current_price
        self.player_status_label.setText(f"Cash: ${self.market.player_cash:,.2f}\nShares: {self.market.player_shares}\nValue: ${player_val:,.2f}")
        
        # FIX: Access trader_counts from the market object
        counts_str = "Trader Counts:\n" + "\n".join([f"  {k}: {v}" for k,v in self.market.trader_counts.items()])
        self.trader_counts_label.setText(counts_str)
        
        market_cap = self.market.total_shares * self.market.current_price
        self.market_cap_label.setText(f"Shares Out: {self.market.total_shares:,}\nMarket Cap: ${market_cap:,.2f}")
        
        self.order_book_label.setText(f"Order Book:\n  Bids: {len(self.market.bids)}\n  Asks: {len(self.market.asks)}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SimulatorWindow()
    window.show()
    sys.exit(app.exec())
