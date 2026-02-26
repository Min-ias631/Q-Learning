import numpy as np
from numba import float64, int64, types
from numba.experimental import jitclass
from numba.typed import List

order_spec = [
    ('order_id', int64),
    ('side', int64),
    ('price', float64),
    ('quantity', float64),
    ('step_placed', int64),
]

@jitclass(order_spec)
class LimitOrder:
    def __init__(self, order_id, side, price, quantity, step_placed):
        self.order_id = order_id
        self.side = side
        self.price = price
        self.quantity = quantity
        self.step_placed = step_placed

LIMIT_ORDER_TYPE = LimitOrder.class_type.instance_type
PENDING_ORDERS_TYPE = types.ListType(LIMIT_ORDER_TYPE)

spec = [
    ('data', float64[:, :]),
    ('trades', float64[:, :]),
    ('idx', int64),
    ('max_steps', int64),
    ('position', float64),
    ('cash', float64),
    ('entry_price', float64),
    ('last_portfolio_val', float64),
    ('transaction_cost_bps', float64),
    ('max_position', float64),
    ('total_trades', int64),
    ('realized_pnl', float64),
    ('initial_cash', float64),
    ('reward_scaling', float64),
    ('pending_orders', PENDING_ORDERS_TYPE),
    ('order_expire_steps', int64),
    ('next_order_id', int64),
    ('trade_idx', int64),
    ('min_cash_reserve', float64),
]

@jitclass(spec)
class BacktestEngine:
    def __init__(self, depth_data, trade_data, transaction_cost_bps=5.0, max_position=5.0,
                 initial_cash=1.0, reward_scaling=100.0, 
                 order_expire_steps=100):
        """
        Backtesting engine - NOW ONLY HANDLES TRADING LOGIC
        Environment will construct observations
        """
        self.data = depth_data
        self.trades = trade_data
        self.max_steps = len(depth_data)
        self.transaction_cost_bps = transaction_cost_bps
        self.max_position = max_position
        self.initial_cash = initial_cash
        self.reward_scaling = reward_scaling
        self.order_expire_steps = order_expire_steps
        self.pending_orders = List.empty_list(LIMIT_ORDER_TYPE)
        
        # Keep at least 10% of initial cash in reserve
        self.min_cash_reserve = initial_cash * 0.1
        
        self.reset_state()
    
    def reset_state(self, start_idx=-1):
        """
        Reset engine state with optional random starting position
        
        Args:
            start_idx: Starting index (-1 for random, >=0 for specific position)
        """
        self.idx = max(0, min(start_idx, len(self.data) - 1))
        
        # Sync trade_idx to match data idx timestamp
        self.trade_idx = 0
        if self.idx > 0:
            current_ts = self.data[self.idx, 0]
            while (self.trade_idx < len(self.trades) and 
                   self.trades[self.trade_idx, 0] < current_ts):
                self.trade_idx += 1
        
        self.position = 0.0
        self.cash = self.initial_cash
        self.entry_price = 0.0
        self.last_portfolio_val = self.initial_cash
        self.total_trades = 0
        self.realized_pnl = 0.0
        self.next_order_id = 0
        self.pending_orders.clear()
    
    def get_total_pnl(self, current_price):
        """
        Calculate TOTAL PnL including unrealized gains/losses
        Total PnL = (Portfolio Value - Initial Cash)
        Portfolio Value = Cash + (Position × Current Price)
        """
        portfolio_value = self.cash + (self.position * current_price)
        return portfolio_value - self.initial_cash
    
    def calculate_trade_qty(self, price, side):
        """
        Calculate trade quantity with proper cash constraints
        """
        # Target: Each trade uses ~2% of initial capital
        target_value = self.initial_cash * 0.02
        qty = target_value / price
        
        # Check position limits
        if side == 1:  # Buy
            max_qty_position = self.max_position - self.position
        else:  # Sell
            max_qty_position = self.max_position + self.position
        
        qty = min(qty, max_qty_position)
        
        # Check cash constraints for buys
        if side == 1:
            total_cost = qty * price * (1.0 + self.transaction_cost_bps / 10_000.0)
            available_cash = self.cash - self.min_cash_reserve
            
            if total_cost > available_cash:
                if available_cash > 0:
                    qty = available_cash / (price * (1.0 + self.transaction_cost_bps / 10_000.0))
                else:
                    qty = 0.0
        
        if qty < 1e-9:
            qty = 0.0
            
        return qty

    def _check_limit_fills(self, start_ts, end_ts):
        actions = np.zeros(len(self.pending_orders), dtype=np.int64)

        min_taker_sell_px = 1e9
        max_taker_buy_px = -1.0
        found_sell = False
        found_buy = False

        while self.trade_idx < len(self.trades) and self.trades[self.trade_idx, 0] < start_ts:
            self.trade_idx += 1
        
        while self.trade_idx < len(self.trades):
            row = self.trades[self.trade_idx]
            if row[0] >= end_ts: 
                break

            px = row[1]
            is_buyer_maker = row[4]

            if is_buyer_maker == 0.0 and px > max_taker_buy_px:
                max_taker_buy_px = px
                found_buy = True
            elif is_buyer_maker == 1.0 and px < min_taker_sell_px:
                min_taker_sell_px = px
                found_sell = True
            self.trade_idx += 1

        row = self.data[self.idx]
        current_bid = row[1]
        current_ask = row[11]

        for i in range(len(self.pending_orders)):
            order = self.pending_orders[i]
            
            if self.idx - order.step_placed >= self.order_expire_steps:
                actions[i] = 2
                continue
            
            is_filled = False

            if order.side == 1:
                if current_ask <= order.price:
                    is_filled = True
                elif found_sell and min_taker_sell_px <= order.price:
                    is_filled = True
            elif order.side == 2:
                if current_bid >= order.price:
                    is_filled = True
                elif found_buy and max_taker_buy_px >= order.price:
                    is_filled = True
            
            if is_filled:
                actions[i] = 1
        return actions
    
    def _get_available_quantity(self, side, price, row_idx):
        row = self.data[row_idx]

        if side == 1:
            for i in range(5):
                bid_px = row[1 + i * 2]
                bid_qty = row[2 + i * 2]
                if abs(bid_px - price) < 1e-9:
                    return bid_qty
            return 0.0
        elif side == 2:
            for i in range(5):
                ask_px = row[11 + i * 2]
                ask_qty = row[12 + i * 2]
                if abs(ask_px - price) < 1e-9:
                    return ask_qty
            return 0.0
        return 0.0
    
    def _execute_limit_order(self, order, max_fill_qty):
        """
        Execute limit order with proper cash constraints
        Returns: (status, filled_qty)
          status: 0=Cancelled, 1=Filled, 2=Partially Filled
        """
        default_return = 1
        filled_qty = 0.0

        quantity = min(order.quantity, max_fill_qty)
        if quantity <= 1e-9:
            return 0, 0.0

        if order.side == 1:  # BUY
            if self.position >= self.max_position:
                return 0, 0.0
            
            # Check if we have enough cash
            total_cost = quantity * order.price * (1.0 + self.transaction_cost_bps / 10_000.0)
            available_cash = self.cash - self.min_cash_reserve
            
            if total_cost > available_cash:
                if available_cash > 0:
                    quantity = available_cash / (order.price * (1.0 + self.transaction_cost_bps / 10_000.0))
                    default_return = 2
                else:
                    return default_return, filled_qty
            
            if quantity <= 1e-9:
                return default_return, filled_qty
            
            trade_cost = quantity * order.price * (self.transaction_cost_bps / 10_000.0)
            self.cash -= quantity * order.price + trade_cost

            # Update entry price with guard against division by zero
            new_position = self.position + quantity
            if new_position > 1e-9:
                self.entry_price = (self.position * self.entry_price + quantity * order.price) / new_position
            else:
                self.entry_price = order.price
            
            self.position += quantity
            self.realized_pnl -= trade_cost
            self.total_trades += 1

            filled_qty += quantity

            if filled_qty < order.quantity - 1e-9:
                return 2, filled_qty
            return default_return, filled_qty

        elif order.side == 2:  # SELL
            if self.position <= -self.max_position:
                return 0, 0.0
            
            # Close long position first
            if self.position > 0:
                closing_qty = min(self.position, quantity)
                close_long_cost = closing_qty * order.price * (self.transaction_cost_bps / 10_000.0)
                self.realized_pnl += closing_qty * (order.price - self.entry_price) - close_long_cost
                self.cash -= close_long_cost
                self.cash += closing_qty * order.price

                filled_qty += closing_qty

                if quantity <= self.position:
                    self.position -= closing_qty
                    if filled_qty < order.quantity - 1e-9:
                        return 2, filled_qty
                    return default_return, filled_qty
                
                self.entry_price = 0.0
                self.position -= closing_qty
                quantity -= closing_qty

            # Check position limit
            if self.position - quantity < -self.max_position:
                quantity = self.max_position + self.position
                default_return = 2

            if quantity <= 1e-9:
                return default_return, filled_qty
            
            trade_cost = quantity * order.price * (self.transaction_cost_bps / 10_000.0)
            self.cash -= trade_cost
            self.cash += quantity * order.price

            # Update entry price with guard against division by zero
            new_position_abs = abs(self.position) + quantity
            if new_position_abs > 1e-9:
                self.entry_price = (abs(self.position) * self.entry_price + quantity * order.price) / new_position_abs
            else:
                self.entry_price = order.price
            
            self.position -= quantity
            self.realized_pnl -= trade_cost
            self.total_trades += 1

            filled_qty += quantity
            if filled_qty < order.quantity - 1e-9:
                return 2, filled_qty
            return default_return, filled_qty
            
        return 0, 0.0
    
    def step(self, action):
        """
        Execute one step of trading
        
        Returns:
            reward: float - scaled portfolio value change
            done: bool - episode termination
            current_mid: float - current mid price
            next_mid: float - next mid price (for observation construction)
        """
        row = self.data[self.idx]

        current_ts = row[0]
        current_bid = row[1]
        current_ask = row[11]
        current_mid = (current_bid + current_ask) * 0.5

        if self.idx < self.max_steps - 1:
            next_ts = self.data[self.idx + 1, 0]
        else:
            if self.idx == 0:
                raise ValueError('Input file has one line of data')
            
            prev_ts = self.data[self.idx - 1, 0]
            next_ts = current_ts + (current_ts - prev_ts)
        
        # Check and fill limit orders
        actions = self._check_limit_fills(current_ts, next_ts)

        for i in range(len(actions) - 1, -1, -1):
            act = actions[i]
            if act == 1:  # FILL
                order = self.pending_orders[i]
                max_fill_qty = self._get_available_quantity(order.side, order.price, self.idx)
                exec_status, filled_qty = self._execute_limit_order(order, max_fill_qty)
                if exec_status == 1:
                    self.pending_orders.pop(i)
                elif exec_status == 2:
                    order.quantity -= filled_qty
                    if order.quantity <= 1e-9:
                        self.pending_orders.pop(i)  

            elif act == 2:  # EXPIRE
                self.pending_orders.pop(i)

        # Execute new action
        if action == 1:  # Market Buy
            if self.position < self.max_position:
                available_qty = row[12]
                adaptive_qty = self.calculate_trade_qty(current_ask, 1)
                qty = min(adaptive_qty, available_qty)
                if qty > 1e-9:
                    order = LimitOrder(-1, 1, current_ask, qty, self.idx)
                    self._execute_limit_order(order, qty)

        elif action == 2:  # Market Sell
            if self.position > -self.max_position:
                available_qty = row[2]
                adaptive_qty = self.calculate_trade_qty(current_bid, 2)
                qty = min(adaptive_qty, available_qty)
                if qty > 1e-9:
                    order = LimitOrder(-1, 2, current_bid, qty, self.idx)
                    self._execute_limit_order(order, qty)
        
        elif 3 <= action <= 7:  # Limit Buy at depth level
            if self.position < self.max_position:
                level = action - 3
                price = row[1 + level * 2]
                qty = self.calculate_trade_qty(price, 1)
                if qty > 1e-9:
                    order = LimitOrder(self.next_order_id, 1, price, qty, self.idx)
                    self.pending_orders.append(order)
                    self.next_order_id += 1
        
        elif 8 <= action <= 12:  # Limit Sell at depth level
            if self.position > -self.max_position:
                level = action - 8
                price = row[11 + level * 2]
                qty = self.calculate_trade_qty(price, 2)
                if qty > 1e-9:
                    order = LimitOrder(self.next_order_id, 2, price, qty, self.idx)
                    self.pending_orders.append(order)
                    self.next_order_id += 1
        
        elif action == 13:  # Limit Both (at best bid/ask)
            if self.position < self.max_position:
                qty = self.calculate_trade_qty(current_bid, 1)
                if qty > 1e-9:
                    order = LimitOrder(self.next_order_id, 1, current_bid, qty, self.idx)
                    self.pending_orders.append(order)
                    self.next_order_id += 1
            if self.position > -self.max_position:
                qty = self.calculate_trade_qty(current_ask, 2)
                if qty > 1e-9:
                    order = LimitOrder(self.next_order_id, 2, current_ask, qty, self.idx)
                    self.pending_orders.append(order)
                    self.next_order_id += 1
        
        elif action == 14:  # Cancel All
            self.pending_orders.clear()

        self.idx += 1
        done = self.idx >= self.max_steps
            
        if not done:
            next_row = self.data[self.idx]
            next_mid = (next_row[1] + next_row[11]) * 0.5
        else:
            next_mid = current_mid
            
            # Close out final position
            if self.position != 0:
                cost = abs(self.position) * next_mid * (self.transaction_cost_bps / 10_000.0)
                
                if self.position > 0:
                    self.realized_pnl += (next_mid - self.entry_price) * self.position - cost
                else:
                    self.realized_pnl += (self.entry_price - next_mid) * abs(self.position) - cost
                
                self.cash += (self.position * next_mid) - cost
                self.position = 0.0
                self.pending_orders.clear()
        
        # Calculate portfolio value for reward
        post_portfolio_val = self.cash + (self.position * next_mid)
        
        if self.idx == 1:
            prev_val = self.initial_cash
        else:
            prev_val = self.last_portfolio_val
        
        # Reward based on PORTFOLIO VALUE CHANGE (includes unrealized PnL)    
        reward = self.reward_scaling * (post_portfolio_val - prev_val) / self.initial_cash
        self.last_portfolio_val = post_portfolio_val
        
        # Return reward, done flag, and price info for observation construction
        return reward, done, current_mid, next_mid