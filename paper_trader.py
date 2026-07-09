import os
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus

class PaperTrader:
    def __init__(self, api_key: str, secret_key: str):
        self.api_key = api_key
        self.secret_key = secret_key
        
        # Initialize the trading client securely pointing to the paper environment
        self.trading_client = TradingClient(
            api_key=self.api_key, 
            secret_key=self.secret_key, 
            paper=True
        )

    def execute_trade(self, symbol: str, signal: int, active_models: int = 1):
        """
        Executes a paper trade based on the given signal.
        signal == 1: Long (Market Buy)
        signal == 0: Flat (Liquidate position)
        active_models: The number of active ticker models deployed in the portfolio.
        """
        try:
            # Check current position
            positions = self.trading_client.get_all_positions()
            current_position = next((p for p in positions if p.symbol == symbol), None)
            
            if signal == 1:
                # We want to be Long
                if current_position is None:
                    # Check for pending open buy orders
                    open_orders = self.trading_client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol]))
                    if open_orders:
                        return f"SKIPPED: Pending buy order already exists for {symbol}."
                        
                    # Not in position, so we buy
                    account = self.trading_client.get_account()
                    
                    # Calculate exactly 1% of total equity
                    account_equity = float(account.equity)
                    trade_amount = account_equity * 0.01

                    if trade_amount >= 10:  # Minimum threshold for a trade
                        try:
                            # Try notional (fractional) buying first
                            market_order_data = MarketOrderRequest(
                                symbol=symbol,
                                notional=round(trade_amount, 2), # Buy fractional shares based on dollar amount
                                side=OrderSide.BUY,
                                time_in_force=TimeInForce.DAY
                            )
                            order = self.trading_client.submit_order(order_data=market_order_data)
                            return f"SUCCESS: Submitted BUY order for ${trade_amount:.2f} of {symbol}. Order ID: {order.id}"
                        except Exception as notional_error:
                            # Fallback to integer qty buying if notional fails
                            try:
                                from alpaca.data.historical import StockHistoricalDataClient
                                from alpaca.data.requests import StockLatestTradeRequest
                                data_client = StockHistoricalDataClient(self.api_key, self.secret_key)
                                trade = data_client.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=[symbol]))
                                current_price = float(trade[symbol].price)
                                qty = int(trade_amount // current_price)
                                
                                if qty <= 0:
                                    return f"ERROR: Insufficient funds to buy even 1 share of {symbol} at ${current_price:.2f}."
                                
                                market_order_data = MarketOrderRequest(
                                    symbol=symbol,
                                    qty=qty,
                                    side=OrderSide.BUY,
                                    time_in_force=TimeInForce.DAY
                                )
                                order = self.trading_client.submit_order(order_data=market_order_data)
                                return f"SUCCESS: Submitted BUY order for {qty} shares of {symbol} (Fallback Integer Math). Order ID: {order.id}"
                            except Exception as qty_error:
                                return f"ERROR executing trade (Notional failed: {str(notional_error)} | Qty fallback failed: {str(qty_error)})"
                    else:
                        return f"SKIPPED: 1% of equity (${trade_amount:.2f}) is below the $10 minimum trade size."
                else:
                    return f"SKIPPED: Already holding a Long position for {symbol}."
            
            elif signal == 0:
                # We want to be Flat
                open_orders = self.trading_client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol]))
                
                if current_position is not None and float(current_position.qty) > 0:
                    try:
                        # We hold a position, liquidate it
                        close_info = self.trading_client.close_position(symbol)
                        return f"SUCCESS: Liquidated position for {symbol}. Order ID: {close_info.id}"
                    except Exception as e:
                        # Alpaca might queue it if market is closed, or reject it.
                        return f"API RESPONSE (Sell): {str(e)}"
                else:
                    if open_orders:
                        for o in open_orders:
                            self.trading_client.cancel_order_by_id(o.id)
                        return f"SUCCESS: Cancelled pending orders for {symbol} to remain flat."
                    return f"SKIPPED: Already flat on {symbol}. No position to close."
            
            else:
                return f"ERROR: Invalid signal {signal}. Must be 1 or 0."
                
        except Exception as e:
            return f"ERROR executing trade: {str(e)}"

    def get_position(self, symbol: str) -> dict:
        """Returns current held shares, average entry price, and unrealized P&L."""
        try:
            positions = self.trading_client.get_all_positions()
            pos = next((p for p in positions if p.symbol == symbol), None)
            if pos:
                return {
                    'shares': float(pos.qty),
                    'market_value': float(pos.market_value),
                    'avg_entry_price': float(pos.avg_entry_price),
                    'unrealized_pl': float(pos.unrealized_pl),
                    'unrealized_plpc': float(pos.unrealized_plpc)
                }
            return None
        except Exception as e:
            return None

    def get_trade_logs(self, symbol: str) -> list:
        """Fetches the closed orders/activities for this specific ticker."""
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            
            req = GetOrdersRequest(
                status=QueryOrderStatus.CLOSED,
                symbols=[symbol],
                limit=100
            )
            orders = self.trading_client.get_orders(filter=req)
            
            logs = []
            for o in orders:
                logs.append({
                    'created_at': o.created_at.strftime('%Y-%m-%d %H:%M:%S') if o.created_at else 'Unknown',
                    'side': str(o.side.value).upper() if o.side else 'UNKNOWN',
                    'qty': float(o.filled_qty) if o.filled_qty else 0.0,
                    'filled_avg_price': float(o.filled_avg_price) if o.filled_avg_price else 0.0,
                    'status': str(o.status.value).upper() if o.status else 'UNKNOWN'
                })
            return logs
        except Exception as e:
            return []

    def get_portfolio_capital(self) -> dict:
        """
        Dynamically calculates the true total equity and currently allocated capital 
        across all open positions directly from the Alpaca API.
        """
        try:
            account = self.trading_client.get_account()
            positions = self.trading_client.get_all_positions()
            
            total_equity = float(account.equity)
            allocated_capital = sum(float(pos.market_value) for pos in positions)
            
            return {
                "total_equity": total_equity,
                "allocated_capital": allocated_capital
            }
        except Exception as e:
            return {"total_equity": 100000.0, "allocated_capital": 0.0}
