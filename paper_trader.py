import os
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

class PaperTrader:
    def __init__(self, api_key: str, secret_key: str):
        self.api_key = api_key
        self.secret_key = secret_key
        
        # CRITICAL SAFETY CHECK: Hardcode the paper trading URL
        self.base_url = "https://paper-api.alpaca.markets"
        
        self.trading_client = TradingClient(
            api_key=self.api_key, 
            secret_key=self.secret_key, 
            paper=True
        )
        
        # Double check to ensure it's pointing to the paper environment
        if "paper-api.alpaca.markets" not in self.trading_client._request._base_url:
            raise ValueError("CRITICAL ERROR: Trading client is NOT pointing to the paper trading URL. Execution aborted.")

    def execute_trade(self, symbol: str, signal: int):
        """
        Executes a paper trade based on the given signal.
        signal == 1: Long (Market Buy)
        signal == 0: Flat (Liquidate position)
        """
        try:
            # Check current position
            positions = self.trading_client.get_all_positions()
            current_position = next((p for p in positions if p.symbol == symbol), None)
            
            if signal == 1:
                # We want to be Long
                if current_position is None:
                    # Not in position, so we buy
                    account = self.trading_client.get_account()
                    
                    # We will use 95% of available buying power for this trade to avoid margin issues
                    buying_power = float(account.buying_power)
                    trade_amount = buying_power * 0.95
                    
                    if trade_amount > 10:  # Minimum threshold for a trade
                        market_order_data = MarketOrderRequest(
                            symbol=symbol,
                            notional=round(trade_amount, 2), # Buy fractional shares based on dollar amount
                            side=OrderSide.BUY,
                            time_in_force=TimeInForce.DAY
                        )
                        order = self.trading_client.submit_order(order_data=market_order_data)
                        return f"SUCCESS: Submitted BUY order for ${trade_amount:.2f} of {symbol}. Order ID: {order.id}"
                    else:
                        return f"SKIPPED: Insufficient buying power (${buying_power:.2f}) to execute trade."
                else:
                    return f"SKIPPED: Already holding a Long position for {symbol}."
            
            elif signal == 0:
                # We want to be Flat
                if current_position is not None:
                    # We hold a position, liquidate it
                    close_info = self.trading_client.close_position(symbol)
                    return f"SUCCESS: Liquidated position for {symbol}. Order ID: {close_info.id}"
                else:
                    return f"SKIPPED: Already flat on {symbol}. No position to close."
            
            else:
                return f"ERROR: Invalid signal {signal}. Must be 1 or 0."
                
        except Exception as e:
            return f"ERROR executing trade: {str(e)}"
