import os
from dotenv import load_dotenv
load_dotenv()
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

client = TradingClient(os.getenv("APCA_API_KEY_ID"), os.getenv("APCA_API_SECRET_KEY"), paper=True)
req = GetOrdersRequest(status=QueryOrderStatus.CLOSED, symbols=["AAPL"])
orders = client.get_orders(filter=req)
print(f"Found {len(orders)} orders.")
