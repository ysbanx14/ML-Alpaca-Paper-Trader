import json
import os

class PortfolioManager:
    def __init__(self, file_path="tracked_tickers.json"):
        self.file_path = file_path
        if not os.path.exists(self.file_path):
            with open(self.file_path, "w") as f:
                json.dump([], f)

    def get_tracked_tickers(self) -> list:
        try:
            with open(self.file_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []

    def add_ticker(self, symbol: str):
        tickers = self.get_tracked_tickers()
        symbol = symbol.upper().strip()
        if symbol not in tickers:
            tickers.append(symbol)
            with open(self.file_path, "w") as f:
                json.dump(tickers, f, indent=4)
            return True
        return False

    def remove_ticker(self, symbol: str):
        tickers = self.get_tracked_tickers()
        symbol = symbol.upper().strip()
        if symbol in tickers:
            tickers.remove(symbol)
            with open(self.file_path, "w") as f:
                json.dump(tickers, f, indent=4)
            return True
        return False
