import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class BrokerAdapter:
    def __init__(self, config: Dict):
        self.config = config
        self.alpaca_config = config.get("alpaca", {})
        self.client = None
        self.paper_trading = config.get("trading", {}).get("paper_trading", True)
        self._initialize_client()

    def _initialize_client(self) -> None:
        try:
            from alpaca.trading.client import TradingClient
            api_key = self.alpaca_config.get("api_key")
            secret_key = self.alpaca_config.get("secret_key")
            base_url = self.alpaca_config.get("base_url", "https://paper-api.alpaca.markets")
            if not api_key or not secret_key:
                raise ValueError("Alpaca API key and secret key required")
            self.client = TradingClient(api_key=api_key, secret_key=secret_key, base_url=base_url)
            logger.info(f"Connected to Alpaca: {base_url}")
        except ImportError:
            raise ImportError("alpaca-py required. Install with: pip install alpaca-py")

    def get_account(self) -> Dict:
        try:
            account = self.client.get_account()
            return {"buying_power": float(account.buying_power), "cash": float(account.cash),
                   "portfolio_value": float(account.portfolio_value), "equity": float(account.equity)}
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return {}

    def get_positions(self) -> Dict:
        try:
            positions = self.client.get_all_positions()
            return {pos.symbol: {"qty": float(pos.qty), "avg_fill_price": float(pos.avg_fill_price),
                               "current_price": float(pos.current_price), "unrealized_pl": float(pos.unrealized_pl)}
                   for pos in positions}
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return {}

    def submit_order(self, symbol: str, qty: float, side: str, order_type: str = "market", time_in_force: str = "day") -> Optional[Dict]:
        try:
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce
            side_enum = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
            tif_enum = TimeInForce.DAY if time_in_force == "day" else TimeInForce.GTC
            order = self.client.submit_order(MarketOrderRequest(symbol=symbol, qty=qty, side=side_enum, time_in_force=tif_enum))
            logger.info(f"Order submitted: {side.upper()} {qty} {symbol}")
            return {"order_id": order.id, "symbol": order.symbol, "qty": float(order.qty), "side": order.side.value, "status": order.status.value}
        except Exception as e:
            logger.error(f"Failed to submit order: {e}")
            return None

    def close_position(self, symbol: str) -> Optional[Dict]:
        try:
            order = self.client.close_position(symbol)
            logger.info(f"Position closed: {symbol}")
            return {"order_id": order.id, "symbol": order.symbol, "qty": float(order.qty), "side": order.side.value}
        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            return None
