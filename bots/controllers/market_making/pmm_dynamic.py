from decimal import Decimal
from typing import List, Set

import pandas_ta as ta  # noqa: F401
from pydantic import Field, field_validator
from pydantic_core.core_schema import ValidationInfo

from hummingbot.core.data_type.common import PositionMode, TradeType
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers.market_making_controller_base import (
    MarketMakingControllerBase,
    MarketMakingControllerConfigBase,
)
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction


class PMMDynamicControllerConfig(MarketMakingControllerConfigBase):
    controller_name: str = "pmm_dynamic"
    candles_config: List[CandlesConfig] = []
    buy_spreads: List[float] = Field(
        default="1,2,4",
        json_schema_extra={
            "prompt": "Enter a comma-separated list of buy spreads measured in units of volatility(e.g., '1, 2'): ",
            "prompt_on_new": True, "is_updatable": True}
    )
    sell_spreads: List[float] = Field(
        default="1,2,4",
        json_schema_extra={
            "prompt": "Enter a comma-separated list of sell spreads measured in units of volatility(e.g., '1, 2'): ",
            "prompt_on_new": True, "is_updatable": True}
    )
    candles_connector: str = Field(
        default=None,
        json_schema_extra={
            "prompt": "Enter the connector for the candles data, leave empty to use the same exchange as the connector: ",
            "prompt_on_new": True})
    candles_trading_pair: str = Field(
        default=None,
        json_schema_extra={
            "prompt": "Enter the trading pair for the candles data, leave empty to use the same trading pair as the connector: ",
            "prompt_on_new": True})
    interval: str = Field(
        default="3m",
        json_schema_extra={
            "prompt": "Enter the candle interval (e.g., 1m, 5m, 1h, 1d): ",
            "prompt_on_new": True})
    macd_fast: int = Field(
        default=21,
        json_schema_extra={"prompt": "Enter the MACD fast period: ", "prompt_on_new": True})
    macd_slow: int = Field(
        default=42,
        json_schema_extra={"prompt": "Enter the MACD slow period: ", "prompt_on_new": True})
    macd_signal: int = Field(
        default=9,
        json_schema_extra={"prompt": "Enter the MACD signal period: ", "prompt_on_new": True})
    natr_length: int = Field(
        default=14,
        json_schema_extra={"prompt": "Enter the NATR length: ", "prompt_on_new": True})

    @field_validator("candles_connector", mode="before")
    @classmethod
    def set_candles_connector(cls, v, validation_info: ValidationInfo):
        if v is None or v == "":
            return validation_info.data.get("connector_name")
        return v

    @field_validator("candles_trading_pair", mode="before")
    @classmethod
    def set_candles_trading_pair(cls, v, validation_info: ValidationInfo):
        if v is None or v == "":
            return validation_info.data.get("trading_pair")
        return v


class PMMDynamicController(MarketMakingControllerBase):
    """
    This is a dynamic version of the PMM controller.It uses the MACD to shift the mid-price and the NATR
    to make the spreads dynamic. It also uses the Triple Barrier Strategy to manage the risk.
    """

    def __init__(self, config: PMMDynamicControllerConfig, *args, **kwargs):
        self.config = config
        self.max_records = max(config.macd_slow, config.macd_fast, config.macd_signal, config.natr_length) + 100
        if len(self.config.candles_config) == 0:
            self.config.candles_config = [CandlesConfig(
                connector=config.candles_connector,
                trading_pair=config.candles_trading_pair,
                interval=config.interval,
                max_records=self.max_records
            )]
        super().__init__(config, *args, **kwargs)

    async def update_processed_data(self):
        candles = self.market_data_provider.get_candles_df(connector_name=self.config.candles_connector,
                                                           trading_pair=self.config.candles_trading_pair,
                                                           interval=self.config.interval,
                                                           max_records=self.max_records)

        # Создаем копию DataFrame чтобы избежать SettingWithCopyWarning
        candles = candles.copy()

        natr = ta.natr(candles["high"], candles["low"], candles["close"], length=self.config.natr_length) / 100
        macd_output = ta.macd(candles["close"], fast=self.config.macd_fast,
                              slow=self.config.macd_slow, signal=self.config.macd_signal)
        macd = macd_output[f"MACD_{self.config.macd_fast}_{self.config.macd_slow}_{self.config.macd_signal}"]
        macd_signal = - (macd - macd.mean()) / macd.std()
        macdh = macd_output[f"MACDh_{self.config.macd_fast}_{self.config.macd_slow}_{self.config.macd_signal}"]
        macdh_signal = macdh.apply(lambda x: 1 if x > 0 else -1)
        max_price_shift = natr / 2
        price_multiplier = ((0.5 * macd_signal + 0.5 * macdh_signal) * max_price_shift).iloc[-1]

        # Теперь безопасно добавляем новые колонки
        candles["spread_multiplier"] = natr
        candles["reference_price"] = candles["close"] * (1 + price_multiplier)

        self.processed_data = {
            "reference_price": Decimal(candles["reference_price"].iloc[-1]),
            "spread_multiplier": Decimal(candles["spread_multiplier"].iloc[-1]),
            "features": candles
        }

    def create_actions_proposal(self):
        actions = super().create_actions_proposal()
        allowed_trade_types = self._resolve_allowed_trade_types(self._get_current_base_pct())
        filtered_actions = []
        for action in actions:
            executor_config = getattr(action, "executor_config", None)
            if isinstance(action, CreateExecutorAction) and executor_config is not None:
                side = getattr(executor_config, "side", None)
                if side is not None and side not in allowed_trade_types:
                    continue
            filtered_actions.append(action)
        return filtered_actions

    def _get_current_base_pct(self) -> Decimal:
        processed_pct = self.processed_data.get("current_base_pct") if isinstance(self.processed_data, dict) else None
        if processed_pct is not None:
            try:
                return Decimal(str(processed_pct))
            except Exception:
                pass
        position = next((position for position in self.positions_held
                         if position.connector_name == self.config.connector_name and
                         position.trading_pair == self.config.trading_pair), None)
        if position is None:
            return Decimal("0")
        total_amount_quote = getattr(self.config, "total_amount_quote", None)
        if total_amount_quote in (None, 0):
            return Decimal("0")
        try:
            total_quote_decimal = Decimal(str(total_amount_quote))
        except Exception:
            total_quote_decimal = Decimal(total_amount_quote)
        if total_quote_decimal == 0:
            return Decimal("0")
        return position.amount_quote / total_quote_decimal

    def _resolve_allowed_trade_types(self, current_pct: Decimal) -> Set[TradeType]:
        if self.config.position_mode != PositionMode.ONEWAY:
            return {TradeType.BUY, TradeType.SELL}

        tolerance_value = getattr(self.config, "position_rebalance_threshold_pct", Decimal("0"))
        if not isinstance(tolerance_value, Decimal):
            try:
                tolerance_value = Decimal(str(tolerance_value))
            except Exception:
                tolerance_value = Decimal("0")
        if tolerance_value == Decimal("0"):
            tolerance_value = Decimal("0.0001")

        active_buy = any(self._executor_matches_side(info, TradeType.BUY) for info in self.executors_info if info.is_active)
        active_sell = any(self._executor_matches_side(info, TradeType.SELL) for info in self.executors_info if info.is_active)

        if active_buy and not active_sell:
            return {TradeType.BUY}
        if active_sell and not active_buy:
            return {TradeType.SELL}

        target_pct = getattr(self.config, "target_base_pct", Decimal("0"))
        if not isinstance(target_pct, Decimal):
            try:
                target_pct = Decimal(str(target_pct))
            except Exception:
                target_pct = Decimal("0")
        if current_pct > target_pct + tolerance_value:
            return {TradeType.SELL}
        if current_pct < target_pct - tolerance_value:
            return {TradeType.BUY}
        return {TradeType.BUY, TradeType.SELL}

    def _executor_matches_side(self, executor, trade_type: TradeType) -> bool:
        custom_info = getattr(executor, "custom_info", {}) or {}
        level_id = custom_info.get("level_id") if isinstance(custom_info, dict) else None
        if isinstance(level_id, str) and "_" in level_id:
            try:
                return self.get_trade_type_from_level_id(level_id) == trade_type
            except Exception:
                return False
        executor_side = getattr(executor, "side", None)
        return executor_side == trade_type

    def get_executor_config(self, level_id: str, price: Decimal, amount: Decimal):
        trade_type = self.get_trade_type_from_level_id(level_id)
        return PositionExecutorConfig(
            timestamp=self.market_data_provider.time(),
            level_id=level_id,
            connector_name=self.config.connector_name,
            trading_pair=self.config.trading_pair,
            entry_price=price,
            amount=amount,
            triple_barrier_config=self.config.triple_barrier_config,
            leverage=self.config.leverage,
            side=trade_type,
        )
