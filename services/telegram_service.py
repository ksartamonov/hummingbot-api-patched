import asyncio
import io
import json
import logging
import zipfile
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from telegram import Bot
from telegram.error import TelegramError

from config import settings

logger = logging.getLogger(__name__)


class TelegramService:
    """Service for sending Telegram notifications about backtesting results."""
    
    def __init__(self):
        self.bot: Optional[Bot] = None
        self.chat_id: Optional[str] = None
        self.enabled: bool = False
        
        if settings.telegram.enabled and settings.telegram.bot_token:
            self.bot = Bot(token=settings.telegram.bot_token)
            self.chat_id = settings.telegram.chat_id
            self.enabled = True
            logger.info("Telegram service initialized")
        else:
            logger.warning("Telegram service disabled - missing token or not enabled")
    
    async def send_backtesting_summary(
        self,
        config: Dict[str, Any],
        results: Dict[str, Any],
        executors: List[Dict[str, Any]],
        processed_data: Dict[str, Any]
    ) -> bool:
        """
        Send a comprehensive backtesting summary to Telegram.
        
        Args:
            config: Backtesting configuration
            results: Backtesting results
            executors: List of executors information
            processed_data: Processed market data
            
        Returns:
            bool: True if message sent successfully, False otherwise
        """
        if not self.enabled or not self.bot or not self.chat_id:
            logger.warning("Telegram service not available")
            return False
        
        try:
            # Create summary message
            message = self._create_summary_message(config, results, executors)
            
            # Send main summary
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='HTML'
            )
            
            # Create structured CSV files and ZIP archive
            zip_buf = self._create_backtest_report_zip(config, results, executors, processed_data)
            
            # Send ZIP archive with structured data
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            strategy_name = self._extract_strategy_name(config).replace(' ', '_').lower()
            
            await self.bot.send_document(
                chat_id=self.chat_id,
                document=zip_buf,
                filename=f"{strategy_name}_backtest_report_{timestamp}.zip"
            )
            
            logger.info("Backtesting summary sent to Telegram successfully")
            return True
            
        except TelegramError as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending Telegram message: {e}")
            return False
    
    def _create_summary_message(
        self,
        config: Dict[str, Any],
        results: Dict[str, Any],
        executors: List[Dict[str, Any]]
    ) -> str:
        """Create a formatted summary message for Telegram."""
        
        # Format timestamps
        start_time = datetime.fromtimestamp(config.get('start_time', 0))
        end_time = datetime.fromtimestamp(config.get('end_time', 0))
        
        # Calculate key metrics - используем правильные ключи из результатов бэктеста
        # Основные метрики
        pnl_absolute = results.get('net_pnl', 0)
        pnl_quote = results.get('net_pnl_quote', 0)
        total_return = results.get('net_pnl_pct', 0)  # если есть процентная метрика
        sharpe_ratio = results.get('sharpe_ratio', 0)
        sortino_ratio = results.get('sortino_ratio', 0)
        calmar_ratio = results.get('calmar_ratio', 0)
        max_drawdown_pct = results.get('max_drawdown_pct', 0)
        max_drawdown_usd = results.get('max_drawdown_usd', 0)
        total_trades = results.get('total_positions', 0)
        accuracy = results.get('accuracy', 0)
        
        # Дополнительные метрики
        total_executors = results.get('total_executors', 0)
        total_executors_with_position = results.get('total_executors_with_position', 0)
        total_volume = results.get('total_volume', 0)
        total_long = results.get('total_long', 0)
        total_short = results.get('total_short', 0)
        accuracy_long = results.get('accuracy_long', 0)
        accuracy_short = results.get('accuracy_short', 0)
        profit_factor = results.get('profit_factor', 0)
        win_signals = results.get('win_signals', 0)
        loss_signals = results.get('loss_signals', 0)
        
        # Проверяем на None/NaN и заменяем на 0
        def safe_value(value):
            if value is None or (hasattr(value, 'isna') and value.isna()):
                return 0
            return value
        
        pnl_absolute = safe_value(pnl_absolute)
        pnl_quote = safe_value(pnl_quote)
        total_return = safe_value(total_return)
        sharpe_ratio = safe_value(sharpe_ratio)
        sortino_ratio = safe_value(sortino_ratio)
        calmar_ratio = safe_value(calmar_ratio)
        max_drawdown_pct = safe_value(max_drawdown_pct)
        max_drawdown_usd = safe_value(max_drawdown_usd)
        total_trades = safe_value(total_trades)
        accuracy = safe_value(accuracy)
        total_executors = safe_value(total_executors)
        total_executors_with_position = safe_value(total_executors_with_position)
        total_volume = safe_value(total_volume)
        total_long = safe_value(total_long)
        total_short = safe_value(total_short)
        accuracy_long = safe_value(accuracy_long)
        accuracy_short = safe_value(accuracy_short)
        profit_factor = safe_value(profit_factor)
        win_signals = safe_value(win_signals)
        loss_signals = safe_value(loss_signals)
        
        # Логируем для отладки
        logger.info(f"Telegram metrics calculation:")
        logger.info(f"  results keys: {list(results.keys())}")
        logger.info(f"  results values: {results}")
        logger.info(f"  PnL (USD): {pnl_absolute}")
        logger.info(f"  PnL (Quote): {pnl_quote}")
        logger.info(f"  Total Return: {total_return}")
        logger.info(f"  Sharpe Ratio: {sharpe_ratio}")
        logger.info(f"  Sortino Ratio: {sortino_ratio}")
        logger.info(f"  Calmar Ratio: {calmar_ratio}")
        logger.info(f"  Max Drawdown %: {max_drawdown_pct}")
        logger.info(f"  Max Drawdown USD: {max_drawdown_usd}")
        logger.info(f"  Total Trades: {total_trades}")
        logger.info(f"  Accuracy: {accuracy}")
        logger.info(f"  Total Executors: {total_executors}")
        logger.info(f"  Active Positions: {total_executors_with_position}")
        logger.info(f"  Total Volume: {total_volume}")
        logger.info(f"  Long/Short: {total_long}/{total_short}")
        logger.info(f"  Long/Short Accuracy: {accuracy_long}/{accuracy_short}")
        logger.info(f"  Profit Factor: {profit_factor}")
        logger.info(f"  Win/Loss Signals: {win_signals}/{loss_signals}")
        
        # Дополнительное логирование для коэффициентов
        logger.info(f"  === COEFFICIENTS DEBUG ===")
        logger.info(f"  Raw results sharpe_ratio: {results.get('sharpe_ratio', 'NOT_FOUND')}")
        logger.info(f"  Raw results sortino_ratio: {results.get('sortino_ratio', 'NOT_FOUND')}")
        logger.info(f"  Raw results calmar_ratio: {results.get('calmar_ratio', 'NOT_FOUND')}")
        logger.info(f"  Calculated sharpe_ratio: {sharpe_ratio}")
        logger.info(f"  Calculated sortino_ratio: {sortino_ratio}")
        logger.info(f"  Calculated calmar_ratio: {calmar_ratio}")
        logger.info(f"  === END COEFFICIENTS DEBUG ===")
        
        # Create message
        message = f"""
🚀 <b>Backtesting Results Summary</b>

📅 <b>Period:</b> {start_time.strftime('%Y-%m-%d %H:%M')} - {end_time.strftime('%Y-%m-%d %H:%M')}
⏱️ <b>Resolution:</b> {config.get('backtesting_resolution', 'N/A')}
💰 <b>Trade Cost:</b> {config.get('trade_cost', 0):.4f}

📊 <b>Performance Metrics:</b>
• PnL (USD): <b>{pnl_absolute:.4f}</b>
• PnL (Quote): <b>{pnl_quote:.4f}</b>
• Total Return: <b>{total_return:.2f}%</b>
• Sharpe Ratio: <b>{sharpe_ratio:.4f}</b>
• Sortino Ratio: <b>{sortino_ratio:.4f}</b>
• Calmar Ratio: <b>{calmar_ratio:.4f}</b>
• Max Drawdown: <b>{max_drawdown_pct:.2f}%</b>
• Max Drawdown (USD): <b>{max_drawdown_usd:.4f}</b>
• Total Trades: <b>{total_trades}</b>
• Accuracy: <b>{accuracy:.2f}%</b>

📈 <b>Position Details:</b>
• Total Executors: <b>{total_executors}</b>
• Active Positions: <b>{total_executors_with_position}</b>
• Total Volume: <b>{total_volume:.4f}</b>
• Long Positions: <b>{total_long}</b>
• Short Positions: <b>{total_short}</b>

🎯 <b>Directional Accuracy:</b>
• Long Accuracy: <b>{accuracy_long:.2f}%</b>
• Short Accuracy: <b>{accuracy_short:.2f}%</b>
• Profit Factor: <b>{profit_factor:.2f}</b>
• Win Signals: <b>{win_signals}</b>
• Loss Signals: <b>{loss_signals}</b>

🤖 <b>Executors:</b> {len(executors)} active

📈 <b>Strategy:</b> {self._extract_strategy_name(config)}

🔍 <b>All Available Metrics:</b>
• {', '.join([f'{k}: {v:.4f}' if isinstance(v, (int, float)) else f'{k}: {v}' for k, v in results.items() if v is not None and v != 0]) if any(v is not None and v != 0 for v in results.values()) else 'No additional metrics available'}
        """.strip()
        
        return message
    
    def _extract_strategy_name(self, config: Dict[str, Any]) -> str:
        """Extract strategy name from configuration."""
        # Сначала проверяем controller_name
        if config.get('controller_name'):
            return config['controller_name'].replace('_', ' ').title()
        
        # Затем проверяем config.controller_name
        if isinstance(config.get('config'), dict):
            controller_name = config['config'].get('controller_name')
            if controller_name:
                return controller_name.replace('_', ' ').title()
            
            strategy_name = config['config'].get('strategy_name')
            if strategy_name:
                return strategy_name.replace('_', ' ').title()
        
        # Затем проверяем config как строку
        elif isinstance(config.get('config'), str):
            # Try to extract from file path
            return config['config'].split('/')[-1].replace('.yml', '').replace('.yaml', '').replace('_', ' ').title()
        
        # Если ничего не найдено, возвращаем PMM Dynamic как дефолт
        return 'PMM Dynamic'
    
    def _stringify(self, v: Any) -> str:
        """Convert complex objects to string representation."""
        if isinstance(v, (list, dict)):
            return json.dumps(v, ensure_ascii=False)
        return str(v)
    
    def _create_backtest_report_zip(
        self,
        config: Dict[str, Any],
        results: Dict[str, Any],
        executors: List[Dict[str, Any]],
        processed_data: Dict[str, Any]
    ) -> io.BytesIO:
        """
        Create a ZIP archive with structured CSV files for backtesting report.
        
        Args:
            config: Backtesting configuration
            results: Backtesting results
            executors: List of executors information
            processed_data: Processed market data
            
        Returns:
            BytesIO buffer containing the ZIP archive
        """
        # Create trades DataFrame
        trades_df = pd.DataFrame(executors)
        
        # Create config DataFrame (param/value)
        cfg_series = pd.Series({k: self._stringify(v) for k, v in config.items()}, name="value")
        config_df = cfg_series.to_frame().reset_index().rename(columns={"index": "param"})
        
        # Create metrics DataFrame (metric/value)
        metrics_series = pd.Series({k: self._stringify(v) for k, v in results.items()}, name="value")
        metrics_df = metrics_series.to_frame().reset_index().rename(columns={"index": "metric"})
        
        # Create timeseries DataFrame
        ts_raw = processed_data
        if isinstance(ts_raw, dict) and 'features' in ts_raw:
            try:
                timeseries_df = pd.DataFrame(ts_raw['features'])
            except Exception:
                timeseries_df = pd.DataFrame()
        elif isinstance(ts_raw, pd.DataFrame):
            timeseries_df = ts_raw
        else:
            try:
                timeseries_df = pd.DataFrame(ts_raw)
            except Exception:
                timeseries_df = pd.DataFrame()
        
        # Create ZIP archive
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            # Add CSV files
            zf.writestr("trades.csv", trades_df.to_csv(index=False))
            zf.writestr("config.csv", config_df.to_csv(index=False))
            zf.writestr("metrics.csv", metrics_df.to_csv(index=False))
            
            if not timeseries_df.empty:
                zf.writestr("timeseries.csv", timeseries_df.to_csv(index=False))
            
            # Add README file
            readme_content = (
                "Backtesting Report Files:\n"
                "========================\n\n"
                "- trades.csv — все сделки (executors)\n"
                "- config.csv — параметры конфигурации\n"
                "- metrics.csv — метрики бэктеста\n"
                "- timeseries.csv — временные ряды (если были)\n\n"
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Strategy: {self._extract_strategy_name(config)}\n"
                f"Period: {datetime.fromtimestamp(config.get('start_time', 0)).strftime('%Y-%m-%d %H:%M')} - "
                f"{datetime.fromtimestamp(config.get('end_time', 0)).strftime('%Y-%m-%d %H:%M')}"
            )
            zf.writestr("README.txt", readme_content)
        
        zip_buf.seek(0)
        return zip_buf
    
    async def send_simple_notification(self, message: str) -> bool:
        """
        Send a simple text notification.
        
        Args:
            message: Text message to send
            
        Returns:
            bool: True if message sent successfully, False otherwise
        """
        if not self.enabled or not self.bot or not self.chat_id:
            logger.warning("Telegram service not available")
            return False
        
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='HTML'
            )
            logger.info("Simple notification sent to Telegram successfully")
            return True
        except TelegramError as e:
            logger.error(f"Failed to send Telegram notification: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending Telegram notification: {e}")
            return False


# Global instance
telegram_service = TelegramService() 