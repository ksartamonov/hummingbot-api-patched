import asyncio
import json
import io
import zipfile
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

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
        
        # Calculate key metrics
        total_return = results.get('total_return', 0)
        sharpe_ratio = results.get('sharpe_ratio', 0)
        max_drawdown = results.get('max_drawdown', 0)
        total_trades = results.get('total_trades', 0)
        win_rate = results.get('win_rate', 0)
        
        # Create message
        message = f"""
🚀 <b>Backtesting Results Summary</b>

📅 <b>Period:</b> {start_time.strftime('%Y-%m-%d %H:%M')} - {end_time.strftime('%Y-%m-%d %H:%M')}
⏱️ <b>Resolution:</b> {config.get('backtesting_resolution', 'N/A')}
💰 <b>Trade Cost:</b> {config.get('trade_cost', 0):.4f}

📊 <b>Performance Metrics:</b>
• Total Return: <b>{total_return:.2%}</b>
• Sharpe Ratio: <b>{sharpe_ratio:.2f}</b>
• Max Drawdown: <b>{max_drawdown:.2%}</b>
• Total Trades: <b>{total_trades}</b>
• Win Rate: <b>{win_rate:.2%}</b>

🤖 <b>Executors:</b> {len(executors)} active

📈 <b>Strategy:</b> {self._extract_strategy_name(config)}
        """.strip()
        
        return message
    
    def _extract_strategy_name(self, config: Dict[str, Any]) -> str:
        """Extract strategy name from configuration."""
        if isinstance(config.get('config'), dict):
            return config['config'].get('strategy_name', 'Unknown Strategy')
        elif isinstance(config.get('config'), str):
            # Try to extract from file path
            return config['config'].split('/')[-1].replace('.yml', '').replace('.yaml', '')
        return 'Unknown Strategy'
    
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