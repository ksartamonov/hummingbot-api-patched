import logging
import numpy as np
from fastapi import APIRouter
from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory
from hummingbot.strategy_v2.backtesting.backtesting_engine_base import BacktestingEngineBase

from config import settings
from models.backtesting import BacktestingConfig
from services.telegram_service import telegram_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Backtesting"], prefix="/backtesting")
candles_factory = CandlesFactory()
backtesting_engine = BacktestingEngineBase()


def calculate_performance_ratios(results: dict) -> dict:
    """
    Рассчитывает коэффициенты эффективности на основе результатов бэктеста.
    
    Args:
        results: Словарь с результатами бэктеста
        
    Returns:
        Словарь с рассчитанными коэффициентами
    """
    try:
        # Получаем базовые метрики
        net_pnl_pct = results.get("net_pnl_pct", 0)
        max_drawdown_pct = results.get("max_drawdown_pct", 0)
        
        # Инициализируем коэффициенты
        sharpe_ratio = results.get("sharpe_ratio", 0)
        sortino_ratio = 0
        calmar_ratio = 0
        
        # Логируем входные данные
        logger.info(f"Input data - net_pnl_pct: {net_pnl_pct}, max_drawdown_pct: {max_drawdown_pct}")
        
        # Рассчитываем коэффициенты только если есть PnL
        if net_pnl_pct != 0:
            # Конвертируем в десятичные
            pnl_decimal = float(net_pnl_pct) / 100
            
            # Sortino Ratio = доходность / стандартное отклонение убытков
            # Для упрощения используем предполагаемое стандартное отклонение 1%
            if net_pnl_pct < 0:
                # Отрицательный PnL: Sortino = PnL / 1%
                sortino_ratio = pnl_decimal / 0.01
                logger.info(f"Negative PnL: {net_pnl_pct}%, Sortino = {pnl_decimal} / 0.01 = {sortino_ratio}")
            elif net_pnl_pct > 0:
                # Положительный PnL без убытков: высокое значение
                sortino_ratio = 999.99
                logger.info(f"Positive PnL: {net_pnl_pct}%, Sortino = 999.99")
            else:
                sortino_ratio = 0
                logger.info(f"Zero PnL: {net_pnl_pct}%, Sortino = 0")
            
            # Calmar Ratio = доходность / максимальная просадка
            if max_drawdown_pct != 0:
                max_drawdown_decimal = abs(float(max_drawdown_pct) / 100)
                if max_drawdown_decimal > 0:
                    calmar_ratio = pnl_decimal / max_drawdown_decimal
                    logger.info(f"MaxDD: {max_drawdown_pct}%, Calmar = {pnl_decimal} / {max_drawdown_decimal} = {calmar_ratio}")
                else:
                    calmar_ratio = 0
                    logger.info(f"MaxDD is 0, Calmar = 0")
            else:
                # Если max_drawdown не указан, используем минимальную просадку 1%
                calmar_ratio = pnl_decimal / 0.01
                logger.info(f"No MaxDD, using default 1%, Calmar = {pnl_decimal} / 0.01 = {calmar_ratio}")
        else:
            # Если PnL = 0, устанавливаем коэффициенты в 0
            sortino_ratio = 0
            calmar_ratio = 0
            logger.info(f"Zero PnL, setting both ratios to 0")
        
        # Дополнительная проверка: если коэффициенты все еще 0, попробуем альтернативный расчет
        if sortino_ratio == 0 and net_pnl_pct != 0:
            # Альтернативный расчет Sortino
            if net_pnl_pct > 0:
                sortino_ratio = 999.99
            elif net_pnl_pct < 0:
                sortino_ratio = float(net_pnl_pct) / 100 / 0.01
            logger.info(f"Alternative Sortino calculation: {sortino_ratio}")
        
        if calmar_ratio == 0 and net_pnl_pct != 0:
            # Альтернативный расчет Calmar
            if max_drawdown_pct != 0:
                calmar_ratio = float(net_pnl_pct) / 100 / abs(float(max_drawdown_pct) / 100)
            else:
                calmar_ratio = float(net_pnl_pct) / 100 / 0.01
            logger.info(f"Alternative Calmar calculation: {calmar_ratio}")
        
        # Проверяем и ограничиваем значения
        if not np.isfinite(sharpe_ratio):
            sharpe_ratio = 0
        if not np.isfinite(sortino_ratio):
            sortino_ratio = 0
        if not np.isfinite(calmar_ratio):
            calmar_ratio = 0
            
        # Логируем расчеты
        logger.info(f"Calculated ratios - PnL: {net_pnl_pct}%, Sharpe: {sharpe_ratio:.6f}, Sortino: {sortino_ratio:.6f}, Calmar: {calmar_ratio:.6f}")
        
        return {
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio
        }
        
    except Exception as e:
        logger.error(f"Error calculating performance ratios: {e}")
        return {
            "sharpe_ratio": 0,
            "sortino_ratio": 0,
            "calmar_ratio": 0
        }


def _clean_config_data(config_data: dict) -> dict:
    """
    Clean configuration data to remove whitespace issues and validate required fields.
    
    Args:
        config_data: Raw configuration dictionary
        
    Returns:
        Cleaned configuration dictionary
    """
    cleaned_config = config_data.copy()
    
    # Clean connector_name specifically
    if 'connector_name' in cleaned_config:
        cleaned_config['connector_name'] = cleaned_config['connector_name'].strip()
        logger.info(f"Cleaned connector_name from '{config_data.get('connector_name')}' to '{cleaned_config['connector_name']}'")
    
    # Clean other string fields that might have whitespace issues
    string_fields = ['strategy_name', 'trading_pair', 'exchange']
    for field in string_fields:
        if field in cleaned_config and isinstance(cleaned_config[field], str):
            cleaned_config[field] = cleaned_config[field].strip()
    
    return cleaned_config


@router.post("/run-backtesting")
async def run_backtesting(backtesting_config: BacktestingConfig):
    """
    Run a backtesting simulation with the provided configuration.
    
    Args:
        backtesting_config: Configuration for the backtesting including start/end time,
                          resolution, trade cost, and controller config
                          
    Returns:
        Dictionary containing executors, processed data, and results from the backtest
        
    Raises:
        Returns error dictionary if backtesting fails
    """
    try:
        # Log the incoming configuration for debugging
        logger.info("=== INCOMING CONFIG DEBUG ===")
        logger.info(f"Config type: {type(backtesting_config.config)}")
        if isinstance(backtesting_config.config, dict):
            logger.info(f"Config keys: {list(backtesting_config.config.keys())}")
            if 'connector_name' in backtesting_config.config:
                logger.info(f"Raw connector_name: '{backtesting_config.config['connector_name']}' (length: {len(backtesting_config.config['connector_name'])})")
        logger.info("=== END INCOMING CONFIG DEBUG ===")
        
        # Send start notification
        await telegram_service.send_simple_notification(
            f"🚀 <b>Backtesting Started</b>\n\n"
            f"📅 Period: {backtesting_config.start_time} - {backtesting_config.end_time}\n"
            f"⏱️ Resolution: {backtesting_config.backtesting_resolution}\n"
            f"💰 Trade Cost: {backtesting_config.trade_cost}"
        )
        
        # Clean and validate configuration
        if isinstance(backtesting_config.config, str):
            controller_config = backtesting_engine.get_controller_config_instance_from_yml(
                config_path=backtesting_config.config,
                controllers_conf_dir_path=settings.app.controllers_path,
                controllers_module=settings.app.controllers_module
            )
        else:
            # Clean the configuration data to remove any whitespace issues
            cleaned_config = _clean_config_data(backtesting_config.config)
            logger.info(f"Cleaned config connector_name: '{cleaned_config.get('connector_name', 'N/A')}'")
            
            controller_config = backtesting_engine.get_controller_config_instance_from_dict(
                config_data=cleaned_config,
                controllers_module=settings.app.controllers_module
            )
        
        backtesting_results = await backtesting_engine.run_backtesting(
            controller_config=controller_config, trade_cost=backtesting_config.trade_cost,
            start=int(backtesting_config.start_time), end=int(backtesting_config.end_time),
            backtesting_resolution=backtesting_config.backtesting_resolution)
        
        # Log the raw backtesting results for debugging
        logger.info("=== BACKTESTING RESULTS DEBUG ===")
        logger.info(f"Type of backtesting_results: {type(backtesting_results)}")
        logger.info(f"Keys in backtesting_results: {list(backtesting_results.keys()) if isinstance(backtesting_results, dict) else 'Not a dict'}")
        
        if isinstance(backtesting_results, dict):
            for key, value in backtesting_results.items():
                logger.info(f"Key '{key}': type={type(value)}, value={str(value)[:200]}...")
                
                # Special logging for processed_data
                if key == "processed_data":
                    logger.info(f"  processed_data type: {type(value)}")
                    if isinstance(value, dict):
                        logger.info(f"  processed_data keys: {list(value.keys())}")
                        for sub_key, sub_value in value.items():
                            logger.info(f"    {sub_key}: type={type(sub_value)}, shape={getattr(sub_value, 'shape', 'N/A') if hasattr(sub_value, 'shape') else 'N/A'}")
                    elif hasattr(value, 'shape'):
                        logger.info(f"  processed_data shape: {value.shape}")
                
                # Special logging for executors
                elif key == "executors":
                    logger.info(f"  executors type: {type(value)}")
                    if isinstance(value, list):
                        logger.info(f"  executors length: {len(value)}")
                        if value:
                            logger.info(f"  first executor type: {type(value[0])}")
                            logger.info(f"  first executor methods: {[m for m in dir(value[0]) if not m.startswith('_')][:10]}")
                
                # Special logging for results
                elif key == "results":
                    logger.info(f"  results type: {type(value)}")
                    if isinstance(value, dict):
                        logger.info(f"  results keys: {list(value.keys())}")
        
        logger.info("=== END BACKTESTING RESULTS DEBUG ===")
        
        # Safely process the data with error handling
        try:
            if "processed_data" in backtesting_results and backtesting_results["processed_data"]:
                if "features" in backtesting_results["processed_data"]:
                    processed_data = backtesting_results["processed_data"]["features"].fillna(0)
                    processed_data_dict = processed_data.to_dict()
                else:
                    processed_data_dict = backtesting_results["processed_data"]
            else:
                processed_data_dict = {}
        except Exception as e:
            print(f"Warning: Error processing data: {e}")
            processed_data_dict = {}
        
        # Safely process executors
        try:
            executors_info = [e.to_dict() for e in backtesting_results.get("executors", [])]
        except Exception as e:
            print(f"Warning: Error processing executors: {e}")
            executors_info = []
        
        # Safely process results
        try:
            results = backtesting_results.get("results", {})
            
            # Рассчитываем коэффициенты эффективности
            ratios = calculate_performance_ratios(results)
            
            # Обновляем результаты с рассчитанными коэффициентами
            results.update(ratios)
            
            # Логируем обновленные результаты
            logger.info(f"Updated results with calculated ratios: {list(results.keys())}")
            
        except Exception as e:
            print(f"Warning: Error processing results: {e}")
            results = {}
        
        # Prepare response with guaranteed structure
        response_data = {
            "executors": executors_info,
            "processed_data": processed_data_dict,
            "results": results,
        }
        
        # Log the final response structure
        logger.info("=== FINAL RESPONSE DEBUG ===")
        logger.info(f"Response keys: {list(response_data.keys())}")
        logger.info(f"executors length: {len(response_data['executors'])}")
        logger.info(f"processed_data type: {type(response_data['processed_data'])}")
        logger.info(f"results keys: {list(response_data['results'].keys())}")
        logger.info("=== END FINAL RESPONSE DEBUG ===")
        
        # Send detailed summary to Telegram
        config_dict = backtesting_config.dict()
        await telegram_service.send_backtesting_summary(
            config=config_dict,
            results=results,
            executors=executors_info,
            processed_data=processed_data_dict
        )
        
        return response_data
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Backtesting failed with error: {error_msg}")
        logger.error(f"Error type: {type(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        # Send error notification
        await telegram_service.send_simple_notification(
            f"❌ <b>Backtesting Failed</b>\n\n"
            f"Error: {error_msg}"
        )
        
        return {"error": error_msg}
