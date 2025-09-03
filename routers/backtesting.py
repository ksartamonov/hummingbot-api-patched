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


def calculate_performance_ratios(results: dict, executors: list | None = None) -> dict:
    """
    Корректный расчёт Sortino и Calmar.
    - returns сделки -> доли
    - частота годовая оценивается из длительности периода
    """
    try:
        net_pnl_pct = float(results.get("net_pnl_pct", 0) or 0.0)
        max_drawdown_pct = results.get("max_drawdown_pct")
        sharpe_ratio = float(results.get("sharpe_ratio", 0) or 0.0)

        # Длительность периода
        start_ts = results.get("start_time")
        end_ts = results.get("end_time")
        n_days = (end_ts - start_ts) / 86400.0 if (start_ts and end_ts and end_ts > start_ts) else None

        logger.info(f"Input data - net_pnl_pct: {net_pnl_pct}, max_drawdown_pct: {max_drawdown_pct}, n_days: {n_days}")
        print(f"🔍 DEBUG: Input data - net_pnl_pct: {net_pnl_pct}, max_drawdown_pct: {max_drawdown_pct}, n_days: {n_days}")

        # ---- Готовим ряд доходностей (по сделкам) в долях ----
        r_list = []
        if executors:
            for ex in executors:
                pnl_pct = None
                if isinstance(ex, dict):
                    pnl_pct = ex.get("pnl_pct") or ex.get("net_pnl_pct")
                    if pnl_pct is None and "net_pnl" in ex and "amount" in ex:
                        try:
                            amt = float(ex["amount"])
                            if amt != 0:
                                pnl_pct = float(ex["net_pnl"]) / amt * 100.0
                        except Exception:
                            pass
                else:
                    # объект с атрибутами
                    if hasattr(ex, "pnl_pct"):
                        pnl_pct = ex.pnl_pct
                    elif hasattr(ex, "net_pnl") and hasattr(ex, "amount"):
                        try:
                            amt = float(ex.amount)
                            if amt != 0:
                                pnl_pct = float(ex.net_pnl) / amt * 100.0
                        except Exception:
                            pass

                if pnl_pct is not None:
                    r_list.append(float(pnl_pct) / 100.0)  # в долях

        r = np.asarray(r_list, dtype=float)
        print(f"🔍 DEBUG: Prepared returns array: {len(r)} returns, sample: {r[:5] if len(r) > 0 else 'empty'}")

        # ---- Оцениваем годовую частоту ----
        # Если знаем длительность в днях — оценим частоту как (кол-во наблюдений в день) * 365
        if n_days and n_days > 0 and len(r) > 1:
            periods_per_year = (len(r) / n_days) * 365.0
        else:
            # разумный дефолт, если ничего не знаем
            periods_per_year = 252.0
        
        print(f"🔍 DEBUG: Estimated periods_per_year: {periods_per_year:.2f}")

        # ---- Sortino ----
        if len(r) > 1:
            target = 0.0
            downside = np.minimum(r - target, 0.0)
            dd = float(np.sqrt(np.mean(np.square(downside))))
            mu_excess = float(np.mean(r - target))
            if dd > 0:
                sortino_ratio = (mu_excess / dd) * np.sqrt(periods_per_year)
                print(f"🔍 DEBUG: Sortino calculation - mu_excess: {mu_excess:.6f}, dd: {dd:.6f}, periods_per_year: {periods_per_year:.2f}, Sortino: {sortino_ratio:.6f}")
            else:
                sortino_ratio = np.inf if mu_excess > 0 else 0.0
                print(f"🔍 DEBUG: Sortino - no downside deviation, setting to {'inf' if mu_excess > 0 else '0'}")
        else:
            sortino_ratio = 0.0  # данных мало — вернём 0
            print(f"🔍 DEBUG: Sortino - insufficient data ({len(r)} returns), setting to 0")

        # ---- Calmar ----
        # CAGR
        if n_days and n_days > 0:
            total_return = float(net_pnl_pct) / 100.0
            cagr = (1.0 + total_return) ** (365.0 / n_days) - 1.0
            print(f"🔍 DEBUG: Calmar CAGR calculation - total_return: {total_return:.6f}, n_days: {n_days:.2f}, CAGR: {cagr:.6f}")
        elif len(r) > 0:
            total_return = float(np.prod(1.0 + r) - 1.0)
            # аппроксимация: приводим к годовой через эффективную частоту
            cagr = (1.0 + total_return) ** (periods_per_year / max(len(r), 1)) - 1.0
            print(f"🔍 DEBUG: Calmar CAGR from returns - total_return: {total_return:.6f}, cagr: {cagr:.6f}")
        else:
            cagr = 0.0
            print(f"🔍 DEBUG: Calmar CAGR - no data, setting to 0")

        # MaxDD: берём из results, если нет — считаем по кумулятивной доходности
        if max_drawdown_pct is not None:
            max_dd = abs(float(max_drawdown_pct)) / 100.0
            print(f"🔍 DEBUG: Calmar MaxDD from results: {max_dd:.6f}")
        elif len(r) > 1:
            eq = np.cumprod(1.0 + r)
            peaks = np.maximum.accumulate(eq)
            dd_path = eq / peaks - 1.0
            max_dd = -float(np.min(dd_path))  # положительное число
            print(f"🔍 DEBUG: Calmar MaxDD calculated from equity curve: {max_dd:.6f}")
        else:
            max_dd = 0.0
            print(f"🔍 DEBUG: Calmar MaxDD - no data, setting to 0")

        if max_dd > 0:
            calmar_ratio = cagr / max_dd
            print(f"🔍 DEBUG: Calmar ratio calculation - CAGR: {cagr:.6f}, MaxDD: {max_dd:.6f}, Calmar: {calmar_ratio:.6f}")
        else:
            calmar_ratio = np.inf if cagr > 0 else 0.0
            print(f"🔍 DEBUG: Calmar ratio - MaxDD is 0, setting to {'inf' if cagr > 0 else '0'}")

        # Возвращаем без «обнуления» inf — так честнее для аналитики
        result = {
            "sharpe_ratio": sharpe_ratio if np.isfinite(sharpe_ratio) else 0.0,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
        }
        
        print(f"🔍 DEBUG: Final ratios - Sharpe: {result['sharpe_ratio']:.6f}, Sortino: {result['sortino_ratio']}, Calmar: {result['calmar_ratio']}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error calculating performance ratios: {e}")
        print(f"🔍 DEBUG: Error calculating ratios: {e}")
        return {
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
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
            ratios = calculate_performance_ratios(results, executors_info)
            
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
