import logging
import numpy as np
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def calculate_performance_ratios(results: dict, executors: list | None = None) -> dict:
    """
    Корректный расчёт Sortino и Calmar.
    - returns сделки -> доли
    - частота годовая оценивается из длительности периода
    """
    try:
        net_pnl_pct = float(results.get("net_pnl_pct", 0) or 0.0)
        
        # Получаем информацию о времени из результатов
        start_ts = results.get("start_time")
        end_ts = results.get("end_time")
        n_days = 0.0
        
        if start_ts and end_ts and end_ts > start_ts:
            n_days = (end_ts - start_ts) / 86400.0  # конвертируем секунды в дни
        else:
            # Пытаемся получить n_days напрямую
            n_days = float(results.get("n_days", 0) or 0.0)
        
        print(f"🔍 DEBUG: Starting performance ratios calculation")
        print(f"🔍 DEBUG: net_pnl_pct: {net_pnl_pct}, n_days: {n_days}, start_ts: {start_ts}, end_ts: {end_ts}")

        # ---- Sharpe Ratio ----
        # Простой расчет на основе общего PnL
        if n_days and n_days > 0:
            # Годовая доходность
            annual_return = (net_pnl_pct / 100.0) * (365.0 / n_days)
            # Предполагаем волатильность 1% в день (стандартное значение)
            daily_volatility = 0.01
            annual_volatility = daily_volatility * np.sqrt(365.0)
            sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0.0
            print(f"🔍 DEBUG: Sharpe calculation - annual_return: {annual_return:.6f}, annual_volatility: {annual_volatility:.6f}, Sharpe: {sharpe_ratio:.6f}")
        else:
            sharpe_ratio = 0.0
            print(f"🔍 DEBUG: Sharpe - no time data, setting to 0")

        # ---- Подготовка returns для Sortino и Calmar ----
        r_list = []
        
        # Если есть executors, извлекаем returns из них
        if executors and len(executors) > 0:
            for executor in executors:
                pnl_pct = None
                
                # Проверяем, является ли executor словарем или объектом
                if isinstance(executor, dict):
                    pnl_pct = executor.get("net_pnl_pct", 0)
                else:
                    # Это объект ExecutorInfo, используем атрибуты
                    pnl_pct = getattr(executor, "net_pnl_pct", 0)
                    # Если это Decimal, конвертируем в float
                    if hasattr(pnl_pct, '__float__'):
                        pnl_pct = float(pnl_pct)
                
                if pnl_pct is not None and pnl_pct != 0:
                    r_list.append(float(pnl_pct) / 100.0)  # в долях
        else:
            # Если нет executors, используем общий PnL
            if net_pnl_pct != 0:
                r_list.append(float(net_pnl_pct) / 100.0)  # в долях

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

        # Max Drawdown
        max_dd = 0.0
        if len(r) > 1:
            cumulative = np.cumprod(1.0 + r)
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = (cumulative - running_max) / running_max
            max_dd = float(np.min(drawdowns))
            print(f"🔍 DEBUG: Calmar MaxDD calculation - max_dd: {max_dd:.6f}")
        elif len(r) == 1 and r[0] < 0:
            max_dd = float(r[0])
            print(f"🔍 DEBUG: Calmar MaxDD from single negative return: {max_dd:.6f}")
        else:
            # Если нет данных о просадках, используем консервативную оценку
            max_dd = -0.01  # 1% минимальная просадка
            print(f"🔍 DEBUG: Calmar MaxDD - no data, using conservative estimate: {max_dd:.6f}")

        # Calmar Ratio
        if max_dd < 0:
            calmar_ratio = cagr / abs(max_dd)
            print(f"🔍 DEBUG: Calmar ratio calculation - cagr: {cagr:.6f}, max_dd: {max_dd:.6f}, Calmar: {calmar_ratio:.6f}")
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
