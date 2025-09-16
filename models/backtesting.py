from typing import Dict, Union, List, Optional
from pydantic import BaseModel


class BacktestingConfig(BaseModel):
    start_time: int = 1735689600  # 2025-01-01 00:00:00
    end_time: int = 1738368000  # 2025-02-01 00:00:00
    backtesting_resolution: str = "1m"
    trade_cost: float = 0.0006
    config: Union[Dict, str]


class BatchBacktestingConfig(BaseModel):
    start_time: int = 1735689600  # 2025-01-01 00:00:00
    end_time: int = 1738368000  # 2025-02-01 00:00:00
    backtesting_resolution: str = "1m"
    trade_cost: float = 0.0006
    configs: List[Union[Dict, str]]
    max_concurrent: Optional[int] = 5  # Максимальное количество одновременных бектестов


class BatchBacktestingResult(BaseModel):
    task_id: str
    status: str  # "pending", "running", "completed", "failed"
    total_configs: int
    completed_configs: int
    failed_configs: int
    results: List[Dict] = []
    errors: List[Dict] = []
    progress_percentage: float = 0.0