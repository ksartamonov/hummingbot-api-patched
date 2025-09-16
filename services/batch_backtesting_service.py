import asyncio
import logging
import uuid
from typing import Dict, List, Optional
from datetime import datetime

from hummingbot.strategy_v2.backtesting.backtesting_engine_base import BacktestingEngineBase
from models.backtesting import BatchBacktestingConfig, BatchBacktestingResult
from config import settings

logger = logging.getLogger(__name__)


class BatchBacktestingService:
    """Сервис для управления пакетным выполнением бектестов"""
    
    def __init__(self):
        self.active_tasks: Dict[str, BatchBacktestingResult] = {}
        self._semaphore = asyncio.Semaphore(5)  # Максимум 5 одновременных бектестов по умолчанию
        self.backtesting_engine = BacktestingEngineBase()
    
    async def start_batch_backtesting(self, config: BatchBacktestingConfig) -> str:
        """Запускает пакетный бектест и возвращает task_id"""
        task_id = str(uuid.uuid4())
        
        # Создаем результат задачи
        result = BatchBacktestingResult(
            task_id=task_id,
            status="pending",
            total_configs=len(config.configs),
            completed_configs=0,
            failed_configs=0,
            results=[],
            errors=[]
        )
        
        self.active_tasks[task_id] = result
        
        # Запускаем задачу в фоне
        asyncio.create_task(self._execute_batch_backtesting(task_id, config))
        
        return task_id
    
    async def _execute_batch_backtesting(self, task_id: str, config: BatchBacktestingConfig):
        """Выполняет пакетный бектест"""
        result = self.active_tasks[task_id]
        result.status = "running"
        
        # Создаем семафор для ограничения количества одновременных бектестов
        semaphore = asyncio.Semaphore(config.max_concurrent or 5)
        
        # Создаем задачи для всех конфигураций
        tasks = []
        for i, config_item in enumerate(config.configs):
            task = asyncio.create_task(
                self._execute_single_backtesting_with_semaphore(
                    semaphore, task_id, i, config_item, config
                )
            )
            tasks.append(task)
        
        # Ждем завершения всех задач
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Обновляем статус
        if result.failed_configs == 0:
            result.status = "completed"
        elif result.completed_configs > 0:
            result.status = "completed"  # Частично успешно
        else:
            result.status = "failed"
        
        result.progress_percentage = 100.0
        logger.info(f"Batch backtesting {task_id} completed. "
                   f"Completed: {result.completed_configs}, Failed: {result.failed_configs}")
    
    async def _execute_single_backtesting_with_semaphore(
        self, 
        semaphore: asyncio.Semaphore, 
        task_id: str, 
        config_index: int, 
        config_item, 
        batch_config: BatchBacktestingConfig
    ):
        """Выполняет один бектест с ограничением семафора"""
        async with semaphore:
            await self._execute_single_backtesting(task_id, config_index, config_item, batch_config)
    
    async def _execute_single_backtesting(
        self, 
        task_id: str, 
        config_index: int, 
        config_item, 
        batch_config: BatchBacktestingConfig
    ):
        """Выполняет один бектест"""
        result = self.active_tasks[task_id]
        
        try:
            # Очищаем конфигурацию
            if isinstance(config_item, dict):
                cleaned_config = self._clean_config_data(config_item)
            else:
                cleaned_config = config_item
            
            # Создаем конфигурацию контроллера
            if isinstance(cleaned_config, str):
                controller_config = self.backtesting_engine.get_controller_config_instance_from_yml(
                    config_path=cleaned_config,
                    controllers_conf_dir_path=settings.app.controllers_path,
                    controllers_module=settings.app.controllers_module
                )
            else:
                controller_config = self.backtesting_engine.get_controller_config_instance_from_dict(
                    config_data=cleaned_config,
                    controllers_module=settings.app.controllers_module
                )
            
            # Запускаем бектест
            backtesting_results = await self.backtesting_engine.run_backtesting(
                controller_config=controller_config,
                trade_cost=batch_config.trade_cost,
                start=int(batch_config.start_time),
                end=int(batch_config.end_time),
                backtesting_resolution=batch_config.backtesting_resolution
            )
            
            # Обрабатываем результаты
            if backtesting_results and isinstance(backtesting_results, dict):
                processed_result = {
                    "config_index": config_index,
                    "config": config_item,
                    "results": backtesting_results,
                    "timestamp": datetime.now().isoformat()
                }
                result.results.append(processed_result)
                result.completed_configs += 1
            else:
                error_info = {
                    "config_index": config_index,
                    "config": config_item,
                    "error": "Invalid backtesting results",
                    "timestamp": datetime.now().isoformat()
                }
                result.errors.append(error_info)
                result.failed_configs += 1
            
        except Exception as e:
            logger.error(f"Error in backtesting config {config_index}: {str(e)}")
            error_info = {
                "config_index": config_index,
                "config": config_item,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            result.errors.append(error_info)
            result.failed_configs += 1
        
        # Обновляем прогресс
        total_processed = result.completed_configs + result.failed_configs
        result.progress_percentage = (total_processed / result.total_configs) * 100.0
    
    def _clean_config_data(self, config_data: dict) -> dict:
        """Очищает данные конфигурации от лишних пробелов"""
        cleaned_config = {}
        for key, value in config_data.items():
            if isinstance(value, str):
                cleaned_config[key] = value.strip()
            else:
                cleaned_config[key] = value
        return cleaned_config
    
    def get_task_status(self, task_id: str) -> Optional[BatchBacktestingResult]:
        """Получает статус задачи"""
        return self.active_tasks.get(task_id)
    
    def get_all_tasks(self) -> Dict[str, BatchBacktestingResult]:
        """Получает все активные задачи"""
        return self.active_tasks.copy()
    
    def cleanup_completed_tasks(self, max_age_hours: int = 24):
        """Очищает завершенные задачи старше указанного времени"""
        current_time = datetime.now()
        tasks_to_remove = []
        
        for task_id, result in self.active_tasks.items():
            if result.status in ["completed", "failed"]:
                # Проверяем возраст задачи (предполагаем, что timestamp есть в результатах)
                if result.results:
                    try:
                        last_timestamp = result.results[-1].get("timestamp")
                        if last_timestamp:
                            last_time = datetime.fromisoformat(last_timestamp)
                            age_hours = (current_time - last_time).total_seconds() / 3600
                            if age_hours > max_age_hours:
                                tasks_to_remove.append(task_id)
                    except Exception:
                        # Если не можем определить время, удаляем задачу
                        tasks_to_remove.append(task_id)
        
        for task_id in tasks_to_remove:
            del self.active_tasks[task_id]
            logger.info(f"Cleaned up completed task: {task_id}")


# Глобальный экземпляр сервиса
batch_backtesting_service = BatchBacktestingService()
