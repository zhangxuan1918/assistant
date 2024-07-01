from asyncio import Task
from dataclasses import dataclass, field
from enum import Enum
from queue import Queue
import stat
from typing import Dict, List

@dataclass(frozen=True)
class LlmGenerationTask:
    task_id: str
    context: str
    question: str

@dataclass(frozen=True)
class LlmGenerationResult:
    task: LlmGenerationTask
    response: str

class TaskStatus(Enum):
    PENDING = 1
    RUNNING = 2
    FINISHED = 3
    UNKNOWN = 4

@dataclass
class LlmManager:
    # Unique id for a conversation.
    conversation_id: str
    # Buffer for text generation tasks. We pop up the front element to process.
    text_gen_tasks: Queue[LlmGenerationTask] = field(default_factory=Queue)
    # Buffer for generated responsese. We use id to consume the response.
    # { task_id: [LlmGenerationResult] ...}
    text_gen_results: Dict[str, List[LlmGenerationResult]] = field(default_factory=dict)
    # Buffer for task status.
    text_gen_tasks_status: Dict[str, TaskStatus] = field(default_factory=dict)

    def add_text_gen_task(self, task: LlmGenerationTask) -> None:
        self.text_gen_tasks.put(task)
        self.text_gen_results[task.task_id] = []
        self.set_task_status(task_id=task.task_id, status=TaskStatus.PENDING)
    
    def get_text_gen_task(self) -> None | LlmGenerationTask:
        if self.has_pending_text_gen_tasks():
            return self.text_gen_tasks.get()
        else:
            return None
    
    def save_text_gen_task(self, result: LlmGenerationResult) -> None:
        self.text_gen_results[result.task.task_id].append(result)
    
    def has_pending_text_gen_tasks(self) -> bool:
        return self.num_pending_text_gen_tasks() > 0
    
    def num_pending_text_gen_tasks(self) -> int:
        return self.text_gen_tasks.qsize()

    def has_text_gen_result(self, task: LlmGenerationTask) -> bool:
        return task.task_id in self.text_gen_results
    
    def num_text_gen_results(self) -> int:
        return len(self.text_gen_results)

    def get_text_gen_result(self, task_id: str, index: int) -> None | str:
        if task_id in self.text_gen_results and len(self.text_gen_results[task_id]) > index:
            return self.text_gen_results[task_id][index].response
        else:
            return None
    
    def set_task_status(self, task_id: str, status: TaskStatus) -> None:
        self.text_gen_tasks_status[task_id] = status
    
    def get_task_status(self, task_id: str) -> TaskStatus:
        if task_id in self.text_gen_tasks_status:
            return self.text_gen_tasks_status[task_id]
        return TaskStatus.UNKNOWN
    
    def clean_up_text_gen_task(self, result: LlmGenerationResult) -> None:
        # Delete responses.
        task_id = result.task.task_id
        if task_id in self.text_gen_results:
            del self.text_gen_results