
from dataclasses import dataclass, field
from queue import Queue
from typing import Dict, List, Tuple

@dataclass(frozen=True)
class SpeechToTextTask:
    task_id: str
    filepath: str

@dataclass(frozen=True)
class SpeechToTextResult:
    task: SpeechToTextTask
    text: str

@dataclass(frozen=True)
class AudioManager:
    # Unique id for a conversation.
    conversation_id: str
    # Buffer for audio to be processed. We pop up the front element to process.
    audio_to_process: Queue[SpeechToTextTask] = field(default=lambda: Queue())
    # Buffer for audio processed. We use id to consume the text. Afterwards, we remove it.
    #   {id: SpeechToTextTask, ...}
    audio_processed: Dict[str, SpeechToTextResult] = field(default=lambda: {})

    def add_task(self, task: SpeechToTextTask) -> None:
        self.audio_to_process.put(task)

    def get_task(self) -> None | SpeechToTextResult:
        if len(self.audio_to_process) == 0:
            return None
        return self.audio_to_process.get()

    def save_result(self, result: SpeechToTextResult) -> None:
        self.audio_processed[result.id] = result
    
    def has_pending_tasks(self) -> bool:
        return self.num_pending_tasks() > 0
    
    def num_pending_tasks(self) -> int:
        return len(self.audio_to_process)
    
    def has_results(self) -> bool:
        return self.num_results > 0
    
    def num_results(self) -> int:
        return len(self.audio_processed)

    