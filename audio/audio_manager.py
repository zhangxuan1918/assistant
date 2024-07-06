from dataclasses import dataclass, field
import os
from queue import Queue
from typing import Any, Dict, List


@dataclass(frozen=True)
class SpeechToTextTask:
    task_id: str
    filepath: str


@dataclass(frozen=True)
class SpeechToTextResult:
    task: SpeechToTextTask
    text: str


@dataclass(frozen=True)
class TextToSpeechTask:
    task_id: str
    text: str


@dataclass
class TextToSpeechResult:
    task: TextToSpeechTask
    raw_response: Dict[Any, Any]
    file_paths: List[str] = field(default_factory=list)
    file_urls: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.raw_response["msg"] == "ok":
            self.file_paths, self.file_urls = [
                res["filename"] for res in self.raw_response["audio_files"]
            ], [res["url"] for res in self.raw_response["audio_files"]]


@dataclass
class AudioManager:
    # Unique id for a conversation.
    conversation_id: str
    # Buffer for audio to be processed. We pop up the front element to process.
    audio_to_text_tasks: Queue[SpeechToTextTask] = field(default_factory=Queue)
    # Buffer for audio processed. We use id to consume the text. Afterwards, we remove it.
    #   {id: SpeechToTextTask, ...}
    audio_to_text_results: Dict[str, SpeechToTextResult] = field(default_factory=dict)
    # Buffer for text to be processed. We pop up the front element to process.
    text_to_audio_tasks: Queue[TextToSpeechTask] = field(default_factory=Queue)
    # Buffer for text processed. We use id to consume the text. Afterwards, we remove it.
    text_to_audio_results: Dict[str, TextToSpeechResult] = field(default_factory=dict)

    def add_audio_to_text_task(self, task: SpeechToTextTask) -> None:
        self.audio_to_text_tasks.put(task)

    def get_audio_to_text_task(self) -> None | SpeechToTextResult:
        if self.has_pending_audio_to_text_tasks():
            return self.audio_to_text_tasks.get()
        else:
            return None

    def save_audio_to_text_result(self, result: SpeechToTextResult) -> None:
        self.audio_to_text_results[result.task.task_id] = result

    def has_pending_audio_to_text_tasks(self) -> bool:
        return self.num_pending_audio_to_text_tasks() > 0

    def num_pending_audio_to_text_tasks(self) -> int:
        return self.audio_to_text_tasks.qsize()

    def has_audio_to_text_results(self, task: SpeechToTextTask) -> bool:
        return task.task_id in self.audio_to_text_results

    def num_audio_to_text_results(self) -> int:
        return len(self.audio_to_text_results)

    def get_audio_to_text_result(self, task_id) -> str | None:
        if task_id in self.audio_to_text_results:
            return self.audio_to_text_results[task_id].text

    def clean_up_audio_to_text_task(self, result: SpeechToTextResult) -> None:
        # Delete audio file.
        filepath = result.task.filepath
        if os.path.exists(filepath):
            os.remove(filepath)
        task_id = result.task.task_id
        if task_id in self.audio_to_text_results:
            del self.audio_to_text_results[task_id]

    def add_text_to_audio_task(self, task: TextToSpeechTask) -> None:
        self.text_to_audio_tasks.put(task)

    def get_text_to_audio_task(self) -> None | TextToSpeechTask:
        if self.has_pending_text_to_audio_tasks():
            return self.text_to_audio_tasks.get()
        else:
            return

    def save_text_to_audio_result(self, result: TextToSpeechResult) -> None:
        self.text_to_audio_results[result.task.task_id] = result

    def has_pending_text_to_audio_tasks(self) -> bool:
        return self.num_pending_text_to_audio_tasks() > 0

    def num_pending_text_to_audio_tasks(self) -> int:
        return self.text_to_audio_tasks.qsize()

    def has_text_to_audio_results(self, task: TextToSpeechTask) -> bool:
        return task.task_id in self.text_to_audio_results

    def num_text_to_audio_results(self) -> int:
        return len(self.text_to_audio_results)

    def get_text_to_audio_result(self, task_id) -> List[str]:
        if task_id in self.text_to_audio_results:
            return self.text_to_audio_results[task_id].file_urls
        else:
            return []

    def clean_up_text_to_audio_task(self, result: TextToSpeechResult) -> None:
        # Delete audio file.
        # TODO: how to clean up generated audio file in another docker container?
        task_id = result.task.task_id
        if task_id in self.text_to_audio_results:
            del self.text_to_audio_results[task_id]
