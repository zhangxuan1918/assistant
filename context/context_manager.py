from dataclasses import dataclass, field
from enum import Enum
import os
import time
from typing import Dict, List
import uuid

import pyaudio
from audio.audio_manager import (
    AudioManager,
    SpeechToTextTask,
    TextToSpeechTask,
)
from audio.util import play_audio, record_audio
from text.text_manager import (
    CopyFromClipboardTask,
    TextManager,
)
from llm.llm_manager import LlmGenerationTask, LlmManager, TaskStatus


class TaskType(Enum):
    AUDIO_TO_TEXT: 1
    TEXT_TO_AUDIO: 2
    LLM_GEN: 3
    COPY_FROM_CLIPBOARD: 4


@dataclass
class ContextManager:
    audio_manager: AudioManager
    text_manager: TextManager
    llm_manager: LlmManager
    # We store temporary data in temp/folder/uuid
    temp_folder: str = "/tmp/assistant"

    def __post_init__(self):
        # Create temp folder.
        self._uuid = str(uuid.uuid4())
        self._temp_folder = os.path.join(self.temp_folder, self._uuid)
        os.makedirs(self._temp_folder)
        self._conversation_turn: int = 0
        self._py_audio = pyaudio.PyAudio()

        self._audio_to_text_tasks: List[SpeechToTextTask] = []
        self._copy_from_clipboard_tasks: List[CopyFromClipboardTask] = []
        self._llm_gen_tasks: List[LlmGenerationTask] = []
        self._text_to_audio_tasks: List[List[TextToSpeechTask]] = []
        self._prompts: List[Dict] = []

    def start_conversation(self):
        self._conversation_turn += 1
        self._prompts.append({})
        # Add speech to text task. This converts the user voice input to text.
        audio_input_filepath = os.path.join(
            self._temp_folder, f"input_{self._conversation_turn}.wav"
        )

        record_audio(p=self._py_audio, filepath=audio_input_filepath)

        audio_to_text_task = SpeechToTextTask(
            task_id=self.get_task_id(TaskType.AUDIO_TO_TEXT, self._conversation_turn),
            filepath=audio_input_filepath,
        )
        self._audio_to_text_tasks.append(audio_to_text_task)
        self.audio_manager.add_audio_to_text_task(audio_to_text_task)

        # Add copy from clipboard task. This copies context from clipboard.
        copy_from_clipboard_task = CopyFromClipboardTask(
            task_id=self.get_task_id(
                TaskType.COPY_FROM_CLIPBOARD, self._conversation_turn
            )
        )
        self._copy_from_clipboard_tasks.append(copy_from_clipboard_task)
        self._prompts[-1]["context"] = self.text_manager.copy_from_clipboard(
            task=copy_from_clipboard_task
        )

        # Get speech to text results.
        while not self.audio_manager.has_audio_to_text_results(task=audio_to_text_task):
            time.sleep(1)
        self._prompts[-1]["question"] = self.audio_manager.get_audio_to_text_result(
            task_id=audio_to_text_task.task_id
        )

        self._generate_response()
        self._play_response()

    def _generate_response(self):
        llm_gen_task = LlmGenerationTask(
            task_id=self._get_task_id(TaskType.LLM_GEN), **self._prompts[-1]
        )
        self._llm_gen_tasks.append(llm_gen_task)
        self.llm_manager.add_text_gen_task(llm_gen_task)

        index, response = 0, ""
        task_status = TaskStatus.UNKNOWN
        self._text_to_audio_tasks.append([])
        while (
            task_status := self.llm_manager.get_task_status(
                task_id=llm_gen_task.task_id
            )
        ) and task_status != TaskStatus.UNKNOWN:
            response = self.llm_manager.get_text_gen_result(
                task_id=llm_gen_task.task_id, index=index
            )
            if response is not None:
                index += 1
                print(response, end="")
                self._text_to_audio_tasks[-1].append(
                    TextToSpeechTask(
                        task_id=self._get_task_id(
                            TaskType.TEXT_TO_AUDIO, self._conversation_turn, index
                        ),
                        text=response,
                    )
                )
            elif task_status == TaskStatus.FINISHED:
                break

    def _play_response(self):
        tasks = self._text_to_audio_tasks[-1]
        for task in tasks:
            while not self.audio_manager.has_text_to_audio_results(task=task):
                time.sleep(1)
            file_urls: List[str] = self.audio_manager.get_text_to_audio_result(
                task.task_id
            )
            for url in file_urls:
                play_audio(url=url)

    def _get_task_id(
        self, task_type: TaskType, turn: int, index: None | int = None
    ) -> int:
        task_id = f"TASK_{self._uuid}_{task_type.name}_{turn}"
        if index is not None:
            task_id += f"_{index}"
        return task_id

