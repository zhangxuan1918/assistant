from dataclasses import dataclass
from enum import Enum
import json
import os
import threading
import time
from typing import Any, Dict, List, Tuple
import uuid

import pyaudio
from audio.stt_service import STTService
from audio.tts_service import TTSService
from audio.util import play_audio, record_audio
from audio.audio_manager import (
    AudioManager,
    SpeechToTextTask,
    TextToSpeechTask,
)
from llm.llm_service import LLMService
from text.text_manager import (
    CopyFromClipboardTask,
    TextManager,
)
from llm.llm_manager import LlmGenerationTask, LlmManager, TaskStatus


class TaskType(Enum):
    AUDIO_TO_TEXT = 1
    TEXT_TO_AUDIO = 2
    LLM_GEN = 3
    COPY_FROM_CLIPBOARD = 4


@dataclass
class ContextManager:
    # We store temporary data in temp/folder/uuid
    temp_folder: str = "/tmp/assistant"

    def __post_init__(self):
        self._conversation_id = str(uuid.uuid4())
        self.audio_manager = AudioManager(conversation_id=self._conversation_id)
        self.text_manager = TextManager(conversation_id=self._conversation_id)
        self.llm_manager = LlmManager(conversation_id=self._conversation_id)

        # Create temp folder.
        self._temp_folder = os.path.join(self.temp_folder, self._conversation_id)
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

        print(
            "================================================================================="
            f"INFO: start conversation: id={self._conversation_id}, turn={self._conversation_turn} ..."
            "================================================================================="
        )

        self._prompts.append({})
        # Add speech to text task. This converts the user voice input to text.
        audio_input_filepath = os.path.join(
            self._temp_folder, f"input_{self._conversation_turn}.wav"
        )
        print("INFO: recording user audio input ...")
        record_audio(p=self._py_audio, filepath=audio_input_filepath)

        audio_to_text_task = SpeechToTextTask(
            task_id=self._get_task_id(TaskType.AUDIO_TO_TEXT, self._conversation_turn),
            filepath=audio_input_filepath,
        )
        self._audio_to_text_tasks.append(audio_to_text_task)
        print("INFO: adding speech to text task ...")
        self.audio_manager.add_audio_to_text_task(audio_to_text_task)

        # Add copy from clipboard task. This copies context from clipboard.
        print("INFO: adding copy from clipboard task ...")
        copy_from_clipboard_task = CopyFromClipboardTask(
            task_id=self._get_task_id(
                TaskType.COPY_FROM_CLIPBOARD, self._conversation_turn
            )
        )
        self._copy_from_clipboard_tasks.append(copy_from_clipboard_task)
        clipboard_text = self.text_manager.copy_from_clipboard(
            task=copy_from_clipboard_task
        )
        self._prompts[-1]["context"] = clipboard_text
        print(f"INFO: context: {clipboard_text[:50]}")

        # Get speech to text results.
        while not self.audio_manager.has_audio_to_text_results(task=audio_to_text_task):
            time.sleep(1)
        user_question = self.audio_manager.get_audio_to_text_result(
            task_id=audio_to_text_task.task_id
        )
        self._prompts[-1]["question"] = user_question
        print(f"INFO: user question: {user_question}")

        self._generate_response()
        self._play_response()

    def _generate_response(self):
        print("Info: generate response from llm ...")
        llm_gen_task = LlmGenerationTask(
            task_id=self._get_task_id(TaskType.LLM_GEN, self._conversation_turn),
            **self._prompts[-1],
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
                print(response)
                text_speech_task = TextToSpeechTask(
                    task_id=self._get_task_id(
                        TaskType.TEXT_TO_AUDIO, self._conversation_turn, index
                    ),
                    text=response,
                )
                self._text_to_audio_tasks[-1].append(text_speech_task)
                self.audio_manager.add_text_to_audio_task(task=text_speech_task)
            elif task_status == TaskStatus.FINISHED:
                break

    def _play_response(self):
        tasks = self._text_to_audio_tasks[-1]
        print(f"Info: add text to speech tasks: #{len(tasks)} ...")
        for i, task in enumerate(tasks):
            print(f"Info: wait for result from task {i} ...")
            while not self.audio_manager.has_text_to_audio_results(task=task):
                time.sleep(1)
            file_urls: List[str] = self.audio_manager.get_text_to_audio_result(
                task.task_id
            )
            print(f"Info: total #{len(file_urls)} generated")
            for j, url in enumerate(file_urls):
                print(f"Info: play audio file {j}th, url: {url}...")
                play_audio(url=url)

    def _get_task_id(
        self, task_type: TaskType, turn: int, index: None | int = None
    ) -> int:
        task_id = f"TASK_{self._conversation_id}_{task_type.name}_{turn}"
        if index is not None:
            task_id += f"_{index}"
        return task_id
    
    def clear(self) -> None:
        # Clean up resources used.
        # Close pyaudio.
        self._py_audio.terminate()

        # Delete temporary folder.
        if os.path.exists(self._temp_folder):
            os.removedirs(self._temp_folder)


def start_services(
    context_manager: ContextManager,
) -> List[Tuple[Any, threading.Thread]]:
    stt_service = STTService(context_manager.audio_manager)
    stt_thread = threading.Thread(target=stt_service.run)
    stt_thread.start()

    tts_service = TTSService(context_manager.audio_manager)
    tts_thread = threading.Thread(target=tts_service.run)
    tts_thread.start()

    llm_service = LLMService(context_manager.llm_manager)
    llm_thread = threading.Thread(target=llm_service.run)
    llm_thread.start()

    return [
        (stt_service, stt_thread),
        (tts_service, tts_thread),
        (llm_service, llm_thread),
    ]


def stop_services(services: List[Tuple[Any, threading.Thread]]) -> None:
    for service, thread in services:
        service.stop()
        thread.join()


def stop_stt(stt_service: STTService, thread: threading.Thread) -> None:
    stt_service.stop()
    thread.join()


if __name__ == "__main__":
    context_manager = ContextManager()
    services = start_services(context_manager=context_manager)
    context_manager.start_conversation()
    stop_services(services=services)
