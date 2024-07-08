from dataclasses import dataclass, field
from enum import Enum
import json
import threading
import time
from typing import Any, Dict, List, Tuple
from openai import audio
import requests
from audio.audio_manager import (
    AudioManager,
    TextToSpeechResultChatTTS,
    TextToSpeechResultMeloTTS,
    TextToSpeechTask,
)
from audio.util import play_audio


class TTSServiceType(Enum):
    CHAT_TTS = 1
    MELO_TTS = 2


@dataclass(frozen=True)
class TTSService:
    audio_manager: AudioManager
    stop_event: threading.Event = field(default_factory=threading.Event)

    def run(self):
        raise NotImplemented("run not implemented")

    def convert(self):
        raise NotImplemented("convert not implemented")

    def stop(self):
        raise NotImplemented("convert not implemented")


@dataclass(frozen=True)
class TTSServiceChatTTS(TTSService):
    """
    Use ChatTTS for text to speech: https://github.com/jianchang512/ChatTTS-ui.
    Too slow!!!
    """

    url: str = "http://192.168.1.26:9966/tts"
    voice: str = "3333"
    temperature: float = 0.3
    top_p: float = 0.7
    top_k: int = 20
    skip_refine: int = 0
    custom_voice: int = 0

    def run(self):
        while not self.stop_event.is_set():
            if self.audio_manager.has_pending_text_to_audio_tasks():
                task = self.audio_manager.get_text_to_audio_task()
                if task is not None:
                    raw_response = self.convert(task)
                    self.audio_manager.save_text_to_audio_result(
                        TextToSpeechResultChatTTS(task, raw_response)
                    )

    def convert(self, task: TextToSpeechTask) -> requests.Response:
        try:
            raw_response = requests.post(
                self.url,
                data={
                    "text": task.text,
                    "prompt": "",
                    "voice": self.voice,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                    "skip_refine": self.skip_refine,
                    "custom_voice": self.custom_voice,
                },
            )
            return raw_response
        except Exception as e:
            print(f"Error converting text to audio: {e}")
            print(f"raw response: {raw_response}")
            return {"code": 1, "msg": "error", "error": e}

    def stop(self):
        self.stop_event.set()


@dataclass(frozen=True)
class TTSServiceMeloTTS(TTSService):
    """
    Use MeloTTS for text to speech: https://github.com/timhagel/MeloTTS-Docker-API-Server
    """

    url: str = "http://192.168.1.26:9966/convert/tts"
    language: str = field(default="EN")
    speaker_id: str = field(default="EN-US")

    def run(self):
        while not self.stop_event.is_set():
            if self.audio_manager.has_pending_text_to_audio_tasks():
                task = self.audio_manager.get_text_to_audio_task()
                if task is not None:
                    raw_response = self.convert(task)
                    self.audio_manager.save_text_to_audio_result(
                        TextToSpeechResultMeloTTS(task, raw_response)
                    )

    def convert(self, task: TextToSpeechTask) -> requests.Response:
        try:
            raw_response = requests.post(
                self.url,
                data=json.dumps(
                    {
                        "text": task.text,
                        "language": self.language,
                        "speaker_id": self.speaker_id,
                    }
                ),
                headers={"Content-Type": "application/json"},
            )
            return raw_response
        except Exception as e:
            print(f"Error converting text to audio: {e}")
            print(f"raw response: {raw_response}")
            return {"code": 1, "msg": "error", "error": e}

    def stop(self):
        self.stop_event.set()


def start_tts(
    audio_manager: AudioManager,
    tts_service_type: TTSServiceType = TTSServiceType.CHAT_TTS,
) -> Tuple[TTSService, threading.Thread]:
    if tts_service_type == TTSServiceType.CHAT_TTS:
        tts_service = TTSServiceChatTTS(audio_manager)
    elif tts_service_type == TTSServiceType.MELO_TTS:
        tts_service = TTSServiceMeloTTS(audio_manager)
    else:
        raise Exception(f"TTSServiceType: {tts_service_type.Name} not supported")
    thread = threading.Thread(target=tts_service.run)
    thread.start()
    return tts_service, thread


def stop_tts(tts_service: TTSService, thread: threading.Thread) -> None:
    tts_service.stop()
    thread.join()


def example_play_audio_chat_tts(
    audio_manager: AudioManager, task: TextToSpeechTask
) -> None:
    result: TextToSpeechResultChatTTS = audio_manager.get_text_to_audio_result(
        task.task_id
    )
    for url in result.file_urls:
        print(url)
        play_audio(url=url)


def example_play_audio_melo_tts(
    audio_manager: AudioManager, task: TextToSpeechTask
) -> None:
    result: TextToSpeechResultMeloTTS = audio_manager.get_text_to_audio_result(
        task.task_id
    )
    play_audio(url=None, content=result.content)


if __name__ == "__main__":
    # texts = ["Hi, my name is George! very nice to meet you", "did you eat? If not, how about some noodles?"]
    # texts = ["did you eat? If not, how about some noodles?", "Hi, my name is George! very nice to meet you"]
    # texts = ["Hi, my name is George! very nict to meet you"]
    # texts = ["did you eat? If not, how about some noodles?"]
    texts = [
        "Recording audio from a microphone using Python is tricky! Why? Because Python doesn't provide a standard library for it. Existing third-party libraries (e.g. PyAudio) are not cross-platform and have external dependencies."
    ]

    tts_service_type = TTSServiceType.MELO_TTS
    audio_manager = AudioManager(conversation_id="1")
    tts_service, thread = start_tts(
        audio_manager=audio_manager, tts_service_type=tts_service_type
    )
    for i, text in enumerate(texts):
        print(f"processing {i}th text: {text}")
        task = TextToSpeechTask(task_id=str(i), text=text)
        audio_manager.add_text_to_audio_task(task)
        while not audio_manager.has_text_to_audio_results(task=task):
            time.sleep(1)
        if tts_service_type == TTSServiceType.CHAT_TTS:
            example_play_audio_chat_tts(audio_manager=audio_manager, task=task)
        elif tts_service_type == TTSServiceType.MELO_TTS:
            example_play_audio_melo_tts(audio_manager=audio_manager, task=task)
    stop_tts(tts_service=tts_service, thread=thread)
