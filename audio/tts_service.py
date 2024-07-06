from dataclasses import dataclass, field
import threading
import time
from typing import Any, Dict, List, Tuple
from openai import audio
import requests
from audio.audio_manager import AudioManager, TextToSpeechTask, TextToSpeechResult
from audio.util import play_audio


@dataclass(frozen=True)
class TTSService:
    audio_manager: AudioManager
    url: str = "http://192.168.1.26:9966/tts"
    voice: str = "3333"
    temperature: float = 0.3
    top_p: float = 0.7
    top_k: int = 20
    skip_refine: int = 0
    custom_voice: int = 0
    stop_event: threading.Event = field(default_factory=threading.Event)

    def run(self):
        while not self.stop_event.is_set():
            if self.audio_manager.has_pending_text_to_audio_tasks():
                task = self.audio_manager.get_text_to_audio_task()
                if task is not None:
                    raw_response = self.convert(task)
                    self.audio_manager.save_text_to_audio_result(
                        TextToSpeechResult(task, raw_response)
                    )

    def convert(self, task: TextToSpeechTask) -> Dict[Any, Any]:
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
            return raw_response.json()
        except Exception as e:
            print(f"Error converting text to audio: {e}")
            print(f"raw response: {raw_response}")
            return {"code": 1, "msg": "error", "error": e}

    def stop(self):
        self.stop_event.set()


def start_tts(audio_manager: AudioManager) -> Tuple[TTSService, threading.Thread]:
    tts_service = TTSService(audio_manager)
    thread = threading.Thread(target=tts_service.run)
    thread.start()
    return tts_service, thread


def stop_tts(tts_service: TTSService, thread: threading.Thread) -> None:
    tts_service.stop()
    thread.join()


if __name__ == "__main__":
    # texts = ["Hi, my name is George! very nice to meet you", "did you eat? If not, how about some noodles?"]
    # texts = ["did you eat? If not, how about some noodles?", "Hi, my name is George! very nice to meet you"]
    # texts = ["Hi, my name is George! very nict to meet you"]
    # texts = ["did you eat? If not, how about some noodles?"]
    texts = ["Recording audio from a microphone using Python is tricky! Why? Because Python doesn't provide a standard library for it. Existing third-party libraries (e.g. PyAudio) are not cross-platform and have external dependencies."]

    audio_manager = AudioManager(conversation_id="1")
    tts_service, thread = start_tts(audio_manager=audio_manager)
    for i, text in enumerate(texts):
        print(f"processing {i}th text: {text}")
        task = TextToSpeechTask(task_id=str(i), text=text)
        audio_manager.add_text_to_audio_task(task)
        while not audio_manager.has_text_to_audio_results(task=task):
            time.sleep(1)
        file_urls: List[str] = audio_manager.get_text_to_audio_result(task.task_id)
        for url in file_urls:
            print(url)
            play_audio(url=url)
    stop_tts(tts_service=tts_service, thread=thread)
