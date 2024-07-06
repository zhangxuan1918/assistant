from dataclasses import dataclass, field
import os
import threading
from math import e
import time
from typing import Tuple
from openai import OpenAI
from audio.audio_manager import AudioManager, SpeechToTextResult, SpeechToTextTask


@dataclass(frozen=True)
class STTService:
    audio_manager: AudioManager
    url: str = "http://192.168.1.26:8000/v1"
    model_name: str = "Systran/faster-distil-whisper-large-v3"
    stop_event: threading.Event = field(default_factory=threading.Event)
    client: OpenAI = OpenAI(api_key="dummy key", base_url=url)

    def run(self):
        while not self.stop_event.is_set():
            if self.audio_manager.has_pending_audio_to_text_tasks():
                task = self.audio_manager.get_audio_to_text_task()
                if task is not None and (text := self.convert(task)) is not None:
                    self.audio_manager.save_audio_to_text_result(
                        SpeechToTextResult(task, text)
                    )

    def convert(self, task: SpeechToTextTask) -> str | None:
        try:
            audio_file = open(task.filepath, "rb")
            transcript = self.client.audio.transcriptions.create(
                model=self.model_name, file=audio_file
            )
            return transcript.text
        except Exception as e:
            print(f"Error converting audio to text: {e}")
            return None
        
    def stop(self):
        self.stop_event.set()

def start_stt(audio_manager: AudioManager) -> Tuple[STTService, threading.Thread]:
    stt_service = STTService(audio_manager)
    thread = threading.Thread(target=stt_service.run)
    thread.start()
    return stt_service, thread


def stop_stt(stt_service: STTService, thread: threading.Thread) -> None:
    stt_service.stop()
    thread.join()


if __name__ == "__main__":
    filenames = ["audio.wav", "audio2.wav"]
    folder = os.path.dirname(os.path.abspath(__file__))
    
    audio_manager = AudioManager(conversation_id="2")
    stt_service, thread = start_stt(audio_manager=audio_manager)
    for i, filename in enumerate(filenames):
        print(f"processing {i}th audio")
        task = SpeechToTextTask(task_id=str(i), filepath=os.path.join(folder, filename))
        audio_manager.add_audio_to_text_task(task)
        print(f"has pending speech to text task: {audio_manager.has_pending_audio_to_text_tasks()}")
        while not audio_manager.has_audio_to_text_results(task=task):
            time.sleep(1)
        text = audio_manager.get_audio_to_text_result(task_id=task.task_id)
        print(f"speech to text result: {text}")
    stop_stt(stt_service=stt_service, thread=thread)
