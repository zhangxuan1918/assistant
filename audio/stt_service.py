from dataclasses import dataclass
import time
from openai import OpenAI
from audio.audio_manager import AudioManager, SpeechToTextResult, SpeechToTextTask

@dataclass(frozen=True)
class STTService:
    audio_manager: AudioManager
    url: str = "http://192.168.1.26:8000/v1"
    model_name: str = "Systran/faster-distil-whisper-large-v3"

    def __post_init__(self):
        self.client: OpenAI = OpenAI(api_key="dummy key", base_url=self.url)

    def main_loop(self):
        while True:
            if self.audio_manager.has_pending_tasks:
                task = self.audio_manager.get_task()
                text = self.send_request(task)
                self.audio_manager.save_result(SpeechToTextResult(task, text))
            else:
                time.sleep(10)

    def convert(self, task: SpeechToTextTask) -> str:
        audio_file = open(task.filepath, "rb")
        transcript = self.client.audio.transcriptions.create(
            model=self.model_name, file=audio_file
        )
        return transcript.text

