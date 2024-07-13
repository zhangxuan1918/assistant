from assistant.audio.stt_service import STTService
from assistant.audio.audio_manager import AudioManager, SpeechToTextTask

import os


if __name__ == "__main__":
    filename = "./audio.wav"
    folder = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(folder, filename)
    with open(filepath, "rb") as wav_file:
        audio_data = wav_file.read()

    audio_manager = AudioManager()
    tts_service = STTService(audio_manager)
    task = SpeechToTextTask(task_id="test", audio_data=audio_data)
    print(f"run speech to text ...")
    text = tts_service.convert(task=task)
    print(f"text: {text}")
