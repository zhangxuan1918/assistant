import os
from stt_service import STTService
from audio_manager import AudioManager, SpeechToTextResult, SpeechToTextTask

if __name__ == "__main__":
    filename = "audio.wav"
    folder = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(folder, filename)
    task = SpeechToTextTask(task_id="test", filepath=filepath)

    audio_manager = AudioManager(conversation_id="test_conv")
    tts_service = STTService(audio_manager)

    print(f"run speech to text ...")
    text = tts_service.convert(task=task)
    result = SpeechToTextResult(task=task, text=text)
    print(f"text: {text}")
