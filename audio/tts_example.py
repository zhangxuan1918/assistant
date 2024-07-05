from audio_manager import AudioManager, TextToSpeechResult, TextToSpeechTask
from tts_service import TTSService
from util import play_audio

if __name__ == "__main__":
    text = "How to Record Audio using Python?"
    task = TextToSpeechTask(task_id="test", text=text)

    audio_manager = AudioManager(conversation_id="test_conv")
    tts_service = TTSService(audio_manager)

    print(f"run text to speech ...")
    raw_response = tts_service.convert(task=task)
    result = TextToSpeechResult(task, raw_response)
    for url in result.file_urls:
        play_audio(url=url)
