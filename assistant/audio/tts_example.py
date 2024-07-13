from assistant.audio.audio_manager import AudioManager, TextToSpeechTask
from assistant.audio.util import play_audio


def example_chat_tts():
    from assistant.audio.tts_service import TTSServiceChatTTS
    from assistant.audio.audio_manager import TextToSpeechResultChatTTS

    text = "How to Record Audio using Python?"
    task = TextToSpeechTask(task_id="test", text=text)

    audio_manager = AudioManager()
    tts_service = TTSServiceChatTTS(audio_manager)

    print(f"run text to speech ...")
    raw_response = tts_service.convert(task=task)
    result = TextToSpeechResultChatTTS(task, raw_response)
    for url in result.file_urls:
        play_audio(url=url)


def example_melo_tts():
    from assistant.audio.tts_service import TTSServiceMeloTTS
    from assistant.audio.audio_manager import TextToSpeechResultMeloTTS

    text = "How to Record Audio using Python?"
    task = TextToSpeechTask(task_id="test", text=text)

    audio_manager = AudioManager()
    tts_service = TTSServiceMeloTTS(audio_manager)

    print(f"run text to speech ...")
    raw_response = tts_service.convert(task=task)
    result = TextToSpeechResultMeloTTS(task, raw_response)
    play_audio(url=None, content=result.content)


if __name__ == "__main__":
    example_melo_tts()
