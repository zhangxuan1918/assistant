import io
import requests
from pydub import AudioSegment
from pydub.playback import play
import speech_recognition as sr

# pydub relies on ffmpeg: brew install ffmpeg
speech_recognizer = sr.Recognizer()


def fetch_audio_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        print(f"Failed to fetch audio from URL: {url}")
        return None


def play_audio(url: str | None, content: bytes | None = None) -> None:

    if content is not None:
        audio_data = content
    elif url is not None:
        audio_data = fetch_audio_from_url(url=url)
    else:
        print("Error: both url and content are none, cannot play audio!")
        return

    try:
        # Need to specify format="wav", otherwise it's very slow.
        audio = AudioSegment.from_file(io.BytesIO(audio_data), format="wav")
        play(audio)
    except Exception as e:
        print(f"Error playing audio: {str(e)}")


def record_audio(
    device_index=None, duration=None, engery_threshold=300, pause_threshold=0.8
) -> bytes | None:
    speech_recognizer.energy_threshold = engery_threshold
    speech_recognizer.pause_threshold = pause_threshold
    with sr.Microphone(device_index=device_index) as source:
        print(f"Recording...")
        try:
            if duration:
                audio = speech_recognizer.listen(
                    source, timeout=duration
                )  # Record for duration seconds
            else:
                audio = speech_recognizer.listen(source)  # Record until silence
            print("Recording finished.")
            return audio.get_wav_data()
        except Exception as e:
            print(f"Error recording audio: {e}")
            return None

if __name__ == "__main__":
    microphone_names = sr.Microphone.list_microphone_names()
    for index, name in enumerate(microphone_names):
        print(f"Microphone with index {index}: {name}")