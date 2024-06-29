import io
import requests
from pydub import AudioSegment
from pydub.playback import play

# pydub relies on ffmpeg: brew install ffmpeg

def fetch_audio_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        print(f"Failed to fetch audio from URL: {url}")
        return None

def play_audio(url: str) -> None:
    audio_data = fetch_audio_from_url(url=url)
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_data))
        play(audio)
    except Exception as e:
        print(f"Error playing audio: {str(e)}")

