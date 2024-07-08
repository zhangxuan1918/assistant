import io
import threading
import requests
from pydub import AudioSegment
from pydub.playback import play
import pyaudio
import wave
from keys.util import (
    AUDIO_INPUT_END,
    AUDIO_INPUT_END_STR,
    monitor_keyboard_and_execute_func,
)

# pydub relies on ffmpeg: brew install ffmpeg


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
        audio = AudioSegment.from_file(io.BytesIO(audio_data))
        play(audio)
    except Exception as e:
        print(f"Error playing audio: {str(e)}")


def record_audio(
    p: pyaudio.PyAudio, filepath: str, channels=1, rate=16000, chunk=1024
) -> None:
    print(f"Recording... Press '{AUDIO_INPUT_END_STR}' to stop")
    frames = []
    stop_recording_flag = threading.Event()

    # Open stream.
    stream = p.open(
        format=pyaudio.paInt16,
        channels=channels,
        rate=rate,
        input=True,
        frames_per_buffer=chunk,
    )

    def _record_audio_chunk():
        while not stop_recording_flag.is_set():
            data = stream.read(chunk, exception_on_overflow=False)
            frames.append(data)

    try:
        monitor_keyboard_and_execute_func(
            expected_keys=AUDIO_INPUT_END,
            stop_flag=stop_recording_flag,
            func=_record_audio_chunk,
        )
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Stop and close the stream
        stream.stop_stream()
        stream.close()

        # Save the recorded data as a WAV file
        wf = wave.open(filepath, "wb")
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(rate)
        wf.writeframes(b"".join(frames))
        wf.close()


if __name__ == "__main__":
    filepath = "/tmp/what_is_langform.wav"
    py_audio = pyaudio.PyAudio()
    record_audio(p=py_audio, filepath=filepath)
