import io
import requests
from pydub import AudioSegment
from pydub.playback import play
import pyaudio
import keyboard
import wave

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

def record_audio(p: pyaudio.PyAudio, filepath: str, channels=1, rate=44100, chunk=1024):
    # Initialize PyAudio
    p = pyaudio.PyAudio()
    # Open stream.
    stream = p.open(format=pyaudio.paInt16, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)
    print("Recording... Press 'q' to stop")

    frames = []
    try:
        while True:
            # Read audio data from the stream
            data = stream.read(chunk)
            frames.append(data)
            
            # Check if the user has pressed the 'q' key
            if keyboard.is_pressed('q'):
                print("Recording stopped.")
                break
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        p.terminate()

        # Save the recorded data as a WAV file
        wf = wave.open(filepath, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
        wf.close()