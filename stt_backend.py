import os
import json
import wave
from google.cloud import speech
from datetime import datetime
import io

# Google Cloud 설정 (환경 변수에 JSON 키 파일 경로 설정 필요)
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key/hyub_google_cloud_key.json"
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/kimhyub/Downloads/STT/key/hyub_google_cloud_key.json"
current_directory = os.path.dirname(os.path.abspath(__file__))
credentials_path = os.path.join(current_directory, "hyub_google_cloud_key.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

# Google Speech-to-Text client
client = speech.SpeechClient()

def process_audio_file(audio_buffer: io.BytesIO):
    """Processes an in-memory .wav file and converts it to text."""

    # Read the audio file
    with wave.open(audio_buffer, "rb") as wf:
        sample_rate = wf.getframerate()  # Get the sample rate
        num_channels = wf.getnchannels()  # Get the number of channels
        audio_data = wf.readframes(wf.getnframes())  # Read the audio frames

    # Ensure correct format (Mono, LINEAR16)
    if num_channels != 1:
        return "Error: Audio file must be in Mono (1 channel)."

    # Google STT request
    audio = speech.RecognitionAudio(content=audio_data)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sample_rate,
        language_code="ko-KR"
    )

    # Perform recognition
    response = client.recognize(config=config, audio=audio)
    if not response.results:
        return "No speech detected."

    # Extract and return transcribed text
    transcript = " ".join(result.alternatives[0].transcript.strip() for result in response.results)
    return transcript