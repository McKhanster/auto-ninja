import pyaudio
import wave
import os
from pydub import AudioSegment
from pydub.playback import play
from ..config.settings import settings
import logging
import threading
import queue

logger = logging.getLogger(__name__)

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

def record_audio(output_path: str) -> None:
    """Record audio until Enter is pressed in the terminal."""
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, 
                       input=True, frames_per_buffer=CHUNK)
    
    logger.info(f"Recording audio to {output_path}... Press Enter to stop.")
    frames = []
    stop_event = queue.Queue()  # Thread-safe way to signal stop

    def wait_for_enter():
        input()  # Blocks until Enter is pressed
        stop_event.put(True)

    # Start a thread to wait for Enter
    input_thread = threading.Thread(target=wait_for_enter)
    input_thread.start()

    # Record until stop signal is received
    while stop_event.qsize() == 0:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
        except IOError as e:
            logger.warning(f"IOError during recording: {str(e)}")
            continue

    logger.info("Recording stopped by user (Enter pressed).")
    stream.stop_stream()
    stream.close()
    audio.terminate()
    input_thread.join()  # Clean up the thread

    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

def play_audio(audio_path: str) -> None:
    """Play audio file using pydub."""
    try:
        logger.info(f"Playing audio from {audio_path}")
        audio = AudioSegment.from_file(audio_path)
        play(audio)
    except Exception as e:
        logger.error(f"Playback failed: {str(e)}")
        raise ValueError(f"Failed to play audio: {str(e)}")

def cleanup_files(*paths: str) -> None:
    """Delete temporary audio files."""
    for path in paths:
        if os.path.exists(path):
            os.remove(path)
            logger.info(f"Deleted temporary file: {path}")