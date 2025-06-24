import requests
import os
from app.speech.stt import SpeechToText
from app.speech.tts import TextToSpeech
from app.speech.utils import record_audio, play_audio, cleanup_files
from app.config.settings import settings
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

API_BASE_URL = f"http://{settings.API_HOST}:{settings.API_PORT}/inference"
INPUT_PATH = os.path.join(settings.SPEECH_DIR, "input.wav")
OUTPUT_PATH = os.path.join(settings.SPEECH_DIR, "output.mp3" if settings.TTS_ENGINE == "gTTS" else "output.wav")

def check_server() -> bool:
    """Check if the FastAPI server is running."""
    health_url = f"{API_BASE_URL}/health"
    try:
        response = requests.get(health_url, timeout=settings.REQUEST_TIMEOUT)
        response.raise_for_status()
        logger.info(f"Server responded: {response.json()}")
        return True
    except requests.RequestException as e:
        logger.error(f"Server not reachable at {health_url}: {str(e)}")
        return False

def select_agent() -> str:
    """Prompt user for an agent ID via keyboard input and validate via API."""
    while True:
        agent_id = input("Please enter the agent ID (e.g., 1, 2, 3): ").strip()
        if not agent_id.isdigit():
            print("Invalid input. Please enter a number.")
            continue

        select_url = f"{API_BASE_URL}/agent/select/{agent_id}"
        try:
            response = requests.post(select_url, timeout=settings.REQUEST_TIMEOUT)
            response.raise_for_status()
            logger.info(f"Selected agent {agent_id}: {response.json()['message']}")
            return agent_id
        except requests.RequestException as e:
            logger.error(f"Failed to select agent {agent_id}: {str(e)}")
            print(f"Error: Could not select agent {agent_id}. Try again.")

def main():
    """CLI speech interaction client for Auto Ninja."""
    stt = SpeechToText()
    tts = TextToSpeech()

    logger.info("Starting speech interaction. Checking server status...")
    if not check_server():
        print(f"Cannot connect to server at {API_BASE_URL}. Start it with 'uvicorn app.main:app --host 0.0.0.0 --port 8000'.")
        return

    try:
        agent_id = select_agent()
    except KeyboardInterrupt:
        logger.info("Agent selection cancelled by user.")
        return

    print(f"Using agent {agent_id}. Speak your command (say 'exit' to quit). Press Enter to stop recording.")

    while True:
        try:
            record_audio(INPUT_PATH)
            user_prompt = stt.transcribe(INPUT_PATH)
            cleanup_files(INPUT_PATH)
            if not user_prompt:
                logger.warning("No speech detected.")
                continue

            predict_url = f"{API_BASE_URL}/predict"
            payload = {"message": user_prompt}
            logger.info(f"Sending request to {predict_url} with payload: {payload}")
            response = requests.post(predict_url, json=payload, timeout=settings.REQUEST_TIMEOUT)
            response.raise_for_status()
            prediction = response.json()["prediction"]
            logger.info(f"Received prediction: {prediction}")

            tts.synthesize(prediction, OUTPUT_PATH)
            play_audio(OUTPUT_PATH)
            cleanup_files(OUTPUT_PATH)

            if user_prompt.lower() in ["exit", "quit"]:
                logger.info("Exiting speech interaction.")
                break

        except (requests.RequestException, ValueError) as e:
            logger.error(f"Error in speech interaction: {str(e)}")
            error_msg = "Sorry, something went wrong. Please try again."
            tts.synthesize(error_msg, OUTPUT_PATH)
            play_audio(OUTPUT_PATH)
            cleanup_files(OUTPUT_PATH)

if __name__ == "__main__":
    main()