import sys
import asyncio
import aiohttp
import logging
import speech_recognition as sr
from app.speech.tts import TextToSpeech
from app.intent.intent_recognizer import IntentRecognizer
from app.scripting.agent_manager import AgentManager
from app.shared import SingletonInference
import os
import threading
import uvicorn
import json
import select
import tty
import termios
import pyaudio
import alsaaudio
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Verify CUDA at startup
cuda_available = torch.cuda.is_available()
device = "cuda" if cuda_available else "cpu"
logger.info(f"Using device: {device} (CUDA Available: {cuda_available})")
if cuda_available:
    logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}, CUDA Version: {torch.version.cuda}")
else:
    logger.warning("CUDA not available, falling back to CPU. Check driver or environment.")

async def check_server_status(session):
    try:
        async with session.get('http://localhost:8080/inference/health', timeout=5) as response:
            status = await response.json()
            logger.info(f"Server responded: {status}")
            return status.get('status') == 'healthy'
    except Exception as e:
        logger.warning(f"Failed to check server status: {e}")
        return False

def start_server():
    try:
        config = uvicorn.Config(
            "app.main:app",
            host="0.0.0.0",
            port=8080,
            log_level="info",
            loop="asyncio"
        )
        server = uvicorn.Server(config)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        logger.info("Starting FastAPI server...")
        loop.run_until_complete(server.serve())
        logger.info("FastAPI server stopped.")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")

def init_tts():
    try:
        return TextToSpeech()
    except Exception as e:
        logger.error(f"Failed to initialize TTS: {e}")
        return None

def speak(text, tts):
    if tts is None:
        logger.warning("TTS not initialized, skipping speech.")
        return
    try:
        if "Expected:" in text and "parameters" in text.lower():
            lines = text.split("\n")
            param_line = next((l for l in lines if l.startswith("Expected:")), lines[0])
            param_name = param_line.split("'")[1] if "'" in param_line else "parameter"
            summary = f"Need correct input for {param_name}. Use text mode for details."
            tts.synthesize(summary, "temp_speech.mp3")
            os.system("mpg123 -q temp_speech.mp3")
            os.remove("temp_speech.mp3")
        else:
            tts.synthesize(text, "temp_speech.mp3")
            os.system("mpg123 -q temp_speech.mp3")
            os.remove("temp_speech.mp3")
    except Exception as e:
        logger.error(f"Failed to synthesize speech: {e}")

def list_audio_devices():
    try:
        p = pyaudio.PyAudio()
        devices = []
        for i in range(p.get_device_count()):
            dev = p.get_device_info_by_index(i)
            if dev['maxInputChannels'] > 0:
                devices.append((i, f"Device {i}: {dev['name']} (Inputs: {dev['maxInputChannels']})"))
        p.terminate()
        return devices
    except Exception as e:
        logger.error(f"Failed to list audio devices: {e}")
        return []

def get_mic_index(target_name="HDA Intel PCH"):
    devices = list_audio_devices()
    for index, desc in devices:
        if target_name in desc:
            return index
    return None if not devices else devices[0][0]

def check_mic_gain():
    try:
        mic = alsaaudio.Mixer('Capture', cardindex=1)
        gain = mic.getvolume()[0]
        if gain < 50:
            mic.setvolume(70)
            logger.info(f"Adjusted mic gain to 70% (was {gain}%)")
        mic.close()
    except Exception as e:
        logger.warning(f"Failed to check/adjust mic gain: {e}")

def listen(recognizer):
    max_retries = 3
    timeout = 15
    device_index = get_mic_index("HDA Intel PCH") or 4
    check_mic_gain()

    devices = list_audio_devices()
    if not devices:
        print("No microphones detected. Please connect a microphone and try again, or type a command.")
        return None
    logger.info(f"Using microphone: Device {device_index}")

    for attempt in range(max_retries):
        try:
            source = sr.Microphone(device_index=device_index, sample_rate=44100)
            with source as src:
                print(f"Listening (attempt {attempt + 1}/{max_retries})... (say 'text' to switch modes, or type a command)")
                recognizer.adjust_for_ambient_noise(src, duration=0.5)
                audio = recognizer.listen(src, timeout=timeout, phrase_time_limit=10)
                command = recognizer.recognize_google(audio).strip().lower()
                logger.info(f"Recognized speech: {command}")
                return command
        except (sr.WaitTimeoutError, sr.UnknownValueError, sr.RequestError) as e:
            print(f"Speech error: {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
            continue
        except Exception as e:
            print(f"Audio device error: {e}")
            print("Available audio devices:")
            for _, desc in devices:
                print(f"- {desc}")
            print("Please check your microphone configuration (e.g., unmute in alsamixer, ensure hw:1,0 is correct).")
            return None
    print("Speech input failed after retries. Type a command or say 'text' to switch modes.")
    return None

async def read_stdin(timeout=0.1):
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            command = sys.stdin.readline().strip()
            return command
        return None
    except Exception as e:
        logger.error(f"Failed to read stdin: {e}")
        return None
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

async def process_command(inference_engine, agent_manager: AgentManager, agent, command, mode, tts, recognizer, intent_recognizer: IntentRecognizer):
    command = command.strip().lower()
    intent, params = intent_recognizer.recognize_intent(command)
    logger.info(f"Processed command: {command}, Intent: {intent}, Params: {params}")

    if intent == "exit":
        return False, mode
    elif intent == "use_speech":
        mode = "speech"
        response = "Switched to speech mode. Speak or type your command, or say 'text' to switch to text-only."
        print(response)
        speak(response, tts)
    elif intent == "use_text":
        mode = "text"
        response = "Switched to text mode. Type your command."
        print(response)
        speak(response, tts)
    elif intent == "list_skills":
        try:
            skills = agent_manager.list_skills()
            logger.info(f"Skills for agent {agent.id}: {[s.id for s in skills]}")
            output_lines = []
            for s in skills:
                line = f"ID: {s.id}, Name: {s.name}, Acquired: {s.acquired}, Description: {s.description or 'None'}"
                if s.instructions["type"] == "script" and s.instructions.get("parameters"):
                    params = s.instructions["parameters"]
                    line += "\nParameters:"
                    for p in params:
                        line += f"\n- {p['name']} ({p['type']}, {'required' if p['required'] else 'optional'}): {p['description']}"
                        if "default" in p and p['default'] is not None:
                            line += f" (default: {p['default']})"
                        if p.get("fields"):
                            line += "\n  Fields:"
                            for f in p["fields"]:
                                line += f"\n  - {f['name']} ({f['type']}): {f['description']}"
                output_lines.append(line)
            output = "\n".join(output_lines) or "No skills found."
            print(output)
            if mode == "speech":
                speak("Listing skills. Check text output for details.", tts)
        except Exception as e:
            logger.error(f"Failed to list skills: {e}")
            error = f"Error: {e}"
            print(error)
            if mode == "speech":
                speak(error, tts)
    elif intent == "acquire_skill":
        try:
            name = params.get("name", "")
            desc = params.get("description", "")
            if not name:
                response = "Please provide a skill name (e.g., 'acquire skill writing, Draft reports')."
                print(response)
                if mode == "speech":
                    speak(response, tts)
                return True, mode
            skill = agent_manager.acquire_skill(name.strip(), desc.strip())
            response = f"Acquired skill '{skill.name}' with ID {skill.id}"
            print(response)
            if mode == "speech":
                speak(response, tts)
        except Exception as e:
            logger.error(f"Failed to acquire skill: {e}")
            error = f"Error: {e}"
            print(error)
            if mode == "speech":
                speak(error, tts)
    elif intent == "use_skill":
        try:
            skill_id = int(params.get("skill_id", 0))
            task = params.get("task", "")
            skill_params = params.get("parameters", {})
            if not skill_id or not task:
                response = "Please provide a skill ID and task (e.g., 'use skill 47, draft report')."
                print(response)
                if mode == "speech":
                    speak(response, tts)
                return True, mode

            skills = agent_manager.list_skills()
            skill = next((s for s in skills if s.id == skill_id), None)
            if not skill:
                response = f"Skill ID {skill_id} not found in agent {agent.id}'s skills"
                print(response)
                if mode == "speech":
                    speak(response, tts)
                return True, mode

            if task and skill.instructions["type"] == "script" and skill.instructions.get("parameters"):
                params = skill.instructions["parameters"]
                if len(params) == 1 and params[0]["type"].startswith("list[") and "," in task and not task.startswith("["):
                    task = json.dumps(task.split(","))
                output = agent_manager.use_skill(skill_id, task, skill_params)
            else:
                output = agent_manager.use_skill(skill_id, task)
            print(output)
            if mode == "speech":
                speak(output, tts)
        except Exception as e:
            logger.error(f"Failed to use skill: {e}")
            error = f"Error: {e}"
            print(error)
            if mode == "speech":
                speak(error, tts)
    elif intent == "update_skill":
        try:
            skill_id = int(params.get("skill_id", 0))
            desc = params.get("description", "")
            if not skill_id or not desc:
                response = "Please provide a skill ID and description (e.g., 'update skill 47, New logic')."
                print(response)
                if mode == "speech":
                    speak(response, tts)
                return True, mode

            skills = agent_manager.list_skills()
            skill = next((s for s in skills if s.id == skill_id), None)
            if not skill:
                response = f"Skill ID {skill_id} not found in agent {agent.id}'s skills"
                print(response)
                if mode == "speech":
                    speak(response, tts)
                return True, mode

            skill = agent_manager.update_skill(skill_id, desc)
            response = f"Updated skill '{skill.name}' with ID {skill_id}"
            print(response)
            if mode == "speech":
                speak(response, tts)
        except Exception as e:
            logger.error(f"Failed to update skill: {e}")
            error = f"Error: {e}"
            print(error)
            if mode == "speech":
                speak(error, tts)
    elif intent == "delete_skill":
        try:
            skill_id = int(params.get("skill_id", 0))
            if not skill_id:
                response = "Please provide a skill ID to delete (e.g., 'delete skill 47')."
                print(response)
                if mode == "speech":
                    speak(response, tts)
                return True, mode

            skills = agent_manager.list_skills()
            skill = next((s for s in skills if s.id == skill_id), None)
            if not skill:
                response = f"Skill ID {skill_id} not found in agent {agent.id}'s skills"
                print(response)
                if mode == "speech":
                    speak(response, tts)
                return True, mode

            response = agent_manager.delete_skill(skill_id)
            print(response)
            if mode == "speech":
                speak(response, tts)
        except Exception as e:
            logger.error(f"Failed to delete skill: {e}")
            error = f"Error: {e}"
            print(error)
            if mode == "speech":
                speak(error, tts)
    elif intent == "acquire_next":
        try:
            n = int(params.get("count", 1))
            response = f"Acquire next {n} not implemented yet"
            print(response)
            if mode == "speech":
                speak(response, tts)
        except Exception as e:
            logger.error(f"Failed to acquire next: {e}")
            error = f"Error: {e}"
            print(error)
            if mode == "speech":
                speak(error, tts)
    elif intent in ["script_generate", "script_execute", "script_update", "script_delete"]:
        try:
            details = {"prompt": command}
            if intent != "script_generate":
                file_name = params.get("file_name", "")
                if not file_name:
                    response = "Please specify a valid script file (e.g., automation.py)."
                    print(response)
                    if mode == "speech":
                        speak(response, tts)
                    return True, mode
                details["file_name"] = file_name
            response = agent_manager.request_confirmation(intent.replace("script_", ""), details)
            print(response)
            if mode == "speech":
                speak(response, tts)
        except Exception as e:
            logger.error(f"Failed to process script command: {e}")
            error = f"Error: {e}"
            print(error)
            if mode == "speech":
                speak(error, tts)
    else:
        try:
            response = inference_engine.process(command)
            print(response)
            if mode == "speech":
                speak(response, tts)
        except Exception as e:
            logger.error(f"Failed to process command: {e}")
            error = f"Error: {e}"
            print(error)
            if mode == "speech":
                speak(error, tts)
    return True, mode

async def main_async():
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    await asyncio.sleep(2)

    async with aiohttp.ClientSession() as session:
        if not await check_server_status(session):
            logger.warning("Server not running, proceeding with local inference.")
        else:
            logger.info("Server is healthy, proceeding with full functionality.")

        try:
            inference_engine = SingletonInference.get_instance()
            memory_manager = inference_engine.memory_manager
            agent_manager = AgentManager(inference_engine, memory_manager)
            intent_recognizer = IntentRecognizer(inference_engine, "memory.db")
            memory_manager.set_intent_recognizer(intent_recognizer)
            agent = memory_manager.current_agent
            print(f"Using agent {agent.name} (Role: {agent.role}). Type your command (e.g., 'list skills', 'exit'): ")
            print("Commands: 'list skills', 'acquire skill <name>, <description>', 'use skill <id>, <task>', ...")
        except Exception as e:
            logger.error(f"Failed to initialize inference engine or agent: {e}")
            return

        mode = "text"
        tts = init_tts()
        recognizer = sr.Recognizer()
        speech_failures = 0
        max_speech_failures = 5

        while True:
            if mode == "text":
                command = input("> ").strip()
                speech_failures = 0
            else:
                command = await read_stdin()
                if command:
                    print(f"Typed command: {command}")
                    speech_failures = 0
                else:
                    command = listen(recognizer)
                    if command == "text":
                        mode = "text"
                        print("Switched to text mode. Type your command.")
                        speak("Switched to text mode.", tts)
                        command = input("> ").strip()
                        speech_failures = 0
                    elif not command:
                        speech_failures += 1
                        print("No command received. Staying in speech mode.")
                        if speech_failures >= max_speech_failures:
                            mode = "text"
                            print("Too many speech failures, switching to text mode.")
                            speak("Switching to text mode due to repeated failures.", tts)
                            command = input("> ").strip()
                            speech_failures = 0
                        continue
                    else:
                        speech_failures = 0

            continue_running, mode = await process_command(
                inference_engine, agent_manager, agent, command, mode, tts, recognizer, intent_recognizer
            )
            if not continue_running:
                break

def main():
    logger.info("Starting interaction. Starting server and checking status...")
    try:
        asyncio.run(main_async())
    except Exception as e:
        logger.error(f"Main loop failed: {e}")

if __name__ == "__main__":
    main()