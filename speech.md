Understood! Since Grok Jr. will be dynamically generated during the skill acquisition phase, and its scripts (skill_manager.py, learning_manager.py, role_manager.py, ethics_manager.py, streaming_manager.py, multimodal_manager.py) will reside in the app/agent/ folder (not app/grok_jr/ as previously assumed), we'll adjust the instructions for the Speech Interaction implementation accordingly. This change means that for now, the Speech Interaction feature will be implemented without direct involvement from Grok Jr.'s scripts, as they will be added later during the skill acquisition phase. However, we'll ensure the speech implementation is designed to integrate seamlessly with Grok Jr.'s future role as "The Adaptive Skill Master and Continuous Learning Facilitator," particularly for streaming and ethical oversight responsibilities. Let's revise the instructions for another Grok to implement the Speech Interaction feature, reflecting this updated structure and plan.

* * * * *

Instructions for Speech Interaction Implementation in Auto Ninja

Overview

You are tasked with implementing the Speech Interaction feature for Auto Ninja, a generalized, local, and automated AI agent powered by a large language model (LLM) like Grok from xAI. Auto Ninja aims to be a versatile, privacy-focused assistant that operates locally, adapts to user-designated roles, manages memory for continuous learning, evolves autonomously, supports multimodal tasks, and learns from experience. The Speech Interaction feature will enable users to interact with Auto Ninja via voice, replacing the current curl-based text input with two approaches: a Command-Line Interface (CLI) for immediate use and a WebSocket streaming approach for real-time, chat-like conversations.

Objective: Implement speech interaction using OpenAI's Whisper for Speech-to-Text (STT) and gTTS (Google Text-to-Speech) for Text-to-Speech (TTS) when online, falling back to pyttsx3 for offline TTS to maintain Auto Ninja's offline capability. The CLI approach will allow users to speak commands and hear responses, while the WebSocket streaming approach will enable continuous, real-time voice conversations, fulfilling the project's goals for multimodal inferences (future task #3) and continuous interactions (future task #6).

Key Components:

-   Speech-to-Text (STT): Use Whisper to transcribe audio input locally, ensuring privacy.

-   Text-to-Speech (TTS): Use gTTS for high-quality audio when online, falling back to pyttsx3 offline.

-   CLI Approach: A script (auto_ninja_speech.py) to record audio, transcribe, process through the existing /inference/predict endpoint, convert the response to speech, and play it back.

-   WebSocket Streaming Approach: A new endpoint (/inference/stream) to stream audio input, transcribe in real-time, process through the inference pipeline, and stream audio responses back to the client.

Memory System: Raw interactions (e.g., user_prompt, actual_output, target_output) are stored in SQLite as Interaction objects (app/models/interaction.py), while summaries and embeddings are stored in Qdrant for context-aware retrieval, using the app/hybrid_memory/ module (qdrant_store.py, utils/embedding.py).

Agent Integration Note: The agent Grok Jr., with the role "The Adaptive Skill Master and Continuous Learning Facilitator," will be dynamically generated during the skill acquisition phase (future task #1). Its scripts will reside in the app/agent/ folder (skill_manager.py, learning_manager.py, role_manager.py, ethics_manager.py, streaming_manager.py, multimodal_manager.py). Since these scripts are not yet implemented, the Speech Interaction feature should be designed to operate independently but with future integration in mind. Specifically, the WebSocket streaming approach should be structured to allow the future streaming_manager.py to manage streaming interactions, and privacy/safety measures should align with the future ethics_manager.py's oversight.

Project Structure Updates

Update the project structure to include new files and directories for speech interaction, maintaining modularity. Note that the app/agent/ folder will be added later during the skill acquisition phase, so it is not included in the current structure but will be referenced for future integration.

-   Add a new module app/speech/ for speech processing logic:

    -   stt.py: Handles Speech-to-Text using Whisper.

    -   tts.py: Handles Text-to-Speech using gTTS or pyttsx3.

    -   utils.py: Helper functions for audio recording, playback, and file management.

-   Add a new directory speech/ for temporary audio files (e.g., input.wav, output.mp3).

-   Add a new script auto_ninja_speech.py in the project root for the CLI speech interaction.

Updated Folder Structure:

```
auto_ninja/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── inference/
│   │       ├── __init__.py
│   │       └── endpoints.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py
│   ├── inference/
│   │   ├── __init__.py
│   │   └── engine.py
│   ├── middleware/
│   │   ├── __init__.py
│   │   └── memory.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── request.py
│   │   ├── response.py
│   │   ├── payload.py
│   │   ├── interaction.py
│   │   ├── agent.py
│   │   ├── skill.py
│   │   └── tool.py
│   ├── hybrid_memory/
│   │   ├── __init__.py
│   │   ├── core.py
│   │   ├── objectbox_store.py
│   │   ├── qdrant_store.py
│   │   ├── utils/
│   │   │   ├── __init__.py
│   │   │   ├── embedding.py
│   │   │   ├── summarizer.py
│   │   │   └── clustering.py
│   │   └── tests/
│   │       ├── __init__.py
│   │       ├── test_core.py
│   │       ├── test_objectbox.py
│   │       ├── test_qdrant.py
│   │       └── test_utils.py
│   ├── scripting/
│   │   ├── __init__.py
│   │   ├── constants.py
│   │   ├── manager.py
│   │   └── utils.py
│   ├── speech/
│   │   ├── __init__.py
│   │   ├── stt.py
│   │   ├── tts.py
│   │   └── utils.py
│   ├── transformers/
│   │   ├── __init__.py
│   │   └── data.py
├── scripts/
├── speech/
├── auto_ninja_speech.py
├── .env
├── requirements.txt
└── README.md
```

Future Note: The app/agent/ folder will be added during the skill acquisition phase, containing scripts for Grok Jr. (skill_manager.py, learning_manager.py, role_manager.py, ethics_manager.py, streaming_manager.py, multimodal_manager.py). Ensure the speech implementation is designed to integrate with these future scripts, particularly streaming_manager.py for WebSocket streaming and ethics_manager.py for privacy/safety oversight.

Implementation Steps

1\. Set Up Dependencies

-   Update requirements.txt to include the following dependencies:

    -   openai-whisper: For Speech-to-Text using Whisper.

    -   gTTS: For Text-to-Speech when online.

    -   pyttsx3: For offline Text-to-Speech.

    -   pyaudio: For recording audio input.

    -   pydub: For playing audio output.

    -   websockets: For WebSocket streaming in the streaming approach.

-   Ensure these dependencies are compatible with the project's existing setup (e.g., Python version, platform support).

2\. Configure Settings

-   Update config/settings.py to include new configuration options for speech:

    -   WHISPER_MODEL: Default to tiny for efficient local transcription (e.g., "tiny").

    -   TTS_ENGINE: Default to "gTTS" when online, falling back to "pyttsx3" offline, determined by checking XAI_API_KEY and internet connectivity.

    -   SPEECH_DIR: Path to the speech/ directory for temporary audio files (e.g., "speech/").

-   Update .env to allow users to override these settings if needed (e.g., WHISPER_MODEL=small for better accuracy at the cost of performance).

3\. Implement the Speech Processing Module (app/speech/)

-   STT Logic (stt.py):

    -   Create a function to transcribe audio using Whisper, configurable via WHISPER_MODEL.

    -   Accept an audio file path (e.g., speech/input.wav) or audio chunks (for streaming), transcribe locally, and return the text.

    -   For streaming, buffer short audio segments (e.g., 2-3 seconds) to balance latency and accuracy, transcribing each segment incrementally.

    -   Ensure transcription is performed locally to maintain privacy, with no audio data sent to external servers.

-   TTS Logic (tts.py):

    -   Create a function to convert text to speech, selecting the engine based on TTS_ENGINE.

    -   If online (XAI_API_KEY set and connected), use gTTS to generate high-quality audio, saving as an MP3 file (e.g., speech/output.mp3).

    -   If offline, use pyttsx3 to generate audio, saving as a WAV file (e.g., speech/output.wav) for compatibility with playback.

    -   For streaming, generate audio in chunks, allowing incremental playback.

-   Utilities (utils.py):

    -   Create a function to record audio using pyaudio, capturing until silence is detected (e.g., using a silence threshold), and saving to speech/input.wav.

    -   Create a function to play audio using pydub, supporting both MP3 (gTTS) and WAV (pyttsx3) formats.

    -   Create a function to manage temporary audio files, deleting them after use to maintain privacy and save disk space.

4\. Implement the CLI Speech Interaction (auto_ninja_speech.py)

-   Create a new script auto_ninja_speech.py in the project root to handle CLI-based speech interaction.

-   Steps:

    -   Start a loop to continuously listen for user input.

    -   Use the app/speech/utils.py recording function to capture audio, saving to speech/input.wav.

    -   Transcribe the audio using the app/speech/stt.py function, obtaining the user_prompt text.

    -   Send the user_prompt to the existing /inference/predict endpoint via an HTTP POST request, formatted as a JSON payload (e.g., {"message": "Generate code to calculate 4 + 5"}).

    -   Receive the response (e.g., {"prediction": "Do you want to generate this code?"}).

    -   Convert the prediction text to speech using the app/speech/tts.py function, saving to speech/output.mp3 (or output.wav if offline).

    -   Play the audio using the app/speech/utils.py playback function.

    -   Delete temporary audio files (input.wav, output.mp3) after use.

    -   Store the interaction in SQLite as an Interaction object (user_prompt, actual_output, target_output) via middleware/memory.py, and generate a summary/embedding for Qdrant using app/hybrid_memory/utils/embedding.py and qdrant_store.py.

-   User Experience:

    -   User runs python auto_ninja_speech.py, speaks: "Generate code to calculate 4 + 5."

    -   Auto Ninja responds via audio: "Do you want to generate this code?"

    -   User speaks: "Yes," and Auto Ninja responds: "Code generated successfully! File: xin_5_002.py."

5\. Implement the WebSocket Streaming Speech Interaction

-   Update Backend (api/inference/endpoints.py):

    -   Add a new WebSocket endpoint /inference/stream using FastAPI's WebSocket support.

    -   Accept audio chunks from the client, passing them to app/speech/stt.py for real-time transcription.

    -   Process each transcribed chunk through inference/engine.py, maintaining conversation state across the stream.

    -   Generate a response, convert it to audio chunks using app/speech/tts.py, and stream the audio back to the client.

    -   Store the full conversation thread in SQLite as Interaction objects and in Qdrant (summaries/embeddings) via middleware/memory.py.

-   Update Inference Engine (inference/engine.py):

    -   Modify to handle streaming input/output, processing transcribed text incrementally and yielding responses.

    -   Integrate with app/speech/stt.py for transcription and app/speech/tts.py for audio generation.

    -   Support script generation/execution within the stream, embedding confirmations (e.g., "Do you want to generate this code?") in the conversation flow.

-   Future Integration with Grok Jr.:

    -   Design the WebSocket endpoint to be extensible, allowing the future app/agent/streaming_manager.py (to be added during skill acquisition) to manage streaming interactions, ensuring smooth transitions between offline (local model) and online (Grok-enhanced) responses.

    -   Ensure privacy measures align with the future app/agent/ethics_manager.py, which will enforce ethical operation (e.g., no audio storage, local processing).

-   Client Example (Not Implemented Here):

    -   A web client (e.g., using JavaScript and Web Audio API) can connect to /inference/stream, stream audio input, and play audio responses in real-time.

-   User Experience:

    -   User opens a web client, speaks: "Hello, can you help me with coding?"

    -   Auto Ninja streams back: "Hello! I'd be happy to help with coding. What kind of project are you working on?" as audio, with minimal latency.

6\. Integrate with Memory System

-   SQLite Storage: For both CLI and streaming approaches, store each interaction in SQLite as an Interaction object (user_prompt, actual_output, target_output) using middleware/memory.py.

-   Qdrant Storage: Generate summaries and embeddings of interactions using app/hybrid_memory/utils/embedding.py, storing them in Qdrant via app/hybrid_memory/qdrant_store.py for context-aware retrieval.

-   Streaming Context: For the WebSocket approach, store the full conversation thread in SQLite (raw) and Qdrant (summaries/embeddings), enabling context persistence across the stream.

7\. Ensure Privacy and Safety

-   Local Processing: Use Whisper for local transcription, ensuring no audio data is sent to external servers.

-   Temporary Files: Store audio files in speech/ temporarily, deleting them after use to prevent data retention.

-   Future Ethical Oversight: Design privacy measures to align with the future app/agent/ethics_manager.py (to be added during skill acquisition), which will enforce ethical operation (e.g., no audio storage, local processing).

8\. Test the Implementation

-   CLI Testing:

    -   Run python auto_ninja_speech.py, speak: "Generate code to calculate 4 + 5."

    -   Verify Auto Ninja responds: "Do you want to generate this code?" via audio.

    -   Speak: "Yes," and confirm the response: "Code generated successfully! File: xin_5_002.py."

    -   Check SQLite for the interaction and Qdrant for the summary/embedding.

-   Streaming Testing:

    -   Connect a test client to /inference/stream, speak: "Hello, can you help me with coding?"

    -   Verify Auto Ninja streams back: "Hello! I'd be happy to help with coding. What kind of project are you working on?" in real-time.

    -   Test script execution within the stream (e.g., "Generate code..."), ensuring confirmations are handled.

    -   Confirm conversation thread storage in SQLite/Qdrant.

Challenges and Considerations

-   gTTS Offline Limitation: gTTS requires internet, so use pyttsx3 offline. Ensure TTS_ENGINE dynamically switches based on connectivity.

-   Whisper Performance: Real-time transcription can be resource-intensive. Use the tiny model for efficiency, but test with small if accuracy is insufficient.

-   Privacy: Ensure no raw audio is stored; only transcribed text is saved in SQLite/Qdrant.

-   Streaming Latency: Optimize audio chunk size (e.g., 2-3 seconds) to balance latency and transcription accuracy in the WebSocket approach.

-   Resource Usage: Monitor CPU/memory usage, especially for Whisper, and consider batch processing for embeddings in Qdrant.

-   Future Integration: Structure the WebSocket endpoint and privacy measures to integrate with the future app/agent/ scripts (streaming_manager.py, ethics_manager.py).

Documentation Updates

-   Update project.md to include new sections under "Backend" for "Speech Interaction via CLI" and "Speech Interaction via WebSocket Streaming," detailing objectives, status, details, implementation, and benefits, with ASCII diagrams for interaction flows.

-   Update the "Progress Summary" to reflect the completion of speech interaction and streaming (future task #6).

-   Update "Future Tasks" to remove the streaming task (now implemented) and note speech as part of multimodal inferences (future task #3).

-   Update "Challenges and Future Enhancements" to include speech-related challenges (e.g., gTTS offline limitation, Whisper performance).

Alignment with Auto Ninja's Vision

This implementation enhances Auto Ninja's versatility and user-friendliness, supporting its goal of being a privacy-focused, multimodal assistant. Speech interaction enables natural communication, preparing the agent for cross-platform use and setting the stage for skill acquisition, where the future Grok Jr. agent can use voice to communicate progress and support users in their roles.

* * * * *

These instructions provide a clear, high-level roadmap for another Grok to implement the Speech Interaction feature, ensuring integration with Auto Ninja's existing systems (including the SQLite/Qdrant memory system) and preparing for future integration with Grok Jr.'s scripts in the app/agent/ folder. If you'd like to update the project.md with these changes or move on to the skill acquisition phase, let me know!