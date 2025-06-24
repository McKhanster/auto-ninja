Auto Ninja Project Documentation
================================

Project Overview
----------------

**Project Name:** Auto Ninja

**Date:** April 17, 2025

**Vision:** Auto Ninja is a generalized, local, and automated AI agent powered by a large language model (LLM) like Grok from xAI. It aims to be a versatile, privacy-focused assistant that operates locally, adapts to user-designated roles (e.g., Software Engineer, Network Admin), manages memory for continuous learning, evolves autonomously, supports multimodal tasks (text, voice), and learns from experience. The ultimate goal is a self-sufficient, agentic agent that masters roles, applies new skills through a cycle of learning-doing-relearning, and collaborates across platforms (iPhone, Android, Windows). Inspired by human memory and skill application, it uses iterative refinement for tasks like "make the best pizza" or "write optimized code," with proactive goal-setting, advanced reasoning, environmental adaptability, swarm collaboration, and continuous learning.

**Objective:** Enable Auto Ninja to function as an agentic agent, operating both offline and online using a local model as the persistent user-facing agent, with Grok instructing online for continuous evolution. The local model handles all interactions, relaying to Grok for enhanced responses. Enhancements include multimodal inferences (speech, vision), memory management for learning, autonomous skill acquisition, robust skill management, and swarm collaboration for complex tasks, all overseen by Grok Jr. as the Adaptive Skill Master.

Current Folder Structure
------------------------

The project maintains modularity and separation of concerns, with recent cleanup (removal of `agent/`, merging `SkillManager` and `ToolManager` into `AgentManager`):

```
auto_ninja/
├── app/
│   ├── main.py
│   ├── api/
│   │   └── inference/
│   │       └── endpoints.py
│   ├── config/
│   │   ├── settings.py
│   │   └── constants.py
│   ├── inference/
│   │   └── engine.py
│   ├── middleware/
│   │   └── memory.py
│   ├── models/
│   │   ├── request.py
│   │   ├── response.py
│   │   ├── payload.py
│   │   ├── interaction.py
│   │   ├── agent.py
│   │   ├── skill.py
│   │   └── tool.py
│   ├── hybrid_memory/
│   │   ├── core.py
│   │   ├── sqlite_store.py
│   │   ├── qdrant_store.py
│   │   ├── utils/
│   │   │   ├── embedding.py
│   │   │   ├── summarizer.py
│   │   │   └── clustering.py
│   │   └── tests/
│   │       ├── test_core.py
│   │       ├── test_sqlite.py
│   │       ├── test_qdrant.py
│   │       └── test_utils.py
│   ├── scripting/
│   │   ├── constants.py
│   │   ├── agent_manager.py
│   │   └── utils.py
│   ├── speech/
│   │   ├── stt.py
│   │   ├── tts.py
│   │   └── utils.py
│   ├── transformers/
│   │   └── data.py
│   ├── grok_jr/
│   │   ├── skill_manager.py
│   │   ├── learning_manager.py
│   │   ├── role_manager.py
│   │   ├── ethics_manager.py
│   │   ├── streaming_manager.py
│   │   └── multimodal_manager.py
│   └── shared.py
├── scripts/
├── speech/
├── auto_ninja.py
├── .env
├── requirements.txt
└── README.md

```

Backend
-------

### Offline and Online Inference with Local Model

-   **Objective:** Enable Auto Ninja to function offline using a local model as the persistent user-facing agent, with Grok managing and instructing online for continuous learning and evolution.

-   **Status:** Fully implemented.

-   **Details:**

    -   A local model (`google/gemma-3-1b-it`, configurable) handles all user interactions, ensuring offline capability.

    -   When online with `XAI_API_KEY`, the local model relays prompts and inferences to Grok (`https://api.x.ai/v1/chat/completions`).

    -   Grok merges inputs into coherent responses, returned to the user.

    -   Middleware (`middleware/memory.py`) captures interactions for learning.

    -   **Interaction Mapping (ASCII Diagram):**

        ```
        +----------------+          +------------------+          +-----------------+          +------------------+
        |     User       |          |    Local Agent   |          |    Grok API     |          |    Middleware    |
        | (curl client)  |          | (FastAPI + Local)|          | (xAI Service)   |          |  (Memory Mgmt)   |
        +----------------+          +------------------+          +-----------------+          +------------------+
                   |                         |                           |                           |
                   | 1. POST /predict        |                           |                           |
                   | {"message": "Hello"}    |                           |                           |
                   |------------------------>|                           |                           |
                   |                         | 2. Local inference        |                           |
                   |                         |    (Local Model)          |                           |
                   |                         |                           |                           |
                   |                         | 3. If XAI_API_KEY &&      |                           |
                   |                         |    connection OK          |                           |
                   |                         |-------------------------->|                           |
                   |                         |    Send prompt + local    | 4. Merge inferences   |
                   |                         |    output                 |    coherently         |
                   |                         |                           |                           |
                   |                         |                           | 5. Return Grok resp   |
                   |                         |    No connection?         |<--------------------------|
                   |                         |    Skip to 6 ---------->|                           |
                   |                         |                           |                           |
                   |                         | 6. To Middleware          |                           |
                   |                         |-------------------------->|                           |
                   |                         |                           |                           | 7. Memory mgmt
                   |                         |                           |                           |    - Store interaction
                   |                         |                           |                           |    - Learn
                   |                         |                           |                           |
                   | 8. Response            |                           |                           | Local learns
                   | {"prediction": "..."}  |                           |                           |<---- from memory
                   |<------------------------|                           |                           |
                   |                         |                           |                           |

        ```

-   **Implementation Details:**

    -   Local Inference: `inference/engine.py` uses a configurable model (`MODEL_NAME` in `.env`).
    -   Grok Integration: Sends prompts and local inferences to Grok for merging.
    -   Memory Management: Stores interactions in SQLite and Qdrant.
    -   API Endpoint: `/inference/predict` in `api/inference/endpoints.py`.
-   **Benefits:**

    -   Offline functionality via local model.
    -   Consistent user experience with Grok enhancements.
    -   Evolving local model through stored interactions.

### Memory Management and Learning

-   **Objective:** Capture interactions for continuous learning and context-aware responses.

-   **Status:** Fully implemented.

-   **Details:**

    -   Interactions stored in SQLite: `user_prompt`, `actual_output` (local), `target_output` (Grok/local).

    -   Summaries embedded in Qdrant for vector-based retrieval.

    -   Middleware provides context from last 3 interactions.

    -   Data structured for fine-tuning (`actual_output` vs. `target_output`).

    -   **Example SQLite Entry:**

        ```
        id: 1
        agent_id: 1
        user_prompt: "Hello, how is the weather at your data center?"
        actual_output: "The weather at my data center is currently stable. The temperature is around 22 degrees Celsius."
        target_output: "Hello! The weather at my data center is stable, around 22°C with typical humidity---ideal for operations. How's your weather?"

        ```

-   **Benefits:**

    -   Enables continuous learning for fine-tuning.
    -   Supports context-aware responses via Qdrant.
    -   Prepares for agentic learning from outcomes.

### Multimodal Readiness

-   **Objective:** Support multimodal inferences (speech, vision) via configurable API interactions.

-   **Status:** Partially implemented.

-   **Details:**

    -   Configurable Grok API URL (`GROK_URL` in `config/settings.py`).
    -   Modular payloads in `models/payload.py` (`GrokPayload` for chat, extensible for vision/speech).
-   **Benefits:**

    -   Ready for multimodal endpoints (e.g., `/v1/vision`).
    -   Maintains modularity for future payload types.

### Script Generation and Execution

-   **Objective:** Generate, execute, update, and delete Python scripts with user confirmation for safety.

-   **Status:** Fully implemented.

-   **Details:**

    -   Commands: "generate code", "run code", "update code", "delete code".
    -   Scripts saved in `scripts/` (e.g., `xin_5_002.py`), stored as `Tool` objects in SQLite.
    -   Triggers defined in `scripting/constants.py`, managed by `AgentManager` (`scripting/agent_manager.py`).
    -   Confirmation required (e.g., "Do you want to generate this code?").
    -   **Example Interaction:**
        -   Request: "Generate code to calculate 4 + 5"
        -   Response: "Do you want to generate this code?"
        -   Confirm: "yes"
        -   Response: "Code generated successfully! File: xin_5_002.py"
-   **Implementation Details:**

    -   `scripting/agent_manager.py`: Manages script lifecycle.
    -   `scripting/utils.py`: File handling (e.g., `save_script`).
    -   Integrated with `middleware/memory.py` for metadata storage.
-   **Benefits:**

    -   Robust coding capabilities.
    -   User control via confirmation.
    -   Seamless memory integration.

### Speech Interaction via CLI

-   **Objective:** Enable speech-based CLI interaction using Whisper (STT) and gTTS/pyttsx3 (TTS).

-   **Status:** Fully implemented.

-   **Details:**

    -   `auto_ninja.py` records audio, transcribes with Whisper, processes via `/inference/predict`, and plays responses.
    -   Temporary audio in `speech/` (e.g., `input.wav`, `output.mp3`), deleted after use.
    -   **Example Interaction:**
        -   User: "Generate code to calculate 4 + 5"
        -   Auto Ninja: "Do you want to generate this code?" (spoken)
        -   User: "Yes"
        -   Auto Ninja: "Code generated successfully! File: xin_5_002.py"
-   **Implementation Details:**

    -   `speech/stt.py`: Whisper transcription.
    -   `speech/tts.py`: gTTS online, pyttsx3 offline.
    -   `speech/utils.py`: Audio recording/playback.
-   **Benefits:**

    -   Privacy-focused local STT.
    -   Offline TTS via pyttsx3.
    -   Seamless CLI integration.

### Speech Interaction via WebSocket Streaming

-   **Objective:** Enable real-time speech via WebSocket for continuous conversations.

-   **Status:** Fully implemented.

-   **Details:**

    -   WebSocket endpoint `/inference/stream` streams audio, transcribes, processes, and returns audio responses.
    -   Uses Whisper for real-time STT, gTTS/pyttsx3 for TTS.
    -   **Example Interaction:**
        -   User: "Help with coding"
        -   Auto Ninja: "What kind of project are you working on?" (streamed audio)
-   **Implementation Details:**

    -   `api/inference/endpoints.py`: WebSocket endpoint.
    -   `inference/engine.py`: Streaming inference.
    -   `speech/stt.py`, `speech/tts.py`: Real-time audio processing.
-   **Benefits:**

    -   Low-latency, chat-like experience.
    -   Supports long-running tasks with live updates.

### Skill Management

-   **Objective:** Manage skills (acquire, use, list, update, delete) with script integration, overseen by Grok Jr.

-   **Status:** Fully implemented, with intent recognition fix completed April 17, 2025.

-   **Details:**

    -   CLI commands: `acquire skill`, `use skill`, `list skills`, `update skill`, `delete skill`.
    -   Skills stored in SQLite with `agent_id`, supporting script-based (e.g., `skill_network_configuration_001.py`) and text-based types.
    -   Intent recognition fixed to classify commands accurately (e.g., `list_skills`), resolving `shapes (384,) and (192,) not aligned` error by updating `np.frombuffer` to use `dtype=np.float32` in `app/intent/intent_recognizer.py`.
    -   **Example Interaction:**
        -   Command: "list skills"
        -   Response: Lists 34 skills (e.g., `ID: 17, Name: Network Configuration, Acquired: True`).
        -   Command: "acquire skill Firewall Optimization, Optimize firewall rules"
        -   Response: "Acquired skill 'Firewall Optimization' with ID 35"
-   **Implementation Details:**

    -   `scripting/agent_manager.py`: Manages skill lifecycle, integrated with script generation.
    -   `intent/intent_recognizer.py`: Fixed to ensure accurate intent classification.
    -   `auto_ninja.py`: CLI interface for skill commands.
-   **Benefits:**

    -   Robust skill management for role-specific tasks.
    -   Fixed intent recognition enables reliable command processing.
    -   Incremental script preservation supports user workflows.

### Auto-Skill Acquisition

-   **Objective:** Auto-acquire role-specific skills at startup, managed by Grok Jr.

-   **Status:** Fully implemented.

-   **Details:**

    -   Checks `skills` table on startup; if empty, acquires role-specific skills (e.g., `network_scan`, `ping_test` for Network Admin).
    -   Uses `AgentManager.auto_acquire_skills`, generating scripts as needed.
    -   **Example Interaction:**
        -   Startup: Creates `Ninja` (Network Admin), acquires skills like `ID: 20, Name: ping, Acquired: True`.
-   **Implementation Details:**

    -   `shared.py`: Triggers `auto_acquire_skills` in `SingletonInference`.
    -   `scripting/agent_manager.py`: Fetches role info, generates skills/scripts.
    -   `middleware/memory.py`: Logs acquisition.
-   **Benefits:**

    -   Immediate utility for new agents.
    -   Role-specific customization via Grok Jr.

Progress Summary (as of April 17, 2025)
---------------------------------------

-   **Inference**: Fully implemented; local model handles interactions, Grok enhances online.
-   **Memory Management**: Fully implemented; SQLite/Qdrant store interactions.
-   **Multimodal Readiness**: Partially implemented; speech supported, vision pending.
-   **Script Generation and Execution**: Fully implemented; robust script lifecycle.
-   **Speech Interaction via CLI**: Fully implemented; Whisper STT, gTTS/pyttsx3 TTS.
-   **Speech Interaction via WebSocket Streaming**: Fully implemented; real-time streaming.
-   **Skill Management**: Fully implemented; intent recognition fixed (April 17, 2025), listing 34 skills, with duplicates (e.g., `Network Configuration`) pending cleanup.
-   **Auto-Skill Acquisition**: Fully implemented; role-specific skills acquired at startup.
-   **Grok Jr. Integration**: Expanded as Adaptive Skill Master, managing skill lifecycle and preparing for agentic swarm collaboration.

Future Tasks (Configured for Agentic Agent)
-------------------------------------------

1.  **Implement Swarm Collaboration for Agentic Coordination**

    -   **Objective**: Enable multiple AutoNinja agents to form a swarm, autonomously coordinating tasks, sharing skills, and collaborating on complex objectives (e.g., network troubleshooting across devices), managed by Grok Jr.
    -   **Proposed Approach**:
        -   Extend `app/shared.py` to support multiple agents with unique `agent_id` and roles.
        -   Add `app/swarm/SwarmManager.py` for message passing via WebSocket/Redis:
            -   Message types: task delegation (e.g., `skill_network_scan_001.py`), skill sharing (e.g., `skill_security_management_001.py`), coordination signals.
            -   Conflict resolution via Grok Jr.'s `grok_jr/skill_manager.py` (e.g., prioritize `Network Admin` for `device_ip` tasks).
        -   Update `agent_manager.py` to delegate tasks (e.g., `use_skill` to another agent).
        -   Enhance `middleware/memory.py` to store swarm interactions (`sender_agent_id`, `receiver_agent_id`).
        -   Example: `Ninja` (Network Admin) detects latency, delegates report generation to `Writer` agent, combining results autonomously.
    -   **Benefits**:
        -   Enables collaborative, agentic problem-solving.
        -   Supports cross-platform collaboration (e.g., iPhone, Windows).
        -   Enhances autonomy via task distribution.
2.  **Enhance Memory Management for Agentic Context**

    -   **Objective**: Optimize Qdrant vector storage for context-aware autonomy, enabling proactive goal inference and interaction pattern analysis.
    -   **Proposed Approach**:
        -   Update `middleware/memory.py` to embed interaction outcomes (e.g., `skill_troubleshooting_001.py` success) in Qdrant.
        -   Implement retrieval of similar interactions for goal inference (e.g., frequent `troubleshooting` → "optimize network").
        -   Add clustering in `hybrid_memory/utils/clustering.py` to identify patterns (e.g., recurring latency issues).
        -   Example: `Ninja` infers "reduce latency" goal from `memory.db`, runs `skill_packet_analysis_001.py` autonomously.
    -   **Benefits**:
        -   Supports proactive, context-driven actions.
        -   Enhances reasoning with historical data.
        -   Prepares for swarm context sharing.
3.  **Integrate Real-Time Environmental Sensing**

    -   **Objective**: Enable AutoNinja to adapt to network environments by integrating APIs (e.g., SNMP, NetFlow) for real-time data, supporting agentic adaptability.
    -   **Proposed Approach**:
        -   Add `app/network/api.py` for SNMP/NetFlow integration, polling devices for metrics (e.g., bandwidth, errors).
        -   Update `agent_manager.py` to trigger skills dynamically (e.g., `skill_network_monitoring_001.py` on traffic spikes).
        -   Store metrics in `memory.db` for pattern analysis.
        -   Example: `Ninja` detects packet loss via SNMP, runs `skill_traceroute_001.py` with `target_host`, adjusts `skill_network_configuration_001.py` commands.
    -   **Benefits**:
        -   Enables environment-driven autonomy.
        -   Supports dynamic script parameter adjustment.
        -   Enhances accuracy (~90-95%) with real-time data.
4.  **Implement Continuous Learning for Agentic Evolution**

    -   **Objective**: Enable AutoNinja to learn from interaction outcomes, refining skills and scripts autonomously, managed by Grok Jr.'s `learning_manager.py`.
    -   **Proposed Approach**:
        -   Add `app/learning/fine_tune.py` to score skill executions (e.g., `skill_security_management_001.py` success) using `memory.db` (`actual_output` vs. `target_output`).
        -   Update `agent_manager.py` to regenerate scripts for low-performing skills (e.g., update `skill_troubleshooting_001.py` if latency persists).
        -   Store learned weights in Qdrant for online/offline access.
        -   Example: `Ninja` refines `skill_network_scan_001.py` to prioritize ports after repeated failures, shares via swarm.
    -   **Benefits**:
        -   Drives autonomous skill improvement.
        -   Maintains ~90-95% accuracy through learning.
        -   Supports swarm-wide knowledge sharing.
5.  **Enhance Reasoning for Agentic Decision-Making**

    -   **Objective**: Enable AutoNinja to evaluate multiple skill options and make context-driven decisions, supporting agentic reasoning.
    -   **Proposed Approach**:
        -   Add `app/reasoning/decision_engine.py` for decision trees or Bayesian models, scoring skills (e.g., `skill_troubleshooting_001.py` vs. `skill_network_scan_001.py`) based on `memory.db` outcomes and network data.
        -   Update `agent_manager.py` to select skills dynamically (e.g., choose `skill_packet_analysis_001.py` for latency).
        -   Integrate with `ethics_manager.py` for constraints (e.g., avoid disabling firewalls).
        -   Example: `Ninja` evaluates latency issue, selects `skill_traceroute_001.py` over `skill_ping_001.py` based on recent failures, confirms with user.
    -   **Benefits**:
        -   Enhances proactive, accurate decisions (~90-95%).
        -   Supports complex, multi-step tasks.
        -   Ensures ethical alignment.

Challenges and Future Enhancements
----------------------------------

-   **Model Selection**: Configurable `MODEL_NAME`, but agentic features (swarm, reasoning) may require larger models, increasing resource needs. Optimize with efficient models (e.g., Gemma-3-1b-it).
-   **Scalability**: Swarm, real-time sensing, and learning increase compute demands. Use batched Qdrant queries, lightweight WebSocket protocols, and CUDA optimization.
-   **Safety**: Local processing ensures privacy; sandbox scripts and swarm communications to prevent malicious actions, with `ethics_manager.py` oversight.
-   **Offline TTS**: pyttsx3 quality lower than gTTS; explore Coqui TTS for offline agentic interactions.
-   **Latency**: Agentic reasoning and swarm coordination may exceed ~5-10s. Cache decisions and optimize API calls.
-   **Skill Duplicates**: Pending cleanup of duplicates (e.g., `Network Configuration`) to streamline agentic skill selection.
-   **Swarm Complexity**: Inter-agent communication requires robust conflict resolution, managed by Grok Jr. to balance autonomy and coordination.
