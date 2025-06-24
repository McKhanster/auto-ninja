rm memory.db
(ninja-env) mcesel@fliegen:~/Documents/proj/autoninja$ python scripts/embed_intents.py
TTS_ENGINE set to gTTS
Enter the agent's name: Ninja
Enter the agent's role: Network Admin
INFO:sentence_transformers.SentenceTransformer:Load pretrained SentenceTransformer: all-MiniLM-L6-v2
INFO:app.shared:Initialized singleton SentenceTransformer on cuda
Batches: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 74.05it/s]
INFO:__main__:Embedded intent: exit
Batches: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 116.56it/s]
INFO:__main__:Embedded intent: use_speech
Batches: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 127.91it/s]
INFO:__main__:Embedded intent: use_text
Batches: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 150.87it/s]
INFO:__main__:Embedded intent: list_skills
Batches: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 147.01it/s]
INFO:__main__:Embedded intent: acquire_skill
Batches: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 134.64it/s]
INFO:__main__:Embedded intent: use_skill
Batches: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 101.94it/s]
INFO:__main__:Embedded intent: update_skill
Batches: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 102.17it/s]
INFO:__main__:Embedded intent: delete_skill
Batches: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 99.72it/s]
INFO:__main__:Embedded intent: acquire_next
Batches: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 96.03it/s]
INFO:__main__:Embedded intent: script_generate
Batches: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 95.88it/s]
INFO:__main__:Embedded intent: script_execute
Batches: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 85.50it/s]
INFO:__main__:Embedded intent: script_update
Batches: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 79.62it/s]
INFO:__main__:Embedded intent: script_delete
Batches: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 124.14it/s]
INFO:__main__:Embedded intent: other
(ninja-env) mcesel@fliegen:~/Documents/proj/autoninja$ sqlite3 memory.db "SELECT intent, length(embedding) FROM action_intents"
exit|1536
use_speech|1536
use_text|1536
list_skills|1536
acquire_skill|1536
use_skill|1536
update_skill|1536
delete_skill|1536
acquire_next|1536
script_generate|1536
script_execute|1536
script_update|1536
script_delete|1536
other|1536
(ninja-env) mcesel@fliegen:~/Documents/proj/autoninja$ docker stop qdrant
qdrant
(ninja-env) mcesel@fliegen:~/Documents/proj/autoninja$ python auto_ninja.py
TTS_ENGINE set to gTTS
Traceback (most recent call last):
  File "/home/mcesel/Documents/proj/ninja-env/lib/python3.12/site-packages/httpx/_transports/default.py", line 101, in map_httpcore_exceptions
    yield
  File "/home/mcesel/Documents/proj/ninja-env/lib/python3.12/site-packages/httpx/_transports/default.py", line 250, in handle_request
    resp = self._pool.handle_request(req)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/mcesel/Documents/proj/ninja-env/lib/python3.12/site-packages/httpcore/_sync/connection_pool.py", line 256, in handle_request
    raise exc from None
  File "/home/mcesel/Documents/proj/ninja-env/lib/python3.12/site-packages/httpcore/_sync/connection_pool.py", line 236, in handle_request
    response = connection.handle_request(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/mcesel/Documents/proj/ninja-env/lib/python3.12/site-packages/httpcore/_sync/connection.py", line 101, in handle_request
    raise exc
  File "/home/mcesel/Documents/proj/ninja-env/lib/python3.12/site-packages/httpcore/_sync/connection.py", line 78, in handle_request
    stream = self._connect(request)
             ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/mcesel/Documents/proj/ninja-env/lib/python3.12/site-packages/httpcore/_sync/connection.py", line 124, in _connect
    stream = self._network_backend.connect_tcp(**kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/mcesel/Documents/proj/ninja-env/lib/python3.12/site-packages/httpcore/_backends/sync.py", line 207, in connect_tcp
    with map_exceptions(exc_map):
  File "/usr/lib/python3.12/contextlib.py", line 158, in __exit__
    self.gen.throw(value)
  File "/home/mcesel/Documents/proj/ninja-env/lib/python3.12/site-packages/httpcore/_exceptions.py", line 14, in map_exceptions
    raise to_exc(exc) from exc
httpcore.ConnectError: [Errno 111] Connection refused

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/mcesel/Documents/proj/ninja-env/lib/python3.12/site-packages/qdrant_client/http/api_client.py", line 116, in send_inner
    response = self._client.send(request)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/mcesel/Documents/proj/ninja-env/lib/python3.12/site-packages/httpx/_client.py", line 914, in send
    response = self._send_handling_auth(
               ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/mcesel/Documents/proj/ninja-env/lib/python3.12/site-packages/httpx/_client.py", line 942, in _send_handling_auth
    response = self._send_handling_redirects(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/mcesel/Documents/proj/ninja-env/lib/python3.12/site-packages/httpx/_client.py", line 979, in _send_handling_redirects
    response = self._send_single_request(request)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/mcesel/Documents/proj/ninja-env/lib/python3.12/site-packages/httpx/_client.py", line 1014, in _send_single_request
    response = transport.handle_request(request)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/mcesel/Documents/proj/ninja-env/lib/python3.12/site-packages/httpx/_transports/default.py", line 249, in handle_request
    with map_httpcore_exceptions():
  File "/usr/lib/python3.12/contextlib.py", line 158, in __exit__
    self.gen.throw(value)
  File "/home/mcesel/Documents/proj/ninja-env/lib/python3.12/site-packages/httpx/_transports/default.py", line 118, in map_httpcore_exceptions
    raise mapped_exc(message) from exc
httpx.ConnectError: [Errno 111] Connection refused

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/mcesel/Documents/proj/autoninja/auto_ninja.py", line 7, in <module>
    from app.intent.intent_recognizer import IntentRecognizer
  File "/home/mcesel/Documents/proj/autoninja/app/intent/intent_recognizer.py", line 5, in <module>
    from ..shared import SingletonSentenceTransformer
  File "/home/mcesel/Documents/proj/autoninja/app/shared.py", line 78, in <module>
    inference_engine = SingletonInference.get_instance()
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/mcesel/Documents/proj/autoninja/app/shared.py", line 29, in get_instance
    cls._instance = AutoNinjaInference()
                    ^^^^^^^^^^^^^^^^^^^^
  File "/home/mcesel/Documents/proj/autoninja/app/inference/engine.py", line 30, in __init__
    self.memory_manager = MemoryManager(
                          ^^^^^^^^^^^^^^
  File "/home/mcesel/Documents/proj/autoninja/app/middleware/memory.py", line 15, in __init__
    self.memory.initialize()
  File "/home/mcesel/Documents/proj/autoninja/app/hybrid_memory/core.py", line 20, in initialize
    self.qdrant_store.initialize()
  File "/home/mcesel/Documents/proj/autoninja/app/hybrid_memory/qdrant_store.py", line 23, in initialize
    self.client.create_collection(
  File "/home/mcesel/Documents/proj/ninja-env/lib/python3.12/site-packages/qdrant_client/qdrant_client.py", line 2310, in create_collection
    return self._client.create_collection(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/mcesel/Documents/proj/ninja-env/lib/python3.12/site-packages/qdrant_client/qdrant_remote.py", line 2810, in create_collection
    result: Optional[bool] = self.http.collections_api.create_collection(
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/mcesel/Documents/proj/ninja-env/lib/python3.12/site-packages/qdrant_client/http/api/collections_api.py", line 294, in create_collection
    return self._build_for_create_collection(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/mcesel/Documents/proj/ninja-env/lib/python3.12/site-packages/qdrant_client/http/api/collections_api.py", line 96, in _build_for_create_collection
    return self.api_client.request(
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/mcesel/Documents/proj/ninja-env/lib/python3.12/site-packages/qdrant_client/http/api_client.py", line 89, in request
    return self.send(request, type_)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/mcesel/Documents/proj/ninja-env/lib/python3.12/site-packages/qdrant_client/http/api_client.py", line 106, in send
    response = self.middleware(request, self.send_inner)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/mcesel/Documents/proj/ninja-env/lib/python3.12/site-packages/qdrant_client/http/api_client.py", line 215, in __call__
    return call_next(request)
           ^^^^^^^^^^^^^^^^^^
  File "/home/mcesel/Documents/proj/ninja-env/lib/python3.12/site-packages/qdrant_client/http/api_client.py", line 118, in send_inner
    raise ResponseHandlingException(e)
qdrant_client.http.exceptions.ResponseHandlingException: [Errno 111] Connection refused
(ninja-env) mcesel@fliegen:~/Documents/proj/autoninja$ ./qdranit_init.sh
qdrant
qdrant
209d268f98dee016e791d3b643264a8be724b2d2d2e7c95a683ebff97d392d07
(ninja-env) mcesel@fliegen:~/Documents/proj/autoninja$ python auto_ninja.py
TTS_ENGINE set to gTTS
INFO:__main__:Using device: cuda (CUDA Available: True)
INFO:__main__:CUDA Device: NVIDIA GeForce RTX 3070 Laptop GPU, CUDA Version: 12.4
INFO:__main__:Starting interaction. Starting server and checking status...
INFO:__main__:Starting FastAPI server...
INFO:httpx:HTTP Request: GET http://localhost:6333 "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: GET http://localhost:6333/collections/interaction_summaries "HTTP/1.1 200 OK"
INFO:app.hybrid_memory.qdrant_store:Collection interaction_summaries already exists
INFO:httpx:HTTP Request: GET http://localhost:6333/collections/skill_summaries "HTTP/1.1 200 OK"
INFO:app.hybrid_memory.qdrant_store:Collection skill_summaries already exists
INFO:app.scripting.agent_manager:AgentManager initialized
INFO:app.main:Starting Auto Ninja with the following package versions:
INFO:app.main:torch: 2.6.0+cu124
INFO:app.main:transformers: 4.50.3
INFO:app.main:torchvision: 0.21.0+cu124
INFO:app.main:sentence-transformers: 3.4.1
INFO:app.main:numpy: 2.1.0
INFO:app.main:Pre-loading inference engine model...
INFO:app.main:Inference engine model pre-loaded
INFO:app.main:openai-whisper: 20240930
INFO:app.main:gtts: 2.5.4
INFO:app.main:pyttsx3: 2.98
INFO:app.main:pyaudio: 0.2.14
INFO:app.main:pydub: 0.25.1
INFO:     Started server process [138286]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
INFO:     127.0.0.1:34408 - "GET /inference/health HTTP/1.1" 200 OK
INFO:__main__:Server responded: {'status': 'healthy'}
INFO:__main__:Server is healthy, proceeding with full functionality.
INFO:app.scripting.agent_manager:AgentManager initialized
INFO:sentence_transformers.SentenceTransformer:Load pretrained SentenceTransformer: all-MiniLM-L6-v2
INFO:app.shared:Initialized singleton SentenceTransformer on cuda
INFO:app.intent.intent_recognizer:Initialized IntentRecognizer with shared SentenceTransformer
INFO:app.middleware.memory:Intent recognizer set for MemoryManager
Using agent Ninja (Role: Network Admin). Type your command (e.g., 'list skills', 'exit'): 
Commands: 'list skills', 'acquire skill <name>, <description>', 'use skill <id>, <task>', ...
INFO:app.speech.tts:Initializing TTS with engine: gTTS
> list skills
Batches: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  8.41it/s]
INFO:app.intent.intent_recognizer:Embedding intent: other, score: 0.0
INFO:app.inference.engine:Raw Grok response in _predict_grok: {'id': '68436b34-73ba-42b5-8652-d7a856363477', 'object': 'chat.completion', 'created': 1744830937, 'model': 'grok-3-mini-beta', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': '{\n  "intent": "list_skills",\n  "parameters": {}\n}', 'reasoning_content': 'First, the user\'s input is: "list skills"\n\nMy role is to act as the Local Agent, which needs to classify the intent based on the user\'s message. The possible intents are: exit, use_speech, use_text, list_skills, acquire_skill, use_skill, update_skill, delete_skill, acquire_next, script_generate, script_execute, script_update, script_delete, other.\n\nFrom the list, "list skills" clearly matches "list_skills".\n\nI need to return a JSON object with two keys: \'intent\' and \'parameters\'. So, something like: {\'intent\': \'list_skills\', \'parameters\': {...}}\n\nNow, for parameters: The instructions say to extract parameters like skill_id, task, or file_name if applicable. In the user\'s input "list skills", there are no additional details provided. It\'s just a straightforward command to list skills. So, probably, no parameters are needed here.\n\nLet me double-check the intents: For list_skills, it might not require any parameters. The examples given are skill_id, task, or file_name, which are more relevant to other intents like acquire_skill or use_skill.\n\nTherefore, for this intent, \'parameters\' could be an empty object, like {}.\n\nFinally, the response I need to provide should be a coherent, standalone response that combines the user\'s input and the local agent\'s response. But the instructions say: "Please provide a coherent, standalone response combining the user\'s input and the local agent\'s response, without repeating the prompt or labels."\n\nRe-reading that: "Please provide a coherent, standalone response combining the user\'s input and the local agent\'s response, without repeating the prompt or labels."\n\nIn the prompt, it says: "User: list skills Local Agent: Classify intent: ... Return JSON: {\'intent\': \'...\', \'parameters\': {...}}."\n\nI think I need to output just the JSON as the local agent\'s response, but make it standalone.\n\nThe full instruction is: "Please provide a coherent, standalone response combining the user\'s input and the local agent\'s response, without repeating the prompt or labels."\n\nBut in context, I am the AI responding as the Local Agent, so I should just output the JSON.\n\nThe very end: "Human: Agent Role: Network Admin User: list skills Local Agent: Classify intent: ... Please provide a coherent, standalone response..."\n\nI need to simulate the Local Agent\'s output.\n\nSo, my final output should be the JSON object.\n\nTo make it coherent and standalone, perhaps just the JSON.\n\nFor example: {\'intent\': \'list_skills\', \'parameters\': {}}\n\nThat seems right.\n\nIs there any parameter? The user said "list skills", which might imply listing all skills, so no specific parameters.\n\nAlright, let\'s confirm.', 'refusal': None}, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 118, 'completion_tokens': 17, 'total_tokens': 692, 'prompt_tokens_details': {'text_tokens': 118, 'audio_tokens': 0, 'image_tokens': 0, 'cached_tokens': 0}, 'completion_tokens_details': {'reasoning_tokens': 557, 'audio_tokens': 0, 'accepted_prediction_tokens': 0, 'rejected_prediction_tokens': 0}}, 'system_fingerprint': 'fp_d133ae3397'}
INFO:app.intent.intent_recognizer:Grok intent: list_skills
INFO:__main__:Processed command: list skills, Intent: list_skills, Params: {}
INFO:app.scripting.agent_manager:Listed 34 skills for agent 1
INFO:__main__:Skills for agent 1: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
ID: 1, Name: Network Configuration, Acquired: False, Description: The ability to set up and manage network devices such as routers, switches, and firewalls, including configuring IP addresses, VLANs, and routing protocols to ensure proper data flow and connectivity.
ID: 2, Name: Troubleshooting, Acquired: False, Description: Identifying and resolving network issues, such as connectivity problems, latency, or outages, using diagnostic tools and logical problem-solving techniques.
ID: 3, Name: ping, Acquired: False, Description: Test network connectivity
ID: 4, Name: traceroute, Acquired: False, Description: Trace the route a packet takes
ID: 5, Name: Security Management, Acquired: False, Description: Implementing and maintaining network security measures, including firewalls, intrusion detection systems, and access controls to protect against threats like unauthorized access or malware.
ID: 6, Name: Risk Assessment, Acquired: False, Description: Identifying, analyzing, and evaluating potential security risks.
ID: 7, Name: Network Monitoring, Acquired: False, Description: Continuously monitoring network performance, traffic, and health using tools to detect anomalies, optimize resources, and prevent downtime.
ID: 8, Name: Protocol Understanding, Acquired: False, Description: Understanding of network protocols such as TCP/IP, DNS, and DHCP to monitor and analyze network traffic and performance in a Network Admin role.
ID: 9, Name: Traceroute Analysis, Acquired: False, Description: Using traceroute to identify network paths, detect latency issues, and troubleshoot connectivity problems for effective network monitoring.
ID: 10, Name: Traffic Monitoring with Wireshark, Acquired: False, Description: Utilizing Wireshark to capture and analyze network traffic, identifying anomalies, bottlenecks, and security threats in network monitoring tasks.
ID: 11, Name: System Monitoring with Nagios, Acquired: False, Description: Employing Nagios for real-time monitoring of network devices, services, and performance metrics to ensure reliability and quick issue detection.
ID: 12, Name: Performance Analysis with SolarWinds, Acquired: False, Description: Using SolarWinds to monitor network performance, bandwidth utilization, and device health, enabling proactive maintenance in a Network Admin context.
ID: 13, Name: Automation Scripting, Acquired: False, Description: Writing scripts to automate repetitive network tasks, such as device configuration, backups, or monitoring alerts, to improve efficiency and reduce errors.
ID: 14, Name: Python, Acquired: False, Description: Proficiency in Python scripting for network automation, including libraries like Netmiko and Paramiko for interacting with network devices.
ID: 15, Name: Network Automation Frameworks, Acquired: False, Description: Experience with frameworks like Ansible for configuration management and automation, or Python-based scripting for network tasks.
ID: 16, Name: Network Configuration, Acquired: False, Description: The ability to set up and manage network devices such as routers, switches, and firewalls, including configuring IP addresses, VLANs, and routing protocols to ensure proper data flow and connectivity.
Parameters:
- intents (list[str], required): List of user intents from interactions to analyze for gaps in Network Configuration, helping identify areas like IP setup or VLAN management that need improvement.
- role (str, optional): The agent's role to filter and prioritize relevant intents, ensuring analysis adapts to contexts like Network Admin for routing and security tasks. (default: Network Admin)
ID: 17, Name: Network Configuration, Acquired: True, Description: The ability to set up and manage network devices such as routers, switches, and firewalls, including configuring IP addresses, VLANs, and routing protocols to ensure proper data flow and connectivity.
ID: 18, Name: Troubleshooting, Acquired: False, Description: Identifying and resolving network issues, such as connectivity problems, latency, or outages, using diagnostic tools and logical problem-solving techniques.
Parameters:
- intent (str, required): The user's intent string, such as a description of the network issue (e.g., 'connectivity problems'). Used to query and analyze similar interactions.
- data_path (str, optional): Optional path to a JSON file containing additional interaction data for enhanced analysis, pulled from memory.db via memory_manager.
- similarity_threshold (float, optional): Optional threshold (0.0 to 1.0) for determining similarity between intents during clustering and analysis. (default: 0.7)
ID: 19, Name: Troubleshooting, Acquired: True, Description: Identifying and resolving network issues, such as connectivity problems, latency, or outages, using diagnostic tools and logical problem-solving techniques.
Parameters:
- target_ip (str, required): The IP address or network range to scan for troubleshooting, e.g., '192.168.1.0/24'
ID: 20, Name: ping, Acquired: True, Description: Test network connectivity
ID: 21, Name: traceroute, Acquired: False, Description: Trace the route a packet takes
Parameters:
- target (str, required): The target network address (e.g., IP or hostname) to simulate tracing the route for, essential for core traceroute functionality.
- intents (list[str], optional): A list of user intents (e.g., from interaction history) for adaptive analysis, helping identify learning gaps related to network tasks like routing. (default: [])
ID: 22, Name: traceroute, Acquired: True, Description: Trace the route a packet takes
Parameters:
- target_host (str, required): The hostname or IP address to perform the traceroute on.
ID: 23, Name: Security Management, Acquired: False, Description: Implementing and maintaining network security measures, including firewalls, intrusion detection systems, and access controls to protect against threats like unauthorized access or malware.
Parameters:
- interactions (list[object[Interaction]], required): A list of past user interactions for security analysis, including details like intent and text to identify gaps in network security measures.
  Fields:
  - id (str): Unique identifier for the interaction.
  - text (str): The text content of the interaction, such as 'Implement firewall rules'.
  - intent (str): The intent label, e.g., 'Security Management', to filter and analyze.
- num_clusters (int, optional): Optional number of clusters for ML analysis to group similar security intents. (default: 2)
ID: 24, Name: Security Management, Acquired: True, Description: Implementing and maintaining network security measures, including firewalls, intrusion detection systems, and access controls to protect against threats like unauthorized access or malware.
Parameters:
- target_ip (str, required): The IP address of the target host to scan for vulnerabilities.
- ports (str, optional): A string specifying the port range to scan, e.g., '1-100' for ports 1 through 100. (default: 1-1024)
ID: 25, Name: Risk Assessment, Acquired: False, Description: Identifying, analyzing, and evaluating potential security risks.
Parameters:
- input_data (object[InputData], required): JSON object containing parameters for querying and analyzing interactions, such as a query string to fetch relevant data from memory.db (e.g., for identifying security risks based on intents).
  Fields:
  - query (str): A string used to query similar interactions from memory.db, such as keywords or patterns related to security risks.
  - data_source (str): Optional string indicating the source of data, e.g., 'database', to specify where interactions are retrieved from.
ID: 26, Name: Risk Assessment, Acquired: True, Description: Identifying, analyzing, and evaluating potential security risks.
Parameters:
- risk_data (object[dict], required): A dictionary containing risk information, including details like risk ID, description, severity, impact, and mitigation actions.
  Fields:
  - risk_id (str): A unique identifier for the risk.
  - risk_description (str): A detailed description of the risk.
  - risk_severity (str): The severity level of the risk (e.g., 'high', 'moderate', 'low').
  - risk_impact (str): The potential impact of the risk (e.g., 'high', 'moderate', 'low').
  - risk_mitigation_actions (str): Recommended actions to mitigate the risk (e.g., 'patching, hardening').
ID: 27, Name: Network Monitoring, Acquired: False, Description: Continuously monitoring network performance, traffic, and health using tools to detect anomalies, optimize resources, and prevent downtime.
Parameters:
- data (list[str], required): A list of interaction IDs or intent strings from memory.db, used to query and filter relevant network monitoring data for the agent.
- max_clusters (int, optional): The number of clusters to use in intent analysis for detecting anomalies in network performance and traffic patterns. (default: 3)
ID: 28, Name: Network Monitoring, Acquired: True, Description: Continuously monitoring network performance, traffic, and health using tools to detect anomalies, optimize resources, and prevent downtime.
Parameters:
- device_ip (str, required): The IP address of the network device to monitor for performance and health.
- check_interval (int, optional): The interval in seconds between each monitoring check. (default: 60)
ID: 29, Name: Protocol Understanding, Acquired: False, Description: Understanding of network protocols such as TCP/IP, DNS, and DHCP to monitor and analyze network traffic and performance in a Network Admin role.
Parameters:
- data (list[str], required): List of user intents or interaction data from memory.db, used to fetch and analyze similar interactions related to network protocols.
ID: 30, Name: Protocol Understanding, Acquired: True, Description: Understanding of network protocols such as TCP/IP, DNS, and DHCP to monitor and analyze network traffic and performance in a Network Admin role.
Parameters:
- protocols (list[str], required): List of network protocols to monitor and analyze, such as ['TCP/IP', 'DNS', 'DHCP']. (default: ['TCP/IP', 'DNS', 'DHCP'])
- interface (str, optional): The network interface to use for monitoring, such as 'eth0'.
- duration (int, optional): The duration in seconds for the monitoring session. (default: 60)
ID: 31, Name: Traceroute Analysis, Acquired: False, Description: Using traceroute to identify network paths, detect latency issues, and troubleshoot connectivity problems for effective network monitoring.
Parameters:
- targets (list[str], required): List of hostnames or IP addresses to analyze for traceroute paths, derived from user intents in memory.db for network troubleshooting.
ID: 32, Name: Traceroute Analysis, Acquired: True, Description: Using traceroute to identify network paths, detect latency issues, and troubleshoot connectivity problems for effective network monitoring.
Parameters:
- traceroute_command (str, required): The full command string to execute traceroute, such as 'traceroute google.com', which includes the target host or IP.
ID: 33, Name: Traffic Monitoring with Wireshark, Acquired: False, Description: Utilizing Wireshark to capture and analyze network traffic, identifying anomalies, bottlenecks, and security threats in network monitoring tasks.
Parameters:
- intent_filter (list[str], required): A list of strings representing keywords or phrases to filter user intents from memory.db, focusing on network-related interactions for the Network Admin role.
- num_clusters (int, optional): An integer specifying the number of clusters for intent analysis using ML, to group similar intents and detect patterns or gaps in learning. (default: 3)
ID: 34, Name: Traffic Monitoring with Wireshark, Acquired: True, Description: Utilizing Wireshark to capture and analyze network traffic, identifying anomalies, bottlenecks, and security threats in network monitoring tasks.
> exit
Batches: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 44.28it/s]
INFO:app.intent.intent_recognizer:Embedding intent: exit, score: 0.8133643865585327
INFO:__main__:Processed command: exit, Intent: exit, Params: {}
(ninja-env) mcesel@fliegen:~/Documents/proj/autoninja$ ls scripts/
embed_intents.py                     skill_risk_assessment_001.py      skill_traffic_monitoring_with_wireshark_001.py
skill_network_configuration_001.py   skill_security_management_001.py  skill_troubleshooting_001.py
skill_network_monitoring_001.py      skill_traceroute_001.py
skill_protocol_understanding_001.py  skill_traceroute_analysis_001.py