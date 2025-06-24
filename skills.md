I agree---using a category_task_001.py naming convention (e.g., math_addition_001.py) makes sense. It provides clarity, organizes scripts by purpose, and allows for incremental numbering to avoid overwrites. The plan looks solid, so let's proceed with the detailed implementation outline for integrating Script Generation and Execution into Auto Ninja as a separate module. I'll avoid writing full code as per your earlier preference but provide a precise, actionable blueprint for each component.

* * * * *

Detailed Implementation Blueprint

New Module: app/scripting/

This module will handle all script-related logic, integrating with the existing Auto Ninja framework.

Folder Structure

```
auto_ninja/
├── app/
│   ├── scripting/
│   │   ├── __init__.py
│   │   ├── manager.py
│   │   ├── utils.py
│   │   └── constants.py
│   ├── inference/
│   ├── hybrid_memory/
│   ├── middleware/
│   └── ... (existing folders)
├── scripts/                # New directory for script storage
├── .env
└── ...
```

* * * * *

1\. app/scripting/constants.py

Purpose: Define constants for the scripting module.

Blueprint:

-   Constants:

    -   SCRIPT_DIR = Path(__file__).parent.parent.parent / 'scripts': Directory for storing scripts.

    -   DEFAULT_LANGUAGE = 'python': Default script language.

    -   TRIGGERS = {'generate': 'generate code', 'execute': 'run code', 'update': 'update code', 'delete': 'delete code'}: Trigger phrases.

    -   FILE_PATTERN = r'(\w+)_(\w+)_(\d{3})\.py': Regex for parsing category, task, and number (e.g., math_addition_001.py).

-   Notes:

    -   Ensure SCRIPT_DIR exists on startup (create if missing).

    -   The regex will help validate and extract parts of filenames for execution, update, and delete actions.

* * * * *

2\. app/scripting/utils.py

Purpose: Provide helper functions for file handling and script management.

Blueprint:

-   Functions:

    -   save_script(file_name: str, content: str) -> bool:

        -   Write content to SCRIPT_DIR / file_name.

        -   Create SCRIPT_DIR if it doesn't exist.

        -   Return True on success, False on failure (e.g., permission error).

    -   load_script(file_name: str) -> str:

        -   Read and return content from SCRIPT_DIR / file_name.

        -   Raise FileNotFoundError if missing.

    -   delete_script(file_name: str) -> bool:

        -   Remove SCRIPT_DIR / file_name.

        -   Return True on success, False if file doesn't exist or fails.

    -   script_exists(file_name: str) -> bool:

        -   Check if SCRIPT_DIR / file_name exists.

    -   generate_file_name(category: str, task: str) -> str:

        -   Count existing files matching category_task_* in SCRIPT_DIR.

        -   Return next numbered name (e.g., math_addition_001.py if none exist, math_addition_002.py if 001 exists).

    -   extract_file_name(prompt: str) -> Optional[str]:

        -   Parse prompt for filename (e.g., "run code math_addition_001.py" -> math_addition_001.py).

        -   Return None if no valid filename found.

    -   validate_file_name(file_name: str) -> bool:

        -   Check if file_name matches FILE_PATTERN and ends with .py.

        -   Return True if valid, False otherwise.

-   Notes:

    -   Use pathlib.Path for cross-platform compatibility.

    -   Add basic error handling (e.g., log failures).

* * * * *

3\. app/scripting/manager.py

Purpose: Core logic for script generation, execution, updating, and deletion.

Blueprint:

-   Class: ScriptManager

-   Initialization:

    -   __init__(self, inference_engine: AutoNinjaInference, memory_manager: MemoryManager):

        -   Store references to inference_engine (for Grok/local model access) and memory_manager (for storage).

        -   Attributes: pending_action = None, pending_details = {} (to track confirmation state).

-   Methods:

    -   is_script_prompt(self, prompt: str) -> Optional[str]:

        -   Check if prompt starts with a trigger from TRIGGERS.

        -   Return action type (generate, execute, update, delete) or None if not a script prompt.

    -   request_confirmation(self, action: str, details: dict) -> str:

        -   Set self.pending_action = action and self.pending_details = details.

        -   Return confirmation message (e.g., "Do you want to generate this code?" for generate).

    -   confirm_action(self, user_response: str) -> Optional[str]:

        -   If user_response is affirmative (e.g., "yes", "y"), execute self.pending_action with self.pending_details.

        -   Clear pending_action and pending_details after execution.

        -   Return result (e.g., success message) or None if not confirmed.

    -   generate_script(self, prompt: str) -> str:

        -   Extract category/task from prompt (e.g., "generate code for a hello script" -> general_hello).

        -   Generate filename (e.g., general_hello_001.py).

        -   Use local model (or Grok if online) to create script content (e.g., print("Hello, World!")).

        -   Save script to filesystem.

        -   Create Tool object: {name: file_name, instructions: content, description: "Generated script"}.

        -   Store Tool in SQLite and link to current agent.

        -   Embed summary in Qdrant (e.g., "Generated a Python script to print 'Hello, World!'").

        -   Return "Code generated successfully!"

    -   execute_script(self, file_name: str) -> str:

        -   Validate file_name and check if it exists.

        -   Run script using subprocess.run(['python', str(SCRIPT_DIR / file_name)]).

        -   Return "Code executed successfully!" (or error message if fails).

    -   update_script(self, prompt: str, file_name: str) -> str:

        -   Validate file_name and check existence.

        -   Extract new content from prompt (e.g., "update code math_addition_001.py to print 'Sum: 5'" -> print("Sum: 5")).

        -   Update file in filesystem.

        -   Update Tool object's instructions in SQLite.

        -   Update Qdrant summary.

        -   Return "Code updated successfully!"

    -   delete_script(self, file_name: str) -> str:

        -   Validate file_name and check existence.

        -   Delete file from filesystem.

        -   Remove Tool from SQLite and its agent link.

        -   Remove summary from Qdrant.

        -   Return "Code deleted successfully!"

-   Notes:

    -   Use inference_engine.process() to generate/update script content, passing a refined prompt (e.g., "Write Python code to print 'Hello, World!'").

    -   Store Tool via memory_manager.memory.store_tool() and link via agent_tools table.

    -   Embed summaries using memory_manager.memory.store_interaction() with a dummy interaction.

* * * * *

4\. Integration with Existing Code

app/inference/engine.py

-   Update __init__:

    -   Add self.script_manager = ScriptManager(self, self.memory_manager).

-   Update process(self, message: str) -> str:

    -   Check if message is a script prompt:

        pseudo

        ```
        action = self.script_manager.is_script_prompt(message)
        if action:
            if self.script_manager.pending_action:
                return self.script_manager.confirm_action(message)
            else:
                details = {'prompt': message, 'file_name': extract_file_name(message)}  # Varies by action
                return self.script_manager.request_confirmation(action, details)
        ```

    -   If not a script prompt, proceed with existing logic.

-   Notes:

    -   This delegates script tasks to ScriptManager while keeping the inference flow intact.

app/middleware/memory.py

-   Update process(self, user_prompt: str, local_output: str, final_output: str) -> str:

    -   If final_output is a script success message, create a Tool and store it:

        pseudo

        ```
        if "successfully" in final_output:
            tool = Tool(name=file_name, instructions=script_content, description="Generated script")
            tool_id = self.memory.store_tool(tool)
            self.memory.store_agent_tool(self.current_agent.id, tool_id)
            summary = f"Generated a Python script: {file_name}"
            interaction = Interaction(user_prompt=user_prompt, actual_output=local_output, target_output=summary)
            self.memory.store_interaction(interaction)
        ```

-   Notes:

    -   This ties scripts to the agent and embeds summaries for context.

app/api/inference/endpoints.py

-   No changes needed; /predict handles all prompts naturally.

* * * * *

5\. Confirmation Mechanism

-   State Tracking: Use ScriptManager.pending_action and pending_details to track the action awaiting confirmation.

-   Flow:

    -   User: "generate code for a hello script"

    -   Agent: "Do you want to generate this code?"

    -   User: "Yes"

    -   Agent: "Code generated successfully!" (e.g., general_hello_001.py)

-   Affirmative Check: Recognize "yes", "y", "sure" (case-insensitive) as confirmation; otherwise, cancel.

* * * * *

Example Workflow

1.  Generate:

    -   Prompt: "generate code for a math addition script"

    -   Agent: "Do you want to generate this code?"

    -   User: "Yes"

    -   File: math_addition_001.py with print(2 + 3)

    -   Tool: Stored in SQLite

    -   Summary: Embedded in Qdrant

    -   Response: "Code generated successfully!"

2.  Execute:

    -   Prompt: "run code math_addition_001.py"

    -   Agent: "Do you want to execute math_addition_001.py?"

    -   User: "Yes"

    -   Response: "Code executed successfully!"

3.  Update:

    -   Prompt: "update code math_addition_001.py to print 'Sum: 5'"

    -   Agent: "Do you want to update math_addition_001.py?"

    -   User: "Yes"

    -   Response: "Code updated successfully!"

4.  Delete:

    -   Prompt: "delete code math_addition_001.py"

    -   Agent: "Do you want to delete math_addition_001.py?"

    -   User: "Yes"

    -   Response: "Code deleted successfully!"

* * * * *

Final Notes

-   Scalability: The category_task_001.py convention supports many scripts; consider a cleanup mechanism for old files later.

-   Error Handling: Add checks for file conflicts, invalid prompts, or execution failures.

-   Grok: Online, Grok can refine script content (e.g., add comments or error handling).

This blueprint should slot neatly into Auto Ninja. If you're happy with it, I can refine specific sections (e.g., ScriptManager methods) or assist with testing ideas. What's your next step?