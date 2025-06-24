
import re
import os
import json
from pathlib import Path
from .constants import SCRIPT_DIR, FILE_PATTERN
from ..inference.engine import AutoNinjaInference
from ..config.constants import BUILT_IN_PACKAGES

def generate_script_content(inference_engine: AutoNinjaInference, prompt: str, skill_name: str = None, agent_name: str = "Agent", agent_role: str = "General", skill_description: str = "") -> tuple[str, list]:
    """Generate Python script content and parameters from a prompt."""
    code_prompt = f"""
You are generating Python code for a skill in an adaptive learning system for an agent name {agent_name} with the role {agent_role}, accessible via `memory_manager.get_context()` (returns e.g., "Agent Role: {agent_role}"). The skill is: `{skill_name}` (description: `{skill_description}`). The code must:
1. Implement the skill’s core functionality (e.g., for `Skill Assessment`, analyze intent frequency to identify gaps).
2. Integrate with `memory_manager` (access `memory.db` via `memory_manager.get_interaction`, `get_similar_interactions`, or SQL queries to `interactions.intent`).
3. Use `sklearn` or `pandas` for ML or data analysis (e.g., clustering intents, predicting skills).
4. Accept 1-3 parameters via `sys.argv`, parsed as JSON (e.g., `sys.argv[1] = '[\"data\"]'` for lists, `{{\"key\": \"value\"}}` for objects).
5. Validate parameters, raising `ValueError` for invalid/missing inputs.
6. Return JSON output (e.g., `{{\"result\": \"value\"}}`) with skill-specific keys (e.g., `next_skill` for `Adaptive Learning`).
7. Avoid trivial code (no `print`-only logic); implement at least one function with ML or data-driven logic.
8. Limit to 100-200 lines, with clear comments and robust error handling.
9. Use `memory_manager.get_context()` to adapt to the current role dynamically (e.g., prioritize intents matching role keywords).
10. Handle `memory_manager` failures (e.g., empty `interactions` return `{{\"error\": \"No data\"}}`).

Return JSON:
{{
  "code": "Python code with imports, validation, and skill logic",
  "parameters": [
    {{
      "name": "param_name",
      "type": "str|int|float|bool|list[str]|list[int]|list[float]|list[bool]|object[class_name]",
      "description": "Brief description (20-50 words, e.g., 'User intent history from memory.db')",
      "required": bool,
      "default": "value or null for primitives, [] for lists, null for objects",
      "fields": [
        {{"name": "field_name", "type": "str|int|float|bool", "description": "Field description (e.g., 'Intent label')"}}
      ]
    }}
  ]
}}
"""
    # Retry logic: 2 attempts
    for attempt in range(2):
        response = inference_engine.process(code_prompt)
        try:
            data = robust_parse_json(response)
            code = data.get("code", "")
            parameters = data.get("parameters", [])
            # Validate non-trivial code
            if code.strip() and any(kw in code.lower() for kw in ["def", "import", "return"]) and "print('Script for" not in code:
                return code.strip(), parameters
        except ValueError:
            if attempt == 1:  # Second attempt failed
                break
    # Fallback template
    task = sanitize_task(skill_name or "task")
    code = f"""
import sys
import json
import pandas as pd
from app.middleware.memory import MemoryManager

def {task}(history):
    \"\"\"Placeholder for {skill_name or 'task'} (agent: {agent_name}, role: {agent_role}).\"\"\"
    try:
        df = pd.DataFrame(history)
        result = {{'intents': df['intent'].value_counts().to_dict()}}
    except Exception as e:
        result = {{'error': f'Failed to process history: {{str(e)}}'}}
    return result

if __name__ == '__main__':
    try:
        history = json.loads(sys.argv[1])
        if not isinstance(history, list):
            raise ValueError('history must be a list of dicts')
        result = {task}(history)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({{'error': str(e)}}))
"""
    parameters = [
        {
            "name": "history",
            "type": "list[dict]",
            "description": f"Intent history from memory.db for {skill_name or 'task'} analysis",
            "required": True,
            "default": [],
            "fields": [
                {"name": "intent", "type": "str", "description": "Intent label"},
                {"name": "user_prompt", "type": "str", "description": "User input"}
            ]
        }
    ]
    return code.strip(), parameters

def save_script(file_name: str, content: str) -> bool:
    try:
        SCRIPT_DIR.mkdir(exist_ok=True)
        with open(SCRIPT_DIR / file_name, "w") as f:
            f.write(content)
        return True
    except Exception:
        return False

def load_script(file_name: str) -> str:
    if not script_exists(file_name):
        raise FileNotFoundError(f"Script {file_name} not found")
    with open(SCRIPT_DIR / file_name, "r") as f:
        return f.read()

def delete_script(file_name: str) -> bool:
    if not script_exists(file_name):
        return False
    os.remove(SCRIPT_DIR / file_name)
    return True

def script_exists(file_name: str) -> bool:
    return (SCRIPT_DIR / file_name).exists()

def generate_file_name(category: str, task: str) -> str:
    """Generate a sanitized script file name in the format category_task_XXX.py."""
    # Sanitize category
    category = category.lower().strip()
    if category not in ["skill", "tool"]:
        category = "skill"  # Default to skill if invalid
    # Sanitize task
    task = sanitize_task(task)
    # Generate unique suffix
    pattern = f"{category}_{task}_\\d{{3}}\\.py"
    existing_files = [f.name for f in SCRIPT_DIR.glob("*.py") if re.match(pattern, f.name)]
    next_num = len(existing_files) + 1
    return f"{category}_{task}_{next_num:03d}.py"

def sanitize_task(task: str) -> str:
    """Sanitize task name for file naming."""
    if not task:
        return "task"
    # Remove invalid characters, prefixes, and normalize
    task = task.lower().strip()
    task = re.sub(r'[^\w\s-]', '', task)  # Remove special chars except spaces, hyphens
    task = re.sub(r'^(def_|import_|this_|```python_|```)', '', task)  # Remove prefixes
    task = re.sub(r'\s+', '_', task)  # Spaces to underscores
    task = task[:50]  # Max 50 chars
    return task or "task"

def extract_file_name(prompt: str) -> str | None:
    match = re.search(r"(?:run|update|delete)\s+code\s+([a-zA-Z]+_[a-zA-Z0-9]+_\d{3}\.py)", prompt, re.IGNORECASE)
    return match.group(1) if match else None

def validate_file_name(file_name: str) -> bool:
    return bool(re.match(FILE_PATTERN, file_name))

def infer_dependencies(code: str, inference_engine: AutoNinjaInference) -> tuple[list[str], list[str]]:
    imports = []
    for line in code.splitlines():
        line = line.strip()
        if line.startswith("import "):
            module = line.split()[1].split(".")[0]
            imports.append(module)
        elif line.startswith("from "):
            parts = line.split()
            if len(parts) > 1:
                module = parts[1].split(".")[0]
                imports.append(module)
    imports = list(set(imports) - set(BUILT_IN_PACKAGES))
    
    if not imports:
        return [], []
    
    prompt = (
        f"For Python imports: {', '.join(imports)}, provide exact pip-installable package names and system dependencies "
        "as JSON: {{'pip_packages': list[str], 'system_deps': list[str]}}. Exclude standard library modules."
    )
    response = inference_engine.process(prompt)
    try:
        data = robust_parse_json(response)
        return data.get("pip_packages", imports), data.get("system_deps", [])
    except ValueError:
        return imports, []

def robust_parse_json(raw_response: str) -> dict:
    json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
    json_str = json_match.group(0) if json_match else raw_response
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        json_str += '}' if not json_str.endswith('}') else ''
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON response")
# # app/scripting/utils.py
# import re
# import os
# from pathlib import Path
# from .constants import SCRIPT_DIR, FILE_PATTERN
# from ..inference.engine import AutoNinjaInference
# import json
# from ..config.constants import BUILT_IN_PACKAGES

# def generate_script_content(inference_engine: AutoNinjaInference, prompt: str, skill_name: str = None) -> tuple[str, list]:
#     """Generate Python script content and parameters from a prompt."""
#     code_prompt = f"""
# Generate Python code for: {prompt}. Use sys.argv for parameters. Return JSON:
# {{
#   "code": "Python code with sys.argv handling",
#   "parameters": [
#     {{
#       "name": "param_name",
#       "type": "str|int|float|bool|list[str]|list[int]|list[float]|list[bool]|object[class_name]",
#       "description": "Brief description (20-50 words)",
#       "required": bool,
#       "default": "value or null for primitives, [] for lists, null for objects",
#       "fields": [
#         {{"name": "field_name", "type": "str|int|float|bool", "description": "Field description"}}
#       ]
#     }}
#   ]
# }}
# Ensure code validates parameters, raising clear errors for missing/invalid inputs. For lists, use JSON parsing (e.g., sys.argv[1] = '[\"a\", \"b\"]'). For objects, parse JSON and validate fields (max 5). Limit to 2-3 parameters for simplicity.
# """
#     response = inference_engine.process(code_prompt)
#     try:
#         data = robust_parse_json(response)
#         code = data.get("code", "")
#         parameters = data.get("parameters", [])
#         if not code.strip() or not any(kw in code.lower() for kw in ["print", "def", "import", "return"]):
#             code = f"import sys\nprint('Script for {skill_name or 'task'} executed with:', sys.argv[1] if len(sys.argv) > 1 else 'no argument')"
#             parameters = []
#         return code.strip(), parameters
#     except ValueError:
#         code = f"import sys\nprint('Script for {skill_name or 'task'} executed with:', sys.argv[1] if len(sys.argv) > 1 else 'no argument')"
#         return code.strip(), []

# # Rest of the file unchanged (save_script, load_script, etc.)
# def save_script(file_name: str, content: str) -> bool:
#     try:
#         SCRIPT_DIR.mkdir(exist_ok=True)
#         with open(SCRIPT_DIR / file_name, "w") as f:
#             f.write(content)
#         return True
#     except Exception:
#         return False

# def load_script(file_name: str) -> str:
#     if not script_exists(file_name):
#         raise FileNotFoundError(f"Script {file_name} not found")
#     with open(SCRIPT_DIR / file_name, "r") as f:
#         return f.read()

# def delete_script(file_name: str) -> bool:
#     if not script_exists(file_name):
#         return False
#     os.remove(SCRIPT_DIR / file_name)
#     return True

# def script_exists(file_name: str) -> bool:
#     return (SCRIPT_DIR / file_name).exists()

# def generate_file_name(category: str, task: str) -> str:
#     pattern = f"{category}_{task}_\\d{{3}}\\.py"
#     existing_files = [f.name for f in SCRIPT_DIR.glob("*.py") if re.match(pattern, f.name)]
#     next_num = len(existing_files) + 1
#     return f"{category}_{task}_{next_num:03d}.py"

# def extract_file_name(prompt: str) -> str | None:
#     match = re.search(r"(?:run|update|delete)\s+code\s+([a-zA-Z]+_[a-zA-Z0-9]+_\d{3}\.py)", prompt, re.IGNORECASE)
#     return match.group(1) if match else None

# def validate_file_name(file_name: str) -> bool:
#     return bool(re.match(FILE_PATTERN, file_name))

# def infer_dependencies(code: str, inference_engine: AutoNinjaInference) -> tuple[list[str], list[str]]:
#     imports = []
#     for line in code.splitlines():
#         line = line.strip()
#         if line.startswith("import "):
#             module = line.split()[1].split(".")[0]
#             imports.append(module)
#         elif line.startswith("from "):
#             parts = line.split()
#             if len(parts) > 1:
#                 module = parts[1].split(".")[0]
#                 imports.append(module)
#     imports = list(set(imports) - set(BUILT_IN_PACKAGES))
    
#     if not imports:
#         return [], []
    
#     prompt = (
#         f"For Python imports: {', '.join(imports)}, provide exact pip-installable package names and system dependencies "
#         "as JSON: {{'pip_packages': list[str], 'system_deps': list[str]}}. Exclude standard library modules."
#     )
#     response = inference_engine.process(prompt)
#     try:
#         data = robust_parse_json(response)
#         return data.get("pip_packages", imports), data.get("system_deps", [])
#     except ValueError:
#         return imports, []

# def robust_parse_json(raw_response: str) -> dict:
#     json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
#     json_str = json_match.group(0) if json_match else raw_response
#     try:
#         return json.loads(json_str)
#     except json.JSONDecodeError:
#         json_str += '}' if not json_str.endswith('}') else ''
#         return json.loads(json_str)