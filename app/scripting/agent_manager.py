from datetime import datetime
import getpass
import json
import os
import subprocess
from typing import List, Optional, Dict
from .constants import TRIGGERS, SCRIPT_DIR
from .utils import (
    save_script, load_script, delete_script, script_exists,
    generate_file_name, extract_file_name, validate_file_name,
    generate_script_content, infer_dependencies, robust_parse_json
)
from ..inference.engine import AutoNinjaInference
from ..middleware.memory import MemoryManager
from ..models.tool import Tool
from ..models.skill import Skill
from ..hybrid_memory.utils.embedding import generate_embedding
import logging
import re
import glob

logger = logging.getLogger(__name__)

class AgentManager:
    def __init__(self, inference_engine: AutoNinjaInference, memory_manager: MemoryManager):
        self.inference_engine = inference_engine
        self.memory_manager = memory_manager
        self.pending_action: Optional[str] = None
        self.pending_details: Dict = {}
        logger.info("AgentManager initialized")

    def is_script_prompt(self, prompt: str) -> Optional[str]:
        prompt_lower = prompt.lower()
        for action, trigger in TRIGGERS.items():
            if trigger in prompt_lower:
                logger.info(f"Detected script prompt: {action}")
                return action
        return None

    def request_confirmation(self, action: str, details: dict) -> str:
        self.pending_action = action
        self.pending_details = details
        action_msg = "this code" if action == "generate" else f"{details.get('file_name', 'the code')}"
        return f"Do you want to {action} {action_msg}?"

    def confirm_action(self, user_response: str) -> Optional[str]:
        if user_response.lower() in ["yes", "y", "sure"]:
            action = self.pending_action
            details = self.pending_details
            self.pending_action = None
            self.pending_details = {}
            if action == "generate":
                return self.generate_script(details["prompt"], skill_name=details.get("skill_name"))
            elif action == "execute":
                return self.execute_script(details["file_name"])
            elif action == "update":
                return self.update_script(details["prompt"], details["file_name"])
            elif action == "delete":
                return self.delete_script(details["file_name"])
        self.pending_action = None
        self.pending_details = {}
        return "Action cancelled."

    def generate_script(self, prompt: str, skill_name: str = None) -> str:
        try:
            agent = self.memory_manager.current_agent
            agent_name = agent.name if agent else "Agent"
            agent_role = agent.role if agent else "General"
            skill_description = ""
            existing_file_name = None
            existing_skill = None
            if skill_name:
                for skill in self.list_skills():
                    if skill.name.lower() == skill_name.lower():
                        skill_description = skill.description
                        existing_skill = skill
                        if skill.instructions.get("type") == "script":
                            existing_file_name = os.path.basename(skill.instructions.get("path", ""))
                        break
            content, parameters = generate_script_content(
                self.inference_engine,
                prompt,
                skill_name=skill_name,
                agent_name=agent_name,
                agent_role=agent_role,
                skill_description=skill_description
            )
            # Check if script needs update (description or parameters changed)
            update_needed = True
            if existing_file_name and existing_skill:
                existing_content = load_script(existing_file_name)
                existing_params = existing_skill.instructions.get("parameters", [])
                if (existing_content == content and 
                    json.dumps(existing_params, sort_keys=True) == json.dumps(parameters, sort_keys=True) and
                    existing_skill.description == skill_description):
                    update_needed = False
                    logger.info(f"Script {existing_file_name} unchanged for skill {skill_name}")
            
            if not update_needed:
                return f"Code unchanged for skill: {skill_name}, File: {existing_file_name}"

            category = "skill" if skill_name else "tool"
            task = skill_name or prompt.lower().replace("generate code", "").strip() or "script"
            file_name = existing_file_name or generate_file_name(category, task)
            if not save_script(file_name, content):
                logger.error(f"Failed to save script: {file_name}")
                return f"Error: Failed to save script {file_name}"

            tool = Tool(
                agent_id=self.memory_manager.current_agent.id,
                name=file_name,
                instructions={"type": "script", "path": f"scripts/{file_name}", "content": content},
                parameters=parameters,
                description=f"Generated script for {task}"
            )
            tool_id = self.memory_manager.memory.store_tool(tool)
            self.memory_manager.memory.sqlite_store.cursor.execute(
                "INSERT OR IGNORE INTO agent_tools (agent_id, tool_id) VALUES (?, ?)",
                (self.memory_manager.current_agent.id, tool_id)
            )
            self.memory_manager.memory.sqlite_store.conn.commit()

            if skill_name and existing_skill:
                existing_skill.instructions = {
                    "type": "script",
                    "path": f"scripts/{file_name}",
                    "venv": "venvs/auto_ninja_venv",
                    "parameters": parameters
                }
                existing_skill.script_id = tool_id
                self.memory_manager.memory.store_skill(existing_skill)
                logger.info(f"Updated skill {skill_name} with script {file_name}")
            elif skill_name:
                for skill in self.list_skills():
                    if skill.name.lower() == skill_name.lower():
                        skill.instructions = {
                            "type": "script",
                            "path": f"scripts/{file_name}",
                            "venv": "venvs/auto_ninja_venv",
                            "parameters": parameters
                        }
                        skill.script_id = tool_id
                        self.memory_manager.memory.store_skill(skill)
                        logger.info(f"Linked skill {skill_name} to script {file_name}")
                        break

            summary = f"Generated Python script: {file_name} with parameters: {json.dumps(parameters)}"
            self.memory_manager.process(prompt, content, summary)
            logger.info(f"Script generated: {file_name}")
            return f"Code generated successfully! File: {file_name}"
        except Exception as e:
            logger.error(f"Failed to generate script: {str(e)}")
            return f"Error generating script: {str(e)}"

    def execute_script(self, command: str) -> str:
        try:
            parts = command.split(" ", 1)
            file_name = parts[0].replace("scripts/", "")
            arg = parts[1] if len(parts) > 1 else ""
            script_path = SCRIPT_DIR / file_name

            if not script_path.exists():
                logger.error(f"Script not found: {file_name}")
                return f"Invalid or missing script file: {file_name}"

            tools = self.memory_manager.memory.get_tools_for_agent(self.memory_manager.current_agent)
            tool = next((t for t in tools if t.name == file_name), None)
            if not tool:
                logger.error(f"Tool metadata not found for script: {file_name}")
                return f"Tool metadata not found for script: {file_name}"

            if tool.parameters:
                error = self._validate_parameters(arg, tool.parameters)
                if error:
                    logger.warning(f"Parameter validation failed: {error}")
                    return error

            venv_path = "venvs/auto_ninja_venv"
            python_bin = os.path.join(venv_path, "bin" if os.name != "nt" else "Scripts", "python")
            content = load_script(file_name)
            pip_packages, system_deps = infer_dependencies(content, self.inference_engine)
            
            success, error_msg = self._setup_venv(venv_path, pip_packages, system_deps)
            if not success:
                logger.error(f"Virtual environment setup failed: {error_msg}")
                return f"Error setting up virtual environment: {error_msg}"

            env = os.environ.copy()
            env["PATH"] = f"/usr/bin:/usr/local/bin:{env.get('PATH', '')}"
            env["MEMORY_MANAGER"] = "app.middleware.memory.MemoryManager"
            cmd = [python_bin, str(script_path)] + ([arg] if arg else [])
            result = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, env=env)
            logger.info(f"Script executed successfully: {file_name}")
            return result.strip()
        except subprocess.CalledProcessError as e:
            logger.error(f"Script execution failed: {e.output}")
            correction_prompt = f"Fix this script that failed with error:\n{e.output}\nScript:\n{content}"
            correction_result = self.inference_engine.process(correction_prompt)
            try:
                match = re.search(r"```python\s*(.*?)\s*```", correction_result, re.DOTALL)
                if match:
                    fixed_code = match.group(1)
                    self.update_script(f"Update with fixed code: {fixed_code}", file_name)
                    logger.info(f"Attempting re-execution of fixed script: {file_name}")
                    return self.execute_script(command)
                return f"Code execution failed: {e.output}"
            except Exception:
                return f"Code execution failed and correction unsuccessful: {e.output}"
        except Exception as e:
            logger.error(f"Unexpected error during script execution: {str(e)}")
            return f"Error executing script: {str(e)}"

    def _validate_parameters(self, arg: str, parameters: list) -> str:
        if not parameters:
            return "" if not arg else "No parameters expected but argument provided."
        
        input_value = None
        if arg:
            try:
                input_value = json.loads(arg)
            except json.JSONDecodeError:
                if parameters[0]["type"].startswith("list[") and "," in arg:
                    input_value = [v.strip() for v in arg.split(",")]

        for param in parameters:
            name = param["name"]
            param_type = param["type"]
            required = param["required"]
            default = param["default"]
            fields = param.get("fields", [])

            if required and input_value is None:
                return self._format_param_error(param, "Missing required parameter")

            if input_value is None and not required:
                continue

            if param_type in ["str", "int", "float", "bool"]:
                if not isinstance(input_value, (str, int, float, bool)):
                    return self._format_param_error(param, f"Expected {param_type}, got {type(input_value).__name__}")
                if param_type == "int":
                    try:
                        input_value = int(input_value)
                    except (ValueError, TypeError):
                        return self._format_param_error(param, "Invalid integer")
                elif param_type == "float":
                    try:
                        input_value = float(input_value)
                    except (ValueError, TypeError):
                        return self._format_param_error(param, "Invalid float")
                elif param_type == "bool":
                    if input_value not in [True, False, "true", "false"]:
                        return self._format_param_error(param, "Invalid boolean")
            elif param_type.startswith("list["):
                inner_type = param_type[5:-1]
                if not isinstance(input_value, list):
                    return self._format_param_error(param, f"Expected {param_type}, got {type(input_value).__name__}")
                for item in input_value:
                    if inner_type == "str" and not isinstance(item, str):
                        return self._format_param_error(param, f"List items must be strings, got {type(item).__name__}")
                    elif inner_type == "int":
                        try:
                            int(item)
                        except (ValueError, TypeError):
                            return self._format_param_error(param, "List items must be integers")
                    elif inner_type == "float":
                        try:
                            float(item)
                        except (ValueError, TypeError):
                            return self._format_param_error(param, "List items must be floats")
                    elif inner_type == "bool":
                        if item not in [True, False, "true", "false"]:
                            return self._format_param_error(param, "List items must be booleans")
            elif param_type.startswith("object["):
                if not isinstance(input_value, dict):
                    return self._format_param_error(param, f"Expected {param_type}, got {type(input_value).__name__}")
                for field in fields:
                    fname = field["name"]
                    ftype = field["type"]
                    if fname not in input_value:
                        return self._format_param_error(param, f"Missing field {fname} in {param_type}")
                    if ftype == "str" and not isinstance(input_value[fname], str):
                        return self._format_param_error(param, f"Field {fname} must be string")
                    elif ftype == "int":
                        try:
                            int(input_value[fname])
                        except (ValueError, TypeError):
                            return self._format_param_error(param, f"Field {fname} must be integer")
                    elif ftype == "float":
                        try:
                            float(input_value[fname])
                        except (ValueError, TypeError):
                            return self._format_param_error(param, f"Field {fname} must be float")
                    elif ftype == "bool":
                        if input_value[fname] not in [True, False]:
                            return self._format_param_error(param, f"Field {fname} must be boolean")
        return ""

    def _format_param_error(self, param: dict, message: str) -> str:
        error = f"Error: {message} for '{param['name']}'.\n"
        error += f"Expected: {param['type']} - {param['description']}\n"
        if param["required"]:
            error += "Required: Yes\n"
        if "default" in param and param["default"] is not None:
            error += f"Default: {param['default']}\n"
        if param.get("fields"):
            error += "Fields:\n" + "\n".join(
                f"- {f['name']} ({f['type']}): {f['description']}" for f in param['fields']
            ) + "\n"
        return error.strip()

    def update_script(self, prompt: str, file_name: str) -> str:
        try:
            if not validate_file_name(file_name) or not script_exists(file_name):
                logger.error(f"Invalid or missing script file: {file_name}")
                return f"Invalid or missing script file: {file_name}"
            
            agent = self.memory_manager.current_agent
            agent_name = agent.name if agent else "Agent"
            agent_role = agent.role if agent else "General"
            skill_name = file_name.replace("skill_", "").replace("tool_", "").rsplit("_", 1)[0].replace("_", " ").title()
            skill_description = next(
                (s.description for s in self.list_skills() if s.name.lower() == skill_name.lower()), prompt
            )
            content, parameters = generate_script_content(
                self.inference_engine,
                prompt.replace("update code", "").strip(),
                skill_name=skill_name,
                agent_name=agent_name,
                agent_role=agent_role,
                skill_description=skill_description
            )
            if not save_script(file_name, content):
                logger.error(f"Failed to save updated script: {file_name}")
                return f"Error: Failed to save updated script {file_name}"

            tools = self.memory_manager.memory.get_tools_for_agent(self.memory_manager.current_agent)
            for tool in tools:
                if tool.name == file_name:
                    tool.instructions["content"] = content
                    tool.parameters = parameters
                    tool.updated_at = datetime.now().isoformat()
                    self.memory_manager.memory.store_tool(tool)
                    break

            for skill in self.list_skills():
                if skill.instructions.get("type") == "script" and os.path.basename(skill.instructions.get("path", "")) == file_name:
                    skill.instructions["parameters"] = parameters
                    self.memory_manager.memory.store_skill(skill)
                    break

            summary = f"Updated Python script: {file_name} with parameters: {json.dumps(parameters)}"
            self.memory_manager.process(prompt, content, summary)
            logger.info(f"Script updated successfully: {file_name}")
            return "Code updated successfully!"
        except Exception as e:
            logger.error(f"Failed to update script: {str(e)}")
            return f"Error updating script: {str(e)}"

    def delete_script(self, file_name: str) -> str:
        try:
            if not validate_file_name(file_name) or not script_exists(file_name):
                logger.error(f"Invalid or missing script file: {file_name}")
                return f"Invalid or missing script file: {file_name}"
            
            delete_script(file_name)
            tools = self.memory_manager.memory.get_tools_for_agent(self.memory_manager.current_agent)
            for tool in tools:
                if tool.name == file_name:
                    self.memory_manager.memory.sqlite_store.cursor.execute(
                        "DELETE FROM agent_tools WHERE agent_id = ? AND tool_id = ?",
                        (self.memory_manager.current_agent.id, tool.id)
                    )
                    self.memory_manager.memory.sqlite_store.cursor.execute(
                        "DELETE FROM tools WHERE id = ?", (tool.id,)
                    )
                    self.memory_manager.memory.sqlite_store.conn.commit()
                    break

            for skill in self.list_skills():
                if skill.instructions.get("type") == "script" and os.path.basename(skill.instructions.get("path", "")) == file_name:
                    skill.instructions = {"type": "text", "content": skill.description or "No script available", "parameters": []}
                    self.memory_manager.memory.store_skill(skill)

            summary = f"Deleted Python script: {file_name}"
            self.memory_manager.process(f"delete code {file_name}", "Deleted", summary)
            logger.info(f"Script deleted successfully: {file_name}")
            return "Code deleted successfully!"
        except Exception as e:
            logger.error(f"Failed to delete script: {str(e)}")
            return f"Error deleting script: {str(e)}"

    def _setup_venv(self, venv_path: str, pip_packages: List[str], system_deps: List[str]) -> tuple[bool, str]:
        try:
            venv_python = os.path.join(venv_path, "bin" if os.name != "nt" else "Scripts", "python")
            venv_pip = os.path.join(venv_path, "bin" if os.name != "nt" else "Scripts", "pip")

            if not os.path.exists(venv_python):
                subprocess.check_call(["python", "-m", "venv", venv_path])
                logger.info(f"Created virtual environment: {venv_path}")

            if pip_packages:
                installed = {line.split()[0].lower() for line in subprocess.check_output([venv_pip, "list"], text=True).splitlines()[2:]}
                to_install = [pkg for pkg in pip_packages if pkg.lower() not in installed]
                if to_install:
                    for pkg in to_install:
                        subprocess.check_call([venv_pip, "install", pkg])
                    logger.info(f"Installed packages: {to_install}")

            if system_deps:
                missing = [dep for dep in system_deps if subprocess.call(["which", dep], stdout=subprocess.PIPE) != 0]
                if missing:
                    try:
                        subprocess.check_call(["sudo", "-n", "apt", "install", "-y"] + missing)
                    except subprocess.CalledProcessError:
                        sudo_pass = getpass.getpass(f"Enter sudo password to install {', '.join(missing)}: ")
                        cmd = ["sudo", "-S", "apt", "install", "-y"] + missing
                        subprocess.check_call(f"echo {sudo_pass} | {' '.join(cmd)}", shell=True)
                    logger.info(f"Installed system dependencies: {missing}")
            return True, ""
        except subprocess.CalledProcessError as e:
            logger.error(f"Virtual environment setup failed: {str(e)}")
            return False, f"Failed to setup venv or install dependencies: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in venv setup: {str(e)}")
            return False, f"Unexpected error: {str(e)}"

    def acquire_skill(self, skill_name: str, description: str) -> Skill:
        try:
            role_context = f"{skill_name}: {description}" if description else skill_name
            skills = self.auto_acquire_skills(role_context, batch_size=1, is_bootstrap=False)
            primary_skill = next((s for s in skills if s.name.lower() == skill_name.lower()), skills[0])
            logger.info(f"Acquired skill: {skill_name}")
            return primary_skill
        except Exception as e:
            logger.error(f"Failed to acquire skill {skill_name}: {str(e)}")
            raise ValueError(f"Failed to acquire skill: {str(e)}")

    def auto_acquire_skills(self, role_or_context: str, batch_size: int = 10, is_bootstrap: bool = True) -> List[Skill]:
        try:
            if is_bootstrap:
                # Full cleanup for bootstrap
                for script_file in glob.glob(os.path.join(SCRIPT_DIR, "skill_*_*.py")):
                    try:
                        os.remove(script_file)
                        logger.info(f"Removed old script: {script_file}")
                    except Exception as e:
                        logger.warning(f"Failed to remove old script {script_file}: {str(e)}")
                self.memory_manager.memory.sqlite_store.cursor.execute(
                    "DELETE FROM tools WHERE instructions LIKE '{\"type\": \"script\"%'"
                )
                self.memory_manager.memory.sqlite_store.cursor.execute(
                    "DELETE FROM agent_tools WHERE tool_id NOT IN (SELECT id FROM tools)"
                )
                self.memory_manager.memory.sqlite_store.conn.commit()
                logger.info("Cleared old script-based tools for bootstrap")

            agent = self.memory_manager.current_agent
            agent_name = agent.name if agent else "Agent"
            agent_role = agent.role if agent else "General"
            role_info = self.inference_engine.get_role_info(role_or_context)
            all_skills_data = role_info.get("skills", [])
            all_skills = self._fetch_all_skills(role_or_context, all_skills_data)
            
            stored_skills = []
            existing_skills = {s.name.lower(): s for s in self.list_skills()}
            for skill_data in all_skills:
                skill_name = skill_data["name"]
                skill_description = skill_data["description"]
                existing_skill = existing_skills.get(skill_name.lower())
                
                if existing_skill:
                    if is_bootstrap:
                        # Replace existing skill in bootstrap mode
                        self.memory_manager.memory.sqlite_store.cursor.execute(
                            "DELETE FROM skills WHERE agent_id = ? AND name = ?",
                            (self.memory_manager.current_agent.id, skill_name)
                        )
                    else:
                        # Update existing skill if description differs
                        if existing_skill.description != skill_description:
                            existing_skill.description = skill_description
                            self.memory_manager.memory.store_skill(existing_skill)
                            logger.info(f"Updated description for existing skill: {skill_name}")
                        continue  # Skip new insertion in incremental mode
                
                skill = Skill(
                    agent_id=self.memory_manager.current_agent.id,
                    name=skill_name,
                    description=skill_description,
                    instructions={"type": "text", "content": skill_description},
                    acquired=False
                )
                skill_id = self.memory_manager.memory.store_skill(skill)
                skill.id = skill_id
                stored_skills.append(skill)
                existing_skills[skill_name.lower()] = skill

            unacquired = [s for s in stored_skills if not s.acquired][:batch_size]
            for skill, skill_data in zip(unacquired, all_skills[:len(unacquired)]):
                self._process_skill(skill, agent_name, agent_role, skill_data.get("code"))
                skill.acquired = True
                self.memory_manager.memory.store_skill(skill)
                summary = f"Skill: {skill.name} - {skill.description or 'No description'}"
                embedding = generate_embedding(summary)
                self.memory_manager.memory.qdrant_store.store_summary(skill, summary, embedding, "skill_summaries")
                logger.info(f"Processed skill: {skill.name}")
            return stored_skills
        except Exception as e:
            logger.error(f"Failed to auto-acquire skills: {str(e)}")
            raise ValueError(f"Failed to acquire skills: {str(e)}")

    def _fetch_all_skills(self, role_or_context: str, initial_skills: List[Dict]) -> List[Dict]:
        try:
            all_skills = []
            seen_names = set()
            for skill in initial_skills:
                skill_name = skill["name"]
                if skill_name.lower() not in seen_names:
                    all_skills.append(skill)
                    seen_names.add(skill_name.lower())
                sub_prompt = (
                    f"""
                    List skills required for '{skill['name']}' in a '{role_or_context}' role. Return JSON:
                    {{
                      "skills": [
                        {{
                          "name": "str",
                          "description": "str",
                          "code": "str or null",
                          "parameters": [
                            {{
                              "name": "str",
                              "type": "str|int|float|bool|list[str]|list[int]|list[float]|list[bool]|object[class_name]",
                              "description": "str",
                              "required": bool,
                              "default": "value or null|[]|null",
                              "fields": [
                                {{"name": "str", "type": "str|int|float|bool", "description": "str"}}
                              ]
                            }}
                          ]
                        }}
                      ]
                    }}
                    """
                )
                sub_response = self.inference_engine.process(sub_prompt)
                try:
                    sub_skills_data = robust_parse_json(sub_response)
                    sub_skills = sub_skills_data.get("skills", [])
                    for sub_skill in sub_skills:
                        sub_name = sub_skill["name"]
                        if sub_name.lower() not in seen_names:
                            if sub_skill.get("code") and not self._is_valid_python(sub_skill["code"]):
                                sub_skill["code"] = None
                            all_skills.append(sub_skill)
                            seen_names.add(sub_name.lower())
                except ValueError:
                    logger.warning(f"Failed to parse sub-skills for {skill['name']}")
                    continue
            return all_skills
        except Exception as e:
            logger.error(f"Failed to fetch all skills: {str(e)}")
            return initial_skills

    def _process_skill(self, skill: Skill, agent_name: str, agent_role: str, code: Optional[str] = None):
        try:
            prompt = (
                f"""
                Does skill '{skill.name}' (Description: '{skill.description}') need a script for agent '{agent_name}' with role '{agent_role}'? Return JSON:
                {{
                  "needs_script": bool,
                  "description": "str",
                  "code": "str or null",
                  "parameters": [
                    {{
                      "name": "str",
                      "type": "str|int|float|bool|list[str]|list[int]|list[float]|list[bool]|object[class_name]",
                      "description": "str",
                      "required": bool,
                      "default": "value or null|[]|null",
                      "fields": [
                        {{"name": "str", "type": "str|int|float|bool", "description": "str"}}
                      ]
                    }}
                  ]
                }}
                """
            )
            response = self.inference_engine.process(prompt)
            details = robust_parse_json(response)
            needs_script = details.get("needs_script", False)
            description = details.get("description", skill.description or "No additional details")
            parameters = details.get("parameters", [])

            # Check for existing script
            existing_file_name = None
            for tool in self.memory_manager.memory.get_tools_for_agent(self.memory_manager.current_agent):
                if tool.name.startswith(f"skill_{skill.name.lower().replace(' ', '_')}_") and tool.instructions.get("type") == "script":
                    existing_file_name = tool.name
                    break

            if needs_script:
                script_response = self.generate_script(
                    code or details.get("code") or f"Generate script for {skill.name}: {description}",
                    skill_name=skill.name
                )
                if "Error" not in script_response:
                    file_name = script_response.split("File: ")[1].strip()
                    skill.instructions = {
                        "type": "script",
                        "path": f"scripts/{file_name}",
                        "venv": "venvs/auto_ninja_venv",
                        "parameters": parameters
                    }
                    skill.script_id = self.memory_manager.memory.sqlite_store.cursor.lastrowid
                    logger.info(f"Generated script for skill: {skill.name}, file: {file_name}")
                else:
                    skill.instructions = {
                        "type": "text",
                        "content": description,
                        "parameters": []
                    }
                    logger.warning(f"Failed to generate script for {skill.name}: {script_response}")
            else:
                skill.instructions = {
                    "type": "text",
                    "content": description,
                    "parameters": []
                }
                logger.info(f"Set text instructions for skill: {skill.name}")
        except ValueError as e:
            logger.warning(f"Failed to process skill {skill.name}: {str(e)}")
            skill.instructions = {
                "type": "text",
                "content": skill.description or "No additional details",
                "parameters": []
            }
        except Exception as e:
            logger.error(f"Unexpected error processing skill {skill.name}: {str(e)}")
            skill.instructions = {
                "type": "text",
                "content": skill.description or "No additional details",
                "parameters": []
            }

    def _is_valid_python(self, code: str) -> bool:
        return code and code.lower() != "null" and any(keyword in code for keyword in ["import", "def", "class", "return"])

    def update_skill(self, skill_id: int, description: str) -> Skill:
        try:
            skill = self.memory_manager.memory.get_skill(skill_id)
            if not skill:
                logger.error(f"Skill {skill_id} not found")
                raise ValueError(f"Skill {skill_id} not found")
            file_name = os.path.basename(skill.instructions.get("path", "")) if skill.instructions.get("type") == "script" else None
            if file_name:
                prompt = f"Update script for '{skill.name}' with description '{description}'. Use sys.argv for parameters."
                self.update_script(prompt, file_name)
            skill.description = description
            skill.instructions["content"] = description if skill.instructions.get("type") == "text" else skill.instructions.get("content", description)
            self.memory_manager.memory.store_skill(skill)
            logger.info(f"Updated skill: {skill.name}")
            return skill
        except Exception as e:
            logger.error(f"Failed to update skill {skill_id}: {str(e)}")
            raise ValueError(f"Failed to update skill: {str(e)}")

    def delete_skill(self, skill_id: int) -> str:
        try:
            skill = self.memory_manager.memory.get_skill(skill_id)
            if not skill:
                logger.error(f"Skill {skill_id} not found")
                raise ValueError(f"Skill {skill_id} not found")
            if skill.instructions.get("type") == "script":
                file_name = os.path.basename(skill.instructions.get("path", ""))
                self.delete_script(file_name)
            self.memory_manager.memory.sqlite_store.cursor.execute("DELETE FROM skills WHERE id = ?", (skill_id,))
            self.memory_manager.memory.sqlite_store.conn.commit()
            logger.info(f"Deleted skill: {skill.name}")
            return f"Skill '{skill.name}' deleted successfully"
        except Exception as e:
            logger.error(f"Failed to delete skill {skill_id}: {str(e)}")
            raise ValueError(f"Failed to delete skill: {str(e)}")

    def use_skill(self, skill_id: int, task: str, parameters: dict = None) -> str:
        try:
            skill = self.memory_manager.memory.get_skill(skill_id)
            if not skill:
                logger.error(f"Skill {skill_id} not found")
                raise ValueError(f"Skill {skill_id} not found")
            if skill.instructions.get("type") == "script":
                file_name = os.path.basename(skill.instructions.get("path", ""))
                arg = json.dumps(parameters) if parameters else task
                return self.execute_script(f"{file_name} {arg}")
            prompt = f"Using skill '{skill.name}' with instructions '{skill.instructions['content']}', complete: {task}"
            result = self.inference_engine.process(prompt)
            logger.info(f"Used skill {skill.name} with text instructions")
            return result
        except Exception as e:
            logger.error(f"Failed to use skill {skill_id}: {str(e)}")
            return f"Error using skill: {str(e)}"

    def list_skills(self) -> List[Skill]:
        try:
            skills = self.memory_manager.memory.sqlite_store.get_skills_by_agent(self.memory_manager.current_agent.id)
            logger.info(f"Listed {len(skills)} skills for agent {self.memory_manager.current_agent.id}")
            return skills
        except Exception as e:
            logger.error(f"Failed to list skills: {str(e)}")
            return []