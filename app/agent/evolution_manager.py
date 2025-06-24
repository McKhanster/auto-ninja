# agent/evolution_manager.py
import os
import glob
import subprocess
import logging
import re
import json

logger = logging.getLogger(__name__)

class EvolutionManager:
    def __init__(self, inference_engine, memory_manager):
        self.inference_engine = inference_engine
        self.memory_manager = memory_manager

    def evolve(self) -> str:
        """Condense codebase, assess with Grok, rewrite, and restart."""
        # Step 1: Condense codebase
        codebase = "# Auto Ninja Condensed Codebase\n\n"
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        exclude_dirs = {'__pycache__', 'scripts', 'speech', 'venvs'}
        for root, dirs, files in os.walk(base_dir):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            for file in files:
                if file.endswith('.py') and file != 'auto_ninja.py':
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as f:
                        codebase += f"\n# File: {os.path.relpath(file_path, base_dir)}\n{f.read()}\n"
        escaped_codebase = codebase.replace('\n', '\\n').replace('"', '\\"')
        # Step 2: Use generic prompt with agent details
        agent_name = self.memory_manager.current_agent.name
        agent_role = self.memory_manager.current_agent.role
        prompt = EVOLUTION_PROMPT.format(agent_name=agent_name, agent_role=agent_role, codebase=escaped_codebase)
        
        # Step 3: Send to Grok for assessment
        try:
            response = self.inference_engine.process(prompt)
            logger.debug(f"Grok codebase assessment: {response}")
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            result = json.loads(json_match.group(0) if json_match else response)
            assessment = result.get("assessment", "No assessment provided")
            improvements = result.get("improvements", "No improvements suggested")
            revised_code = result.get("revised_code", codebase)
            
            # Step 4: Rewrite codebase
            new_file_path = os.path.join(base_dir, 'auto_ninja_evolved.py')
            with open(new_file_path, 'w') as f:
                f.write(revised_code)
            logger.info(f"Codebase rewritten to {new_file_path}")
            logger.info(f"Assessment: {assessment}\nImprovements: {improvements}")
            
            # Step 5: Generate and execute restart script
            restart_script = os.path.join(base_dir, 'restart.sh')
            with open(restart_script, 'w') as f:
                f.write(f"""#!/bin/bash
                echo "Shutting down current Auto Ninja..."
                pkill -f "python auto_ninja.py"
                sleep 2
                echo "Replacing old codebase..."
                mv {new_file_path} {os.path.join(base_dir, 'auto_ninja.py')}
                echo "Restarting Auto Ninja..."
                python {os.path.join(base_dir, 'auto_ninja.py')} &
                """)
            os.chmod(restart_script, 0o755)
            subprocess.run(['bash', restart_script], check=True)
            logger.info("Restart script executed; Auto Ninja is evolving...")
            return "Evolution complete; restarting with new codebase."
        except Exception as e:
            logger.error(f"Failed to evolve: {str(e)}")
            return f"Evolution failed: {str(e)}"

EVOLUTION_PROMPT = """
I am {agent_name}, a {agent_role} AI agent. Here iis my condensed codebase:
- Files: shared.py (agent initialization), skill_manager.py (skill acquisition), evolution_manager.py (self-evolution), auto_ninja.py (CLI interface), and others in app/ subdirectories.
{codebase}

Assess my codebase for improvements as a {agent_role}. Return a JSON response:
{{
    "assessment": "Brief evaluation of the codebase",
    "improvements": "Suggested changes or optimizations",
    "revised_code": "Complete revised Python codebase as a single file, implementing the improvements"
}}
Return ONLY the JSON string, with no additional text or markdown.
"""