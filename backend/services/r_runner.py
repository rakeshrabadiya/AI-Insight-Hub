"""
AI Insight Hub — R Execution Engine Service (RRunner)
Manages subprocess execution of R scripts, handles data passing, captures output,
detects errors, enforces timeouts, and parses structured JSON responses.
"""
import os
import sys
import json
import shutil
import logging
import time
import subprocess
from typing import Dict, Any, Optional, List, Union

logger = logging.getLogger("ai_insight_hub.r_runner")


class RRunner:
    """
    Service for executing R scripts via Rscript subprocess with robust error handling,
    timeout management, and input/output serialization.
    """

    def __init__(self, default_timeout: int = 30):
        self.default_timeout = default_timeout
        self._rscript_path = None

    def find_rscript(self) -> Optional[str]:
        """
        Locates the Rscript executable by checking:
        1. RSCRIPT_PATH environment variable
        2. System PATH (via shutil.which)
        3. Common installation directories on Windows, Linux, and macOS
        """
        # 1. Environment variable override
        env_path = os.environ.get("RSCRIPT_PATH")
        if env_path and os.path.isfile(env_path) and os.access(env_path, os.X_OK):
            return env_path

        # 2. System PATH
        path_which = shutil.which("Rscript") or shutil.which("Rscript.exe")
        if path_which:
            return path_which

        # 3. Standard fallback directories
        candidates = []
        if sys.platform.startswith("win"):
            # Windows standard install roots
            program_files = [
                os.environ.get("ProgramFiles", r"C:\Program Files"),
                os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
            ]
            for pf in program_files:
                r_root = os.path.join(pf, "R")
                if os.path.isdir(r_root):
                    # Search inside version subdirectories like R-4.6.1, R-4.3.0
                    for entry in sorted(os.listdir(r_root), reverse=True):
                        cand_x64 = os.path.join(r_root, entry, "bin", "x64", "Rscript.exe")
                        cand_bin = os.path.join(r_root, entry, "bin", "Rscript.exe")
                        candidates.extend([cand_x64, cand_bin])
        elif sys.platform.startswith("darwin"):
            # macOS standard locations
            candidates.extend([
                "/usr/local/bin/Rscript",
                "/opt/homebrew/bin/Rscript",
                "/Library/Frameworks/R.framework/Resources/bin/Rscript"
            ])
        else:
            # Linux / POSIX locations
            candidates.extend([
                "/usr/bin/Rscript",
                "/usr/local/bin/Rscript",
                "/opt/R/bin/Rscript"
            ])

        for cand in candidates:
            if os.path.isfile(cand) and os.access(cand, os.X_OK):
                return cand

        return None

    def get_rscript_path(self) -> str:
        """Returns cached or newly discovered Rscript executable path."""
        if not self._rscript_path:
            self._rscript_path = self.find_rscript()
        return self._rscript_path

    def is_available(self) -> bool:
        """Checks whether Rscript executable exists and can be invoked."""
        path = self.get_rscript_path()
        if not path:
            return False
        try:
            res = subprocess.run(
                [path, "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5
            )
            return res.returncode == 0
        except Exception as e:
            logger.warning(f"Rscript availability probe failed: {e}")
            return False

    def get_version_info(self) -> Dict[str, Any]:
        """Returns detailed R environment and version metadata."""
        path = self.get_rscript_path()
        if not path:
            return {
                "available": False,
                "executable_path": None,
                "version": "Not installed / not in PATH",
                "error": "Rscript executable could not be found."
            }

        try:
            res = subprocess.run(
                [path, "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5
            )
            version_text = (res.stderr or res.stdout).strip()
            return {
                "available": True,
                "executable_path": path,
                "version": version_text.splitlines()[0] if version_text else "Unknown R Version",
                "raw_output": version_text
            }
        except Exception as e:
            return {
                "available": False,
                "executable_path": path,
                "version": "Error querying version",
                "error": str(e)
            }

    def execute_script(
        self,
        script_path: str,
        input_data: Optional[Union[Dict[str, Any], List[Any], str]] = None,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Executes a target R script file, passing input data as a JSON string argument,
        capturing stdout/stderr, and returning structured results.

        :param script_path: Absolute or project-relative path to .R script file.
        :param input_data: Optional dictionary, list, or string to pass as input.
        :param timeout: Execution timeout in seconds (defaults to self.default_timeout).
        :return: Structured result dictionary.
        """
        timeout_sec = timeout or self.default_timeout
        start_time = time.perf_counter()

        # Resolve absolute script path
        if not os.path.isabs(script_path):
            # Resolve relative to project root or current working directory
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            resolved_path = os.path.normpath(os.path.join(base_dir, script_path))
            if not os.path.isfile(resolved_path):
                # Fallback to direct path
                resolved_path = os.path.abspath(script_path)
        else:
            resolved_path = script_path

        # Verify script existence
        if not os.path.isfile(resolved_path):
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(f"R script not found at path: {resolved_path}")
            return {
                "success": False,
                "engine": "R",
                "message": f"R script file not found: {script_path}",
                "error": f"File not found: {resolved_path}",
                "data": None,
                "stdout": "",
                "stderr": "",
                "return_code": -1,
                "execution_time_ms": elapsed_ms
            }

        # Locate Rscript executable
        rscript = self.get_rscript_path()
        if not rscript or not os.path.isfile(rscript):
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error("Rscript executable could not be found.")
            return {
                "success": False,
                "engine": "R",
                "message": "R execution engine is not available on this system.",
                "error": "Rscript executable not found. Please ensure R is installed and in PATH.",
                "data": None,
                "stdout": "",
                "stderr": "",
                "return_code": -1,
                "execution_time_ms": elapsed_ms
            }

        # Format input argument as JSON string
        cmd = [rscript, resolved_path]
        if input_data is not None:
            if isinstance(input_data, (dict, list)):
                cmd.append(json.dumps(input_data))
            else:
                cmd.append(str(input_data))

        # Prepare environment variables including R_LIBS_USER
        env = os.environ.copy()
        if sys.platform.startswith("win"):
            # Check user library locations
            user_lib_candidates = [
                os.path.expanduser(r"~\Documents\R\win-library\4.6"),
                os.path.expanduser(r"~\AppData\Local\R\win-library\4.6"),
                os.environ.get("R_LIBS_USER", "")
            ]
            valid_libs = [p for p in user_lib_candidates if p and os.path.isdir(p)]
            if valid_libs:
                env["R_LIBS_USER"] = ";".join(valid_libs)

        logger.info(f"Executing R Script: {' '.join(cmd[:2])} (Timeout: {timeout_sec}s)")

        try:
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_sec,
                encoding="utf-8",
                errors="replace",
                env=env
            )

            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            stdout = process.stdout.strip()
            stderr = process.stderr.strip()

            if process.returncode != 0:
                logger.warning(f"R script exited with non-zero code {process.returncode}: {stderr}")
                return {
                    "success": False,
                    "engine": "R",
                    "message": f"R script execution failed with exit code {process.returncode}",
                    "error": stderr or stdout or "Unknown R execution error",
                    "data": None,
                    "stdout": stdout,
                    "stderr": stderr,
                    "return_code": process.returncode,
                    "execution_time_ms": elapsed_ms
                }

            # Attempt to parse stdout as JSON if formatted as such
            parsed_data = None
            if stdout:
                try:
                    # Find JSON substring if R output contains startup warnings
                    json_start = stdout.find("{")
                    json_end = stdout.rfind("}")
                    if json_start != -1 and json_end != -1 and json_end >= json_start:
                        json_str = stdout[json_start : json_end + 1]
                        parsed_data = json.loads(json_str)
                    else:
                        parsed_data = {"raw_output": stdout}
                except json.JSONDecodeError:
                    parsed_data = {"raw_output": stdout}

            return {
                "success": True,
                "engine": "R",
                "message": (parsed_data.get("message") if isinstance(parsed_data, dict) and "message" in parsed_data
                            else "R engine executed successfully"),
                "data": parsed_data,
                "stdout": stdout,
                "stderr": stderr,
                "return_code": 0,
                "execution_time_ms": elapsed_ms,
                "error": None
            }

        except subprocess.TimeoutExpired:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(f"R script execution timed out after {timeout_sec} seconds")
            return {
                "success": False,
                "engine": "R",
                "message": f"R script execution timed out after {timeout_sec} seconds.",
                "error": f"Execution exceeded maximum timeout limit of {timeout_sec}s",
                "data": None,
                "stdout": "",
                "stderr": "",
                "return_code": -2,
                "execution_time_ms": elapsed_ms
            }

        except Exception as e:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(f"Unexpected error running R script: {e}")
            return {
                "success": False,
                "engine": "R",
                "message": "Unexpected error occurred while running R script.",
                "error": str(e),
                "data": None,
                "stdout": "",
                "stderr": "",
                "return_code": -3,
                "execution_time_ms": elapsed_ms
            }


# Global singleton instance for easy import
r_runner = RRunner()
