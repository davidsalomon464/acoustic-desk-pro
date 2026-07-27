import os
import sys
import subprocess
import webbrowser
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ActionsEngine")

def execute_action(action_config: dict):
    """
    Executes a configured action based on action_config dictionary:
    {
        "type": "url" | "app" | "cmd" | "shortcut" | "system",
        "value": string target or payload,
        "name": optional display name
    }
    """
    if not action_config or not isinstance(action_config, dict):
        logger.warning("Empty or invalid action config received.")
        return False, "Invalid action config"

    action_type = action_config.get("type", "none")
    value = action_config.get("value", "").strip()

    if action_type == "none" or not value:
        logger.info("Action configured as 'none' or value empty. Skipping.")
        return True, "No action executed"

    try:
        if action_type == "url":
            if not value.startswith(("http://", "https://")):
                value = "https://" + value
            logger.info(f"Opening URL: {value}")
            webbrowser.open(value)
            return True, f"Opened URL: {value}"

        elif action_type == "app":
            logger.info(f"Opening Application: {value}")
            if sys.platform == "darwin":  # macOS
                if value.endswith(".app") or "/" in value:
                    subprocess.Popen(["open", value])
                else:
                    subprocess.Popen(["open", "-a", value])
            elif sys.platform == "win32":
                os.startfile(value)
            else:  # Linux
                subprocess.Popen(["xdg-open", value])
            return True, f"Launched application: {value}"

        elif action_type == "cmd":
            logger.info(f"Executing command: {value}")
            subprocess.Popen(value, shell=True)
            return True, f"Executed command: {value}"

        elif action_type == "shortcut" or action_type == "system":
            logger.info(f"Triggering system action/shortcut: {value}")
            if sys.platform == "darwin":
                if value == "volume_up":
                    subprocess.run(["osascript", "-e", "set volume output volume ((output volume of (get volume settings)) + 10)"])
                elif value == "volume_down":
                    subprocess.run(["osascript", "-e", "set volume output volume ((output volume of (get volume settings)) - 10)"])
                elif value == "mute":
                    subprocess.run(["osascript", "-e", "set volume output muted (not (output muted of (get volume settings)))"])
                elif value == "screenshot":
                    subprocess.run(["screencapture", "-c"])  # Screenshot to clipboard
                elif value == "sleep":
                    subprocess.run(["pmset", "sleepnow"])  # Sleep mac
                elif value == "lock_screen":
                    subprocess.run(["osascript", "-e", 'tell application "System Events" to keystroke "q" using {control down, command down}'])
                elif value == "shutdown_computer":
                    subprocess.run(["osascript", "-e", 'tell application "System Events" to shut down'])
                else:
                    subprocess.run(["osascript", "-e", value])
            elif sys.platform == "win32":
                if value == "sleep":
                    subprocess.run(["rundll32.exe", "powrprof.dll,SetSuspendState", "0", "1", "0"])
                elif value == "shutdown_computer":
                    subprocess.run(["shutdown", "/s", "/t", "0"])
                elif value == "lock_screen":
                    subprocess.run(["rundll32.exe", "user32.dll,LockWorkStation"])
            return True, f"Triggered system action: {value}"

        else:
            logger.warning(f"Unknown action type: {action_type}")
            return False, f"Unknown action type: {action_type}"

    except Exception as e:
        logger.error(f"Failed to execute action {action_type} -> {value}: {e}")
        return False, str(e)
