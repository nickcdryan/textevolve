import re
import json
from datetime import datetime, timedelta

def main(question):
    """
    Schedules a meeting by leveraging an LLM-simulated, multi-agent system.
    This approach uses a chain-of-thought prompting strategy to guide the scheduling process.
    It extracts relevant information, reasons about constraints, and generates a suitable meeting time.
    """

    def call_llm(prompt, system_prompt=None):
        """Simulates a call to a Large Language Model."""
        # In a real implementation, this function would interact with an LLM API.
        # For this example, we use a simplified simulation.
        # The simulation focuses on demonstrating the reasoning chain.

        if system_prompt:
            prompt = f"{system_prompt}\n\n{prompt}"

        # Simple example responses based on keywords in the prompt.
        # This is NOT intended to be a real LLM but provides a framework.
        if "extract participants" in prompt.lower():
            participants = re.findall(r"for (.*?) for", question)
            if participants:
                participants = [p.strip() for p in participants[0].split(',')]
                return json.dumps({"participants": participants})
            else:
                return json.dumps({"participants": []})

        if "extract meeting duration" in prompt.lower():
            duration = re.search(r"for (.*?) between", question)
            if duration:
                duration_str = duration.group(1).strip()
                if "hour" in duration_str:
                    if "half" in duration_str:
                         return json.dumps({"duration": 30})
                    else:
                        return json.dumps({"duration": 60})
                else:
                    return json.dumps({"duration": 30}) # default half hour
            else:
                return json.dumps({"duration": 30})

        if "extract existing schedules" in prompt.lower():
            schedules = {}
            lines = question.split("\n")
            for line in lines:
                if "is busy" in line or "has meetings" in line or "blocked their calendar" in line:
                    participant = line.split(" ")[0]
                    times = re.findall(r"(\d{1,2}:\d{2} to \d{1,2}:\d{2})", line)
                    schedules[participant] = times
                elif "'s calendar is wide open" in line:
                  participant = line.split(" ")[0]
                  schedules[participant] = []
            return json.dumps({"schedules": schedules})

        if "extract time preferences" in prompt.lower():
            preferences = []
            if "not want to meet" in question:
                preferences.append("avoid certain times")
            if "can not meet" in question:
                preferences.append("avoid certain times")
            if "would like to avoid" in question:
                preferences.append("avoid certain times")
            return json.dumps({"preferences": preferences})

        if "find available time slots" in prompt.lower():
            data = json.loads(re.search(r"