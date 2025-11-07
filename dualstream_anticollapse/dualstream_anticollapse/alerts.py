
import json, time, os, sys
from typing import Dict, Any, Optional
def emit(event_type: str, payload: Dict[str, Any], sink: Optional[str]=None):
    evt = {"ts": time.time(), "event": event_type, "payload": payload}
    if sink is None or sink == "stdout":
        print(json.dumps(evt))
    elif sink == "file":
        path = payload.get("_path", "artifacts/events.log.jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(evt) + "\n")
    # Placeholder for webhooks etc.
