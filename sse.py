from enum import Enum

class EventType(str, Enum):
    MESSAGE = "message"
    DONE = "done"


def event(event_type: EventType, message: str) -> str:
    return f"event: {event_type.value}\nmessage: {message}\n\n"
