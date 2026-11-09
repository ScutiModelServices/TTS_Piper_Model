"""The options the admin page can change. Each can still start from its environment variable; a saved value wins."""

from engine import DEFAULT_REPO, DEFAULT_VOICES
from settings import Setting

SETTINGS = [
    Setting("update_check_hours", "Check for updates every (hours)", "float", 6.0, "SERVICE_UPDATE_CHECK_HOURS", 0, 720,
            help="0 = only when you press Check now"),
    Setting("max_inflight", "Calls allowed to wait for the model", "int", 0, "SERVICE_MAX_INFLIGHT", 0, 10000,
            help="Per replica; more get 429. 0 = 4"),
    Setting("cache_size", "Answers kept per replica", "int", 64, "SERVICE_CACHE_SIZE", 0, 100000,
            help="The last answers for the same text, voice and speed, sent again without speaking. Each is at most 1 MB. 0 = off"),
    Setting("voices", "Voices served", "text", DEFAULT_VOICES, "SERVICE_VOICES", blank=False,
            help="Comma separated Piper voice names, such as en_US-lessac-medium. The first is the default. Each downloads on first use"),
    Setting("model", "Voices repository", "text", DEFAULT_REPO, "SERVICE_MODEL", restart=True, blank=False,
            help="A Hugging Face repository laid out like rhasspy/piper-voices"),
    Setting("warm", "Load the default voice at start", "bool", True, "SERVICE_WARM", restart=True),
]
