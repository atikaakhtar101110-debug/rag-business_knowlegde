import os
from dataclasses import dataclass

def _str(name: str, default: str) -> str:
    raw = os.getenv(name)
    return default if raw is None or raw.strip() == "" else raw.strip()

@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str = ""   # ANTHROPIC_API_KEY (required for answers)
    model: str = "claude-sonnet-5-5"
    max_tokens: int = 1024

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            anthropic_api_key=_str("ANTHROPIC_API_KEY", ""),   # <-- key is read here
            model=_str("CLAUDE_MODEL", "claude-sonnet-5-5"),
            max_tokens=int(_str("MAX_OUTPUT_TOKENS", "1024")),
        )
