```python
import os
from dataclasses import dataclass


def _str(name: str, default: str) -> str:
    raw = os.getenv(name)
    return default if raw is None or raw.strip() == "" else raw.strip()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = ""  # OPENAI_API_KEY (required for answers)
    model: str = "gpt-4.1-mini"
    max_tokens: int = 1024

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            openai_api_key=_str("OPENAI_API_KEY", ""),
            model=_str("OPENAI_MODEL", "gpt-4.1-mini"),
            max_tokens=int(_str("MAX_OUTPUT_TOKENS", "1024")),
        )
```
