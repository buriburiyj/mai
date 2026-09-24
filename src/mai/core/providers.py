from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CredentialField:
    name: str
    label: str
    sensitive: bool = True


@dataclass(frozen=True, slots=True)
class Provider:
    name: str
    display_name: str
    credentials: tuple[CredentialField, ...]


API_KEY = (
    CredentialField(
        name="api_key",
        label="API key",
    ),
)

PROVIDERS: dict[str, Provider] = {
    "gemini": Provider("gemini", "Google Gemini", API_KEY),
    "cerebras": Provider("cerebras", "Cerebras", API_KEY),
    "openrouter": Provider("openrouter", "OpenRouter", API_KEY),
    "groq": Provider("groq", "Groq", API_KEY),
    "nvidia": Provider("nvidia", "NVIDIA", API_KEY),
    "mistral": Provider("mistral", "Mistral", API_KEY),
    "cloudflare": Provider(
        "cloudflare",
        "Cloudflare Workers AI",
        (
            CredentialField(
                name="api_token",
                label="Workers AI API token",
            ),
            CredentialField(
                name="account_id",
                label="Cloudflare Account ID",
                sensitive=False,
            ),
        ),
    ),
    "huggingface": Provider(
        "huggingface",
        "Hugging Face",
        (
            CredentialField(
                name="token",
                label="Hugging Face token",
            ),
        ),
    ),
}

ALIASES = {
    "google": "gemini",
    "cf": "cloudflare",
    "hf": "huggingface",
}


def normalize_provider_name(name: str) -> str:
    normalized = name.strip().lower()
    return ALIASES.get(normalized, normalized)


def get_provider(name: str) -> Provider:
    normalized = normalize_provider_name(name)

    if normalized not in PROVIDERS:
        available = ", ".join(PROVIDERS)
        raise ValueError(f"Unknown provider: {name}. Available providers: {available}")

    return PROVIDERS[normalized]
