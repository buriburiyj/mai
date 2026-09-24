import pytest

from mai.core.providers import get_provider


def test_provider_aliases() -> None:
    assert get_provider("hf").name == "huggingface"
    assert get_provider("cf").name == "cloudflare"
    assert get_provider("google").name == "gemini"


def test_cloudflare_requires_two_credentials() -> None:
    provider = get_provider("cloudflare")
    assert len(provider.credentials) == 2


def test_unknown_provider() -> None:
    with pytest.raises(ValueError):
        get_provider("unknown-provider")
