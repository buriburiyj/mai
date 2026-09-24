from mai.core.catalog import extract_ids, pick_model


def test_extract_openai_shape() -> None:
    assert extract_ids("groq", {"data": [{"id": "a"}, {"id": "b"}]}) == ["a", "b"]


def test_extract_gemini_strips_prefix() -> None:
    payload = {"models": [{"name": "models/gemini-flash-lite-latest"}]}
    assert extract_ids("gemini", payload) == ["gemini-flash-lite-latest"]


def test_extract_cloudflare_result() -> None:
    payload = {"result": [{"name": "@cf/meta/llama-3.1-8b-instruct"}]}
    assert extract_ids("cloudflare", payload) == ["@cf/meta/llama-3.1-8b-instruct"]


def test_pick_keeps_configured() -> None:
    assert pick_model("groq", "x", ["x", "y"]) == "x"


def test_pick_uses_preference() -> None:
    models = ["whisper-large-v3", "openai/gpt-oss-20b"]
    assert pick_model("groq", "missing", models) == "openai/gpt-oss-20b"


def test_pick_without_list_keeps_configured() -> None:
    assert pick_model("groq", "keep-me", []) == "keep-me"
