# M4 spec: assistant

A typed chat assistant that answers from the team's own notes first, searches the web only when the notes do not cover the question, and knows who the camera currently recognises. Hosted model: Claude through the Anthropic Messages API. English for now.

Read `CLAUDE.md` first. `car_vision/state.py`, `car_vision/config.py` and `config.yaml` already exist and are fixed; read them, do not change them.

## Read before writing any SDK call

Do not write Anthropic SDK code from memory; several request shapes changed this year. The reference is on this machine, under
`C:\Users\USER\AppData\Local\Temp\claude\bundled-skills\2.1.328\f6d7b0f273dd9f5a2be79c48e87834e2\claude-api\`:

- `SKILL.md`: defaults, thinking and effort rules, server tools table, common pitfalls.
- `python\claude-api\README.md`: client, system prompts, prompt caching, error handling, stop reasons, refusal fallbacks.
- `python\claude-api\tool-use.md` and `shared\tool-use-concepts.md`: server tools and `pause_turn`.
- `shared\prompt-caching.md`: where the cache breakpoint goes.
- `shared\model-migration.md`, the Claude Opus 5.5 section and the preserved-thinking material it points to: what may and may not be changed in conversation history between requests.
- `shared\error-codes.md`: the typed exception classes.

If that folder is missing, load the `claude-api` skill instead. Follow the reference where it and this spec disagree on an SDK detail, and say so in your report.

## Design decisions already made

- One plain Messages API call per question, with Anthropic's server-side web search tool. No agent framework, no client-side tools, no vector database.
- Model and effort come from `config.yaml` (`claude-opus-5-5`, effort `low`). Adaptive thinking; never send `budget_tokens`, sampling parameters or a forced `tool_choice`.
- The whole notes file goes into the system prompt, which is stable from question to question and carries the cache breakpoint. Nothing that changes per question (names in front of the camera, the time) may go into the system prompt.
- Who the camera sees goes into the user turn, as one context line in front of the question.
- Web search uses the newest web search tool type the reference lists for this model, with `max_uses` from `web_search_max_uses`. When `web_search` is false, send no tools.
- Handle `stop_reason` properly: `pause_turn` (continue the same turn as the reference shows), `refusal`, `max_tokens`. Include the refusal fallback setting the reference says to include by default for this model.
- Conversation history must follow the history rules for this model. Keep assistant turns exactly as the API returned them and never edit an earlier turn. When the conversation passes `history_turns` exchanges, do whatever the reference allows (for example starting a fresh conversation) rather than rewriting history. State in your report which rule you followed and where you read it.

## File: `car_vision/assistant.py`

The only module that imports `anthropic`.

```python
class AssistantError(Exception):
    """Something the user can act on: no key, no internet, rate limit, refusal."""

@dataclass(frozen=True)
class Answer:
    text: str
    used_web: bool = False
    sources: tuple[str, ...] = ()        # page URLs, in the order cited, no repeats

class LLMClient(Protocol):
    """The seam that lets another provider or a local model replace Claude later."""
    def complete(self, system: str, turns: list, allow_web: bool) -> "LLMReply": ...

class AnthropicClient:                    # implements LLMClient
    def __init__(self, settings: dict) -> None: ...

class Assistant:
    def __init__(self, settings: dict, state: WorldState | None = None,
                 client: LLMClient | None = None) -> None: ...
    def ask(self, question: str) -> Answer: ...
    def reset(self) -> None: ...          # forget the conversation

def load_notes(path: Path) -> str: ...
def build_system_prompt(notes: str) -> str: ...
def people_line(names: list[str]) -> str: ...
def main(argv: list[str] | None = None) -> int: ...
```

You may shape `LLMReply` and the `turns` type as the SDK requires, as long as `Assistant` holds no Anthropic-specific code and a fake client can stand in for tests.

- `settings` is the `assistant:` section. Read every key with `require`: `provider`, `model`, `effort`, `max_tokens`, `notes_file`, `web_search`, `web_search_max_uses`, `history_turns`, `seen_within_seconds`. A `provider` other than `anthropic` raises `ConfigError`.
- The API key is read from the environment after `dotenv.load_dotenv` on the project's `.env`. The SDK client is created on first use, not at import. A missing key raises `AssistantError` telling the user to copy `.env.example` to `.env`. The key is never logged, printed or put into an exception message.
- `load_notes` reads the file as UTF-8. A missing file raises `AssistantError` naming the path. An empty file is allowed: the assistant then relies on web search.
- `build_system_prompt` is a pure function of the notes. It tells the model: it is the assistant of a student robot car project, shown on a screen at a live demo; answer from the notes when they cover the question; when they do not, search the web if the tool is available, otherwise say it does not know; never invent facts about the team or the people; keep answers to two or three sentences unless asked for more; answer in English.
- `people_line(["Sara", "Omar"])` returns one sentence such as `In front of the camera right now: Sara, Omar.`, and for an empty list a sentence saying nobody recognised is in front of the camera. `ask` gets the names from `state.current_people(seen_within_seconds)` when a state was given.
- `ask` strips the question; an empty one raises `ValueError`. It returns the final answer text with thinking and tool blocks left out. `used_web` is true when the reply contains a web search call; `sources` are the URLs from the reply's citations or search results.
- Map SDK exceptions most-specific-first into `AssistantError` with a plain sentence for each case: wrong key, rate limit, no connection, timeout, other API error. A failed question must not corrupt the history: the next `ask` works.
- `ask` may take several seconds and will be called from a worker thread in M5. One `Assistant` is used by one thread at a time; guard `ask` with a lock so a second call waits.

`main` is a typed chat loop: prompt, answer, and under a web answer the line `(from the web)` followed by the source URLs. `/reset` forgets the conversation, `/quit` or an empty line exits. An `AssistantError` prints its message and the loop continues.

## Also create

- `data/knowledge.md`: a short template with headings (Project, Team, Department, Demo) and one placeholder line under each, clearly marked as text the team must replace. No invented facts.
- `tests/test_assistant.py`.

## Tests

All pass with no network and no key, using a fake `LLMClient`.

- The system prompt contains the notes, is identical for two different questions, and contains no names from the camera.
- The people line reaches the client inside the user turn; empty and non-empty cases.
- History grows across questions and assistant turns are passed back unchanged; `reset` clears it; the behaviour at `history_turns` matches the rule you followed.
- `allow_web` follows the `web_search` setting.
- `used_web` and `sources` come through from the reply; duplicate URLs collapse.
- A client that raises leaves the history as it was, and the next question works.
- Missing key, missing notes file, empty question, unknown provider: each raises the right error, and no error message contains the key (set a recognisable fake key in the test environment and assert it is absent).
- A test of `AnthropicClient` with the SDK call replaced by a stub: the request has the model, effort, max tokens, a cached system prompt, the web search tool only when allowed, and no `budget_tokens`, sampling parameters or forced tool choice. Cover `pause_turn` continuation and a `refusal` stop.
- One test that reads every `.py` file under `car_vision/` and fails if any file other than `assistant.py` imports `anthropic`.
- One live test marked `@pytest.mark.hardware` that asks a real question; it is skipped by default and skipped when no key is set.

## Exit test on the laptop

1. `.venv\Scripts\python.exe -m pytest` passes.
2. `.venv\Scripts\python.exe -m car_vision.assistant` with no `.env` prints the missing-key message and does not crash.

The live test (10 questions from the notes, one web question, one "who is this?") needs the user's own key and notes and is done later.
