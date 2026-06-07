import os
import re
import unicodedata
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass
from typing import Any, Callable

SAMPLE_RATE = 16000
CHANNELS = 1
RECORD_SECONDS = 7
SILENCE_THRESHOLD = 100
MAX_INT16 = 32767
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")
WHISPER_LANGUAGE = "pl"
WHISPER_BEAM_SIZE = 5
WHISPER_TARGET_PEAK = 0.85
WHISPER_INITIAL_PROMPT = (
    "Slownictwo: krowki PJATK, cukierki, slodycze, dlugopis, dlugopisy, "
    "zapisac, zanotowac, numer telefonu, podpis, karteczki, sentencje, "
    "cytat, studiowanie, technologia, AI."
)
OLLAMA_MODEL = "qwen2.5:0.5b"
OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")
WAVE_OUTPUT = "command.wav"
RAG_TOP_K = 2
POLISH_TRANSLATION = str.maketrans(
    {
        "\u0105": "a",
        "\u0107": "c",
        "\u0119": "e",
        "\u0142": "l",
        "\u0144": "n",
        "\u00f3": "o",
        "\u015b": "s",
        "\u017a": "z",
        "\u017c": "z",
    }
)


@dataclass(frozen=True)
class ContainerDocument:
    id: int
    name: str
    description: str
    keywords: tuple[str, ...]
    examples: tuple[str, ...]


CONTAINER_DOCUMENTS: tuple[ContainerDocument, ...] = (
    ContainerDocument(
        id=0,
        name="Cukierki krowki PJATK",
        description=(
            "Pojemnik z cukierkami krowkami PJATK. Dobry wybor, gdy uzytkownik "
            "chce cos slodkiego, przekaske, cukierka albo nagrode."
        ),
        keywords=(
            "cukierek",
            "cukierki",
            "krowka",
            "krowki",
            "slodycze",
            "slodkie",
            "przekaska",
            "nagroda",
            "pjatk",
            "zjesc",
            "candy",
            "sweets",
            "sweet",
            "snack",
            "treat",
        ),
        examples=(
            "Mam ochote na cos slodkiego.",
            "Daj mi cukierka.",
            "Poprosze krowke z PJATK.",
            "I want something sweet.",
        ),
    ),
    ContainerDocument(
        id=1,
        name="Dlugopisy",
        description=(
            "Pojemnik z dlugopisami. Dobry wybor, gdy uzytkownik chce cos "
            "zapisac, zanotowac, podpisac, napisac numer telefonu albo robic notatki."
        ),
        keywords=(
            "dlugopis",
            "dlugopisy",
            "pisanie",
            "pisac",
            "napisac",
            "zapisac",
            "zanotowac",
            "notatka",
            "notatki",
            "podpisac",
            "telefon",
            "numer",
            "pen",
            "write",
            "writing",
            "note",
            "notebook",
            "phone",
            "number",
            "sign",
        ),
        examples=(
            "Chcialbym zapisac sobie numer telefonu.",
            "Musze cos zanotowac.",
            "Potrzebuje dlugopisu do podpisu.",
            "I need to write down a phone number.",
        ),
    ),
    ContainerDocument(
        id=2,
        name="Karteczki z sentencjami",
        description=(
            "Pojemnik z wymieszanymi karteczkami z madrymi sentencjami o "
            "studiowaniu, technologii i AI. Dobry wybor, gdy uzytkownik szuka "
            "inspiracji, motywacji, cytatu, mysli albo sentencji."
        ),
        keywords=(
            "karteczka",
            "karteczki",
            "sentencja",
            "sentencje",
            "cytat",
            "cytaty",
            "madrosc",
            "mysl",
            "inspiracja",
            "motywacja",
            "studia",
            "studiowanie",
            "technologia",
            "ai",
            "sztuczna",
            "inteligencja",
            "quote",
            "wisdom",
            "nuda",
            "nudzi mi",
            "inspiration",
            "motivation",
            "study",
            "studying",
            "technology",
            "artificial",
            "intelligence",
        ),
        examples=(
            "Potrzebuje inspirujacej mysli o studiowaniu.",
            "Daj mi cytat o AI.",
            "Chce przeczytac madra sentencje.",
            "Give me a quote about AI.",
        ),
    ),
)


def action_candy() -> None:
    print("[ACTION id=0] Wybrano pojemnik: cukierki krowki PJATK.")


def action_pen() -> None:
    print("[ACTION id=1] Wybrano pojemnik: dlugopisy.")


def action_quote_cards() -> None:
    print("[ACTION id=2] Wybrano pojemnik: karteczki z sentencjami.")


ACTIONS: dict[int, Callable[[], None]] = {
    0: action_candy,
    1: action_pen,
    2: action_quote_cards,
}


def normalize_text(text: str) -> str:
    """Normalize text for simple lexical retrieval."""
    text = text.lower().translate(POLISH_TRANSLATION)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9 ]+", " ", text)


def tokenize(text: str) -> set[str]:
    """Return searchable tokens without very short filler words."""
    return {token for token in normalize_text(text).split() if len(token) > 2}


def document_text(document: ContainerDocument) -> str:
    return " ".join(
        (
            document.name,
            document.description,
            " ".join(document.keywords),
            " ".join(document.examples),
        )
    )


def score_document(query: str, document: ContainerDocument) -> int:
    """Score one document against the query using exact and light stem matches."""
    query_tokens = tokenize(query)
    if not query_tokens:
        return 0

    doc_tokens = tokenize(document_text(document))
    keyword_tokens = {token for keyword in document.keywords for token in tokenize(keyword)}

    exact_hits = query_tokens & doc_tokens
    keyword_hits = query_tokens & keyword_tokens
    stem_hits = {
        query_token
        for query_token in query_tokens
        for doc_token in doc_tokens
        if len(query_token) >= 5
        and len(doc_token) >= 5
        and query_token[:5] == doc_token[:5]
    }

    return len(exact_hits) + (2 * len(keyword_hits)) + len(stem_hits)


def retrieve_documents(query: str, top_k: int = RAG_TOP_K) -> list[ContainerDocument]:
    """Retrieve the most relevant container descriptions for a short query."""
    scored: list[tuple[int, ContainerDocument]] = []
    for document in CONTAINER_DOCUMENTS:
        scored.append((score_document(query, document), document))

    scored.sort(key=lambda item: item[0], reverse=True)
    best = [document for score, document in scored[:top_k] if score > 0]
    return best or list(CONTAINER_DOCUMENTS)


def best_lexical_match(query: str) -> int | None:
    """Return the best document ID using only local retrieval scores."""
    scored = [(score_document(query, document), document.id) for document in CONTAINER_DOCUMENTS]
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_id = scored[0]
    return best_id if best_score > 0 else None


def parse_container_id(raw: str) -> int | None:
    """Parse a strict container ID from an LLM response."""
    normalized = raw.strip().upper()
    if normalized == "NONE":
        return None

    match = re.search(r"\b[012]\b", normalized)
    if match:
        return int(match.group(0))
    return None


def format_rag_context(documents: list[ContainerDocument]) -> str:
    blocks = []
    for document in documents:
        blocks.append(
            "\n".join(
                (
                    f"ID={document.id}: {document.name}",
                    f"Opis: {document.description}",
                    f"Slowa kluczowe: {', '.join(document.keywords)}",
                    f"Przyklady: {' | '.join(document.examples)}",
                )
            )
        )
    return "\n\n".join(blocks)


def normalize_ollama_base_url(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        url = f"http://{url}"
    return url.rstrip("/")


def is_ollama_available(base_url: str = OLLAMA_BASE_URL, timeout: float = 1.5) -> bool:
    url = f"{normalize_ollama_base_url(base_url)}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout):
            return True
    except (OSError, urllib.error.URLError):
        return False


def record_audio(duration: int = RECORD_SECONDS) -> Any:
    """Record audio from the default microphone."""
    import sounddevice as sd

    frames = int(duration * SAMPLE_RATE)
    recording = sd.rec(frames, samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16")
    sd.wait()
    return recording


def normalize_audio(recording: Any) -> Any:
    """Normalize audio to use full 16-bit range for best Whisper accuracy."""
    import numpy as np

    max_amp = int(np.max(np.abs(recording)))
    if max_amp == 0:
        return recording
    gain = MAX_INT16 / max_amp
    amplified = recording.astype("float32") * gain
    return np.clip(amplified, -32768, 32767).astype("int16")


def prepare_whisper_audio(recording: Any) -> Any:
    """Convert int16 microphone audio to float32 and gently lift quiet speech."""
    import numpy as np

    audio = np.squeeze(recording).astype("float32") / MAX_INT16
    peak = float(np.max(np.abs(audio)))
    if peak == 0:
        return audio

    if peak < WHISPER_TARGET_PEAK:
        audio = audio * min(WHISPER_TARGET_PEAK / peak, 8.0)

    return np.clip(audio, -1.0, 1.0)


def save_wave(path: str, recording: Any) -> None:
    """Save a NumPy int16 array to a mono WAV file."""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(recording.tobytes())


def audio_stats(recording: Any) -> tuple[int, float]:
    """Return (max_amplitude, mean_amplitude) for debugging."""
    import numpy as np

    return int(np.max(np.abs(recording))), float(np.mean(np.abs(recording)))


def is_silence(recording: Any, threshold: int = SILENCE_THRESHOLD) -> bool:
    """Return True if the recording is mostly silence."""
    import numpy as np

    mean_amp = np.mean(np.abs(recording))
    return mean_amp < threshold


def transcribe(model: Any, audio: Any) -> str:
    """Transcribe microphone audio with Whisper without requiring ffmpeg."""
    import numpy as np

    if isinstance(audio, str):
        whisper_audio = audio
    else:
        audio_array = np.squeeze(np.asarray(audio))
        if np.issubdtype(audio_array.dtype, np.floating):
            whisper_audio = audio_array.astype("float32")
        else:
            whisper_audio = audio_array.astype("float32") / MAX_INT16

    result = model.transcribe(
        whisper_audio,
        fp16=False,
        language=WHISPER_LANGUAGE,
        task="transcribe",
        initial_prompt=WHISPER_INITIAL_PROMPT,
        condition_on_previous_text=False,
        temperature=0.0,
        beam_size=WHISPER_BEAM_SIZE,
        no_speech_threshold=0.35,
        logprob_threshold=-0.8,
    )
    return result.get("text", "").strip()


class ContainerSelector:
    """Maps a voice transcript to one of the known container IDs."""

    def __init__(self, model_name: str = OLLAMA_MODEL) -> None:
        self.llm: Any | None = None
        self.last_query = ""
        self.last_candidates: list[ContainerDocument] = []
        self.using_llm = is_ollama_available()
        if self.using_llm:
            from langchain_ollama import OllamaLLM

            self.llm = OllamaLLM(
                model=model_name,
                base_url=normalize_ollama_base_url(OLLAMA_BASE_URL),
                temperature=0,
            )
        else:
            print("WARNING: Ollama is not available. Using lexical RAG fallback only.")
        self._query_prompt = (
            "Zamien wypowiedz uzytkownika na krotkie zapytanie do wyszukiwarki RAG.\n"
            "Uwzglednij rzecz lub czynnosc, ktorej uzytkownik potrzebuje.\n"
            "Odpowiedz tylko zapytaniem, bez komentarza.\n\n"
            "Wypowiedz: \"{text}\""
        )
        self._selection_prompt = (
            "Wybierz najlepszy pojemnik dla uzytkownika na podstawie kontekstu RAG.\n"
            "Mozliwe odpowiedzi: 0, 1, 2 albo NONE.\n\n"
            "Reguly:\n"
            "- ID=0, gdy chodzi o cukierki, krowki, slodycze lub cos slodkiego.\n"
            "- ID=1, gdy chodzi o dlugopis, pisanie, zapisanie, zanotowanie, podpis lub numer telefonu.\n"
            "- ID=2, gdy chodzi o karteczke, sentencje, cytat, inspiracje, studiowanie, technologie lub AI.\n"
            "- Jesli nie da sie wybrac, odpowiedz NONE.\n\n"
            "Kontekst RAG:\n"
            "{context}\n\n"
            "Wypowiedz uzytkownika: \"{text}\"\n"
            "Zapytanie RAG: \"{query}\"\n\n"
            "Odpowiedz tylko jednym tokenem: 0, 1, 2 albo NONE."
        )

    def parse(self, text: str) -> int | None:
        """Return container ID 0-2, or None if no container matches."""
        if not text:
            return None

        self.last_query = self.make_query(text)
        retrieval_query = f"{text} {self.last_query}"
        self.last_candidates = retrieve_documents(retrieval_query)
        context = format_rag_context(self.last_candidates)

        if self.llm is None:
            return best_lexical_match(retrieval_query)

        prompt = self._selection_prompt.format(
            context=context,
            text=text,
            query=self.last_query,
        )
        try:
            raw = self.llm.invoke(prompt).strip()
        except Exception as exc:
            print(f"WARNING: LLM selection failed, using lexical fallback: {exc}")
            return best_lexical_match(retrieval_query)

        parsed = parse_container_id(raw)
        if parsed is not None:
            return parsed

        return best_lexical_match(retrieval_query)

    def make_query(self, text: str) -> str:
        """Ask the LLM for a compact RAG query; fall back to the transcript."""
        if self.llm is None:
            return text

        prompt = self._query_prompt.format(text=text)
        try:
            query = self.llm.invoke(prompt).strip()
        except Exception as exc:
            print(f"WARNING: LLM query generation failed, using transcript: {exc}")
            return text

        query = " ".join(query.split())
        return query[:160] if query else text


@dataclass(frozen=True)
class VoiceSelectionResult:
    container_id: int
    transcript: str
    rag_query: str
    candidates: tuple[ContainerDocument, ...]


class VoiceContainerSelector:
    """Records one voice command and maps it to a container/marker ID."""

    def __init__(
        self,
        whisper_model_size: str = WHISPER_MODEL_SIZE,
        record_seconds: int = RECORD_SECONDS,
        save_debug_wave: bool = True,
    ) -> None:
        import whisper

        self.record_seconds = record_seconds
        self.save_debug_wave = save_debug_wave
        print(f"Loading Whisper model ({whisper_model_size})...")
        self.whisper_model = whisper.load_model(whisper_model_size)
        print("Whisper ready!")

        print(f"Checking Ollama ({OLLAMA_MODEL})...")
        self.container_selector = ContainerSelector()
        print("Voice container selector ready!")

    def ask_container_id(self, detected_ids: list[int] | None = None) -> int | None:
        """Ask the operator to record a command and return a selected ID."""
        if detected_ids:
            print(f"Detected marker IDs: {detected_ids}")
        else:
            print("No marker detected yet")

        while True:
            raw = input(
                "Press ENTER and say what you need, type an ID, or q to quit: "
            ).strip()

            if raw.lower() in {"q", "quit", "exit"}:
                return None

            if raw.isdigit():
                return int(raw)

            result = self.select_once()
            if result is None:
                print("Could not select a container. Try again.\n")
                continue

            print(f'INFO: You said: "{result.transcript}"')
            print(f'INFO: RAG query: "{result.rag_query}"')
            candidates = ", ".join(
                f"ID={doc.id} {doc.name}" for doc in result.candidates
            )
            print(f"INFO: RAG candidates: {candidates}")
            print(f"INFO: Selected container ID={result.container_id}")
            return result.container_id

    def select_once(self) -> VoiceSelectionResult | None:
        """Record, transcribe and select one container ID."""
        print("\nINFO: Recording... speak now!")
        recording = record_audio(self.record_seconds)

        max_amp, mean_amp = audio_stats(recording)
        print(f"INFO: Audio levels - max: {max_amp}, mean: {mean_amp:.1f}")

        if is_silence(recording):
            print("WARNING: No speech detected. Try speaking louder or closer to the mic.")
            return None

        normalized = normalize_audio(recording)
        n_max, n_mean = audio_stats(normalized)
        print(f"INFO: Normalized - max: {n_max}, mean: {n_mean:.1f}")

        if self.save_debug_wave:
            save_wave(WAVE_OUTPUT, normalized)

        whisper_audio = prepare_whisper_audio(recording)
        print("INFO: Transcribing...")
        text = transcribe(self.whisper_model, whisper_audio)

        if not text:
            print("WARNING: Could not understand audio.")
            return None

        print("INFO: Selecting container...")
        container_id = self.container_selector.parse(text)
        if container_id is None:
            print("ERROR: No matching container ID (0-2) detected.")
            return None

        return VoiceSelectionResult(
            container_id=container_id,
            transcript=text,
            rag_query=self.container_selector.last_query,
            candidates=tuple(self.container_selector.last_candidates),
        )

    def cleanup(self) -> None:
        if self.save_debug_wave and os.path.exists(WAVE_OUTPUT):
            os.remove(WAVE_OUTPUT)


def print_banner() -> None:
    print("=" * 60)
    print("Voice Box - wybieranie pojemnika ID")
    print("=" * 60)
    print()
    print("Pojemniki:")
    print("  ID=0 - cukierki krowki PJATK")
    print("  ID=1 - dlugopisy")
    print("  ID=2 - karteczki z sentencjami")
    print()
    print("Sterowanie:")
    print("  ENTER - nagraj komende glosowa")
    print("  q + ENTER - zakoncz")
    print("=" * 60)
    print()


def main() -> None:
    print_banner()
    selector = VoiceContainerSelector()

    while True:
        try:
            container_id = selector.ask_container_id()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if container_id is None:
            break

        action = ACTIONS.get(container_id)
        if action:
            action()
        else:
            print(f"ERROR: Detected container ID={container_id}, but no handler is defined.\n")

        print()

    selector.cleanup()


if __name__ == "__main__":
    main()
