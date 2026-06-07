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
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "").strip() or None
WHISPER_BEAM_SIZE = 5
WHISPER_TARGET_PEAK = 0.85
WHISPER_INITIAL_PROMPT = (
    "Polish and English vocabulary: krowki PJATK, cukierki, slodycze, candy, "
    "sweets, snack, treat, dlugopis, dlugopisy, pen, pens, write, notes, "
    "phone number, signature, karteczki, sentencje, cytat, quote, wisdom, "
    "motivation, studying, technology, AI, losowy produkt, random product, "
    "olowek, olowki, pencil, pencils, draw, drawing, sketch, naklejki, stickers, labels, klikacz, klikacze, "
    "breloczki, opaski na nadgarstek, wristbands, clicker, fidget, keychain, "
    "stress relief, antystres."
)
OLLAMA_MODEL = "qwen2.5:0.5b"
OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")
WAVE_OUTPUT = "command.wav"
RAG_TOP_K = 3
RANDOM_PRODUCT_CONTAINER_ID = 3
RANDOM_PRODUCT_TRIGGERS = (
    "losowy produkt",
    "losowego produktu",
    "losowy przedmiot",
    "losowa rzecz",
    "random product",
    "random item",
)
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

STOP_WORDS = {
    "a",
    "an",
    "and",
    "can",
    "could",
    "for",
    "get",
    "give",
    "have",
    "i",
    "me",
    "need",
    "please",
    "some",
    "something",
    "the",
    "thing",
    "things",
    "to",
    "want",
    "with",
    "you",
    "chce",
    "chcialbym",
    "cos",
    "daj",
    "dla",
    "do",
    "jakas",
    "jakie",
    "jakies",
    "jakis",
    "mam",
    "mi",
    "moge",
    "musze",
    "poprosze",
    "potrzebuje",
    "prosze",
    "sobie",
}

QUERY_EXPANSIONS: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        (
            "cos slodkiego",
            "slodkiego",
            "slodycze",
            "cukierka",
            "cukierek",
            "krowka",
            "something sweet",
            "sweet",
            "sweets",
            "candy",
            "snack",
            "treat",
            "dessert",
            "sugar",
        ),
        "cukierki krowki PJATK candy sweets sweet snack treat dessert sugar",
    ),
    (
        (
            "dlugopis",
            "pisac",
            "zapisac",
            "zanotowac",
            "notatki",
            "podpis",
            "numer telefonu",
            "pen",
            "pens",
            "write",
            "writing",
            "take notes",
            "note taking",
            "phone number",
            "signature",
            "sign",
        ),
        "dlugopisy pen pens write writing notes phone number signature sign",
    ),
    (
        (
            "karteczka",
            "karteczki",
            "sentencja",
            "sentencje",
            "cytat",
            "inspiracja",
            "motywacja",
            "madrosc",
            "quote",
            "quotes",
            "wisdom",
            "inspiration",
            "motivation",
            "thought",
            "saying",
            "aphorism",
        ),
        "karteczki sentencje cytat quote wisdom inspiration motivation thought saying",
    ),
    (
        (
            "olowek",
            "olowki",
            "rysowac",
            "rysowanie",
            "szkic",
            "szkicowac",
            "losowy produkt",
            "losowego produktu",
            "losowy przedmiot",
            "losowa rzecz",
            "random product",
            "random item",
            "pencil",
            "pencils",
            "draw",
            "drawing",
            "sketch",
            "sketching",
            "art",
            "design",
        ),
        "olowki do rysowania pencil pencils random product random item draw drawing sketch art design",
    ),
    (
        (
            "naklejka",
            "naklejki",
            "przykleic",
            "ozdobic",
            "dekorowac",
            "etykieta",
            "sticker",
            "stickers",
            "label",
            "labels",
            "decorate",
            "decoration",
            "decal",
        ),
        "naklejki sticker stickers label labels decorate decoration decal",
    ),
    (
        (
            "klikacz",
            "klikacze",
            "klikac",
            "breloczek",
            "breloczki",
            "opaska",
            "opaski",
            "opaska na nadgarstek",
            "opaski na nadgarstek",
            "nadgarstek",
            "odstresowac",
            "odstresowujace",
            "antystres",
            "clicker",
            "clickers",
            "fidget",
            "keychain",
            "keychains",
            "wristband",
            "wristbands",
            "wrist band",
            "wrist bands",
            "stress",
            "stress relief",
            "anti stress",
            "toy",
            "relax",
            "calm down",
        ),
        "klikacze breloczki opaski na nadgarstek odstresowujace clicker fidget keychain wristband stress relief anti stress toy relax",
    ),
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
        name="Cukierki PJATK",
        description=(
            "Pojemnik z cukierkami krowkami PJATK. Dobry wybor, gdy uzytkownik "
            "chce cos slodkiego, przekaske, cukierka albo nagrode. Choose this "
            "container for candy, sweets, a sweet snack, dessert or a small treat."
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
            "sweet thing",
            "something sweet",
            "snack",
            "treat",
            "dessert",
            "sugar",
            "sugary",
            "chocolate",
        ),
        examples=(
            "Mam ochote na cos slodkiego.",
            "Daj mi cukierka.",
            "Poprosze krowke z PJATK.",
            "I want something sweet.",
            "Can I get some candy?",
            "I need a sweet snack.",
        ),
    ),
    ContainerDocument(
        id=1,
        name="Dlugopisy",
        description=(
            "Pojemnik z dlugopisami. Dobry wybor, gdy uzytkownik chce cos "
            "zapisac, zanotowac, podpisac, napisac numer telefonu albo robic notatki. "
            "Choose this container for pens, writing, note taking, signing or "
            "writing down a phone number."
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
            "write down",
            "note",
            "notes",
            "take notes",
            "note taking",
            "notebook",
            "phone",
            "number",
            "phone number",
            "sign",
            "signature",
            "autograph",
        ),
        examples=(
            "Chcialbym zapisac sobie numer telefonu.",
            "Musze cos zanotowac.",
            "Potrzebuje dlugopisu do podpisu.",
            "I need to write down a phone number.",
            "I need a pen.",
            "I want to take notes.",
        ),
    ),
    ContainerDocument(
        id=2,
        name="Karteczki z sentencjami",
        description=(
            "Pojemnik z wymieszanymi karteczkami z madrymi sentencjami o "
            "studiowaniu, technologii i AI. Dobry wybor, gdy uzytkownik szuka "
            "inspiracji, motywacji, cytatu, mysli albo sentencji. Choose this "
            "container for quote cards, wisdom, inspiration, motivation or a "
            "short thought about studying, technology or AI."
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
            "quotes",
            "quote card",
            "quote cards",
            "wisdom",
            "thought",
            "saying",
            "aphorism",
            "message",
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
            "I need some inspiration.",
            "Give me a motivational thought.",
        ),
    ),
    ContainerDocument(
        id=3,
        name="Olowki do rysowania",
        description=(
            "Pojemnik z olowkami do rysowania, szkicowania i zaznaczania. "
            "Dobry wybor, gdy uzytkownik chce rysowac, szkicowac, projektowac "
            "albo potrzebuje olowka zamiast dlugopisu. Choose this container "
            "for pencils, drawing, sketching, art or design."
        ),
        keywords=(
            "olowek",
            "olowki",
            "rysowanie",
            "rysowac",
            "narysowac",
            "szkic",
            "szkicowac",
            "szkicowanie",
            "projekt",
            "projektowac",
            "zaznaczac",
            "pencil",
            "pencils",
            "draw",
            "drawing",
            "sketch",
            "sketching",
            "art",
            "design",
            "designer",
            "draft",
        ),
        examples=(
            "Chce cos narysowac.",
            "Potrzebuje olowka do szkicu.",
            "Daj mi cos do rysowania.",
            "I need a pencil for drawing.",
            "I want to draw something.",
            "I need something for sketching.",
        ),
    ),
    ContainerDocument(
        id=4,
        name="Naklejki",
        description=(
            "Pojemnik z naklejkami. Dobry wybor, gdy uzytkownik chce cos "
            "ozdobic, przykleic, oznaczyc, udekorowac albo potrzebuje naklejki. "
            "Choose this container for stickers, labels, decals, decorating "
            "or marking something."
        ),
        keywords=(
            "naklejka",
            "naklejki",
            "przykleic",
            "przyklejanie",
            "ozdoba",
            "ozdobic",
            "dekoracja",
            "dekorowac",
            "oznaczyc",
            "etykieta",
            "sticker",
            "stickers",
            "label",
            "labels",
            "decorate",
            "decoration",
            "decal",
            "decals",
            "mark",
            "tag",
        ),
        examples=(
            "Chce naklejke.",
            "Potrzebuje czegos do ozdobienia zeszytu.",
            "Daj mi naklejki.",
            "I want stickers.",
            "I need a label.",
            "Give me something to decorate with.",
        ),
    ),
    ContainerDocument(
        id=5,
        name="Klikacze breloczki i opaski na nadgarstek",
        description=(
            "Pojemnik z klikaczami, breloczkami oraz opaskami na nadgarstek. "
            "Dobry wybor, gdy uzytkownik chce cos klikac, bawic sie w dloni, "
            "odstresowac sie, potrzebuje breloczka antystresowego albo opaski "
            "na nadgarstek. Choose this container for clickers, fidget toys, "
            "stress relief keychains, wristbands, anti-stress toys or something "
            "calming to hold."
        ),
        keywords=(
            "klikacz",
            "klikacze",
            "klikac",
            "breloczek",
            "breloczki",
            "opaska",
            "opaski",
            "opaska na nadgarstek",
            "opaski na nadgarstek",
            "nadgarstek",
            "odstresowujacy",
            "odstresowujace",
            "stres",
            "antystres",
            "antystresowy",
            "relaks",
            "zabawka",
            "fidget",
            "clicker",
            "clickers",
            "keychain",
            "keychains",
            "wristband",
            "wristbands",
            "wrist band",
            "wrist bands",
            "bracelet",
            "stress",
            "anti stress",
            "antistress",
            "stress relief",
            "stress reliever",
            "calm",
            "calm down",
            "relax",
            "relaxing",
            "toy",
            "fidget toy",
        ),
        examples=(
            "Potrzebuje czegos odstresowujacego.",
            "Daj mi klikacz.",
            "Chce breloczek antystresowy.",
            "Poprosze opaske na nadgarstek.",
            "I need a stress relief clicker.",
            "I want a fidget toy.",
            "I need a wristband.",
            "Give me something calming.",
        ),
    ),
)


def action_candy() -> None:
    print("[ACTION id=0] Wybrano pojemnik: cukierki krowki PJATK.")


def action_pen() -> None:
    print("[ACTION id=1] Wybrano pojemnik: dlugopisy.")


def action_quote_cards() -> None:
    print("[ACTION id=2] Wybrano pojemnik: karteczki z sentencjami.")


def action_pencil() -> None:
    print("[ACTION id=3] Wybrano pojemnik: olowki do rysowania.")


def action_stickers() -> None:
    print("[ACTION id=4] Wybrano pojemnik: naklejki.")


def action_clickers() -> None:
    print("[ACTION id=5] Wybrano pojemnik: klikacze, breloczki i opaski na nadgarstek.")


ACTIONS: dict[int, Callable[[], None]] = {
    0: action_candy,
    1: action_pen,
    2: action_quote_cards,
    3: action_pencil,
    4: action_stickers,
    5: action_clickers,
}


def normalize_text(text: str) -> str:
    """Normalize text for simple lexical retrieval."""
    text = text.lower().translate(POLISH_TRANSLATION)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9 ]+", " ", text)


def tokenize(text: str) -> set[str]:
    """Return searchable tokens without very short filler words."""
    return {
        token
        for token in normalize_text(text).split()
        if len(token) > 2 and token not in STOP_WORDS
    }


def phrase_matches(text: str, phrases: tuple[str, ...]) -> set[str]:
    """Return normalized keyword phrases that appear in text."""
    normalized_text = normalize_text(text)
    text_tokens = set(normalized_text.split())
    matches = set()

    for phrase in phrases:
        normalized_phrase = normalize_text(phrase).strip()
        if not normalized_phrase:
            continue

        phrase_tokens = normalized_phrase.split()
        if not phrase_tokens:
            continue

        if len(phrase_tokens) == 1:
            if phrase_tokens[0] in text_tokens:
                matches.add(normalized_phrase)
            continue

        if normalized_phrase in normalized_text:
            matches.add(normalized_phrase)

    return matches


def is_random_product_request(text: str) -> bool:
    """Return True when the user asks for a random product."""
    return bool(phrase_matches(text, RANDOM_PRODUCT_TRIGGERS))


def expand_search_text(text: str) -> str:
    """Add bilingual synonyms for short Polish/English voice commands."""
    normalized_text = normalize_text(text)
    text_tokens = set(normalized_text.split())
    expansions = []

    for triggers, expansion in QUERY_EXPANSIONS:
        for trigger in triggers:
            normalized_trigger = normalize_text(trigger).strip()
            if not normalized_trigger:
                continue

            trigger_tokens = set(normalized_trigger.split())
            if len(trigger_tokens) == 1 and normalized_trigger in text_tokens:
                expansions.append(expansion)
                break

            if len(trigger_tokens) > 1 and normalized_trigger in normalized_text:
                expansions.append(expansion)
                break

    return " ".join((text, *expansions))


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
    """Score one document against the query using bilingual lexical signals."""
    expanded_query = expand_search_text(query)
    query_tokens = tokenize(expanded_query)
    if not query_tokens:
        return 0

    doc_tokens = tokenize(document_text(document))
    keyword_tokens = {token for keyword in document.keywords for token in tokenize(keyword)}
    keyword_phrase_hits = phrase_matches(expanded_query, document.keywords)
    name_phrase_hits = phrase_matches(expanded_query, (document.name,))
    example_phrase_hits = phrase_matches(expanded_query, document.examples)

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

    return (
        len(exact_hits)
        + (3 * len(keyword_hits))
        + (5 * len(keyword_phrase_hits))
        + (4 * len(name_phrase_hits))
        + (2 * len(example_phrase_hits))
        + len(stem_hits)
    )


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

    match = re.search(r"\b[0-5]\b", normalized)
    if match:
        return int(match.group(0))
    return None


def is_known_container_id(container_id: int) -> bool:
    return any(document.id == container_id for document in CONTAINER_DOCUMENTS)


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


def audio_visualization_level(recording: Any) -> float:
    """Return a browser-friendly 0-255 level for a short audio chunk."""
    import numpy as np

    audio = np.asarray(recording).astype("float32")
    if audio.size == 0:
        return 0.0

    rms = float(np.sqrt(np.mean(np.square(audio)))) / MAX_INT16
    return min(255.0, rms * 512.0)


def record_audio(
    duration: int = RECORD_SECONDS,
    level_callback: Callable[[float], None] | None = None,
) -> Any:
    """Record audio from the default microphone."""
    import numpy as np
    import sounddevice as sd

    target_frames = int(duration * SAMPLE_RATE)

    if level_callback is None:
        recording = sd.rec(
            target_frames,
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
        )
        sd.wait()
        return recording

    chunks: list[Any] = []
    recorded_frames = 0

    def callback(indata: Any, frames: int, time_info: Any, status: Any) -> None:
        nonlocal recorded_frames

        remaining = target_frames - recorded_frames
        if remaining <= 0:
            raise sd.CallbackStop()

        chunk = indata[:remaining].copy()
        chunks.append(chunk)
        recorded_frames += len(chunk)

        if level_callback is not None:
            level_callback(audio_visualization_level(chunk))

        if recorded_frames >= target_frames:
            raise sd.CallbackStop()

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=512,
            callback=callback,
        ) as stream:
            while stream.active and recorded_frames < target_frames:
                sd.sleep(50)
    finally:
        if level_callback is not None:
            level_callback(0.0)

    if not chunks:
        return np.zeros((target_frames, CHANNELS), dtype="int16")

    recording = np.concatenate(chunks, axis=0)
    if len(recording) < target_frames:
        missing = target_frames - len(recording)
        padding = np.zeros((missing, CHANNELS), dtype=recording.dtype)
        recording = np.concatenate((recording, padding), axis=0)

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

    transcribe_options = {
        "fp16": False,
        "task": "transcribe",
        "initial_prompt": WHISPER_INITIAL_PROMPT,
        "condition_on_previous_text": False,
        "temperature": 0.0,
        "beam_size": WHISPER_BEAM_SIZE,
        "no_speech_threshold": 0.35,
        "logprob_threshold": -0.8,
    }
    if WHISPER_LANGUAGE is not None:
        transcribe_options["language"] = WHISPER_LANGUAGE

    result = model.transcribe(whisper_audio, **transcribe_options)
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
            "Convert the user's Polish or English utterance into a short RAG search query.\n"
            "Keep the object, purpose and activity the user needs. You may include both "
            "Polish and English synonyms if useful.\n"
            "Answer only with the query, no commentary.\n\n"
            "Utterance: \"{text}\""
        )
        self._selection_prompt = (
            "Choose the best container for a Polish or English user request using the RAG context.\n"
            "Allowed answers: 0, 1, 2, 3, 4, 5 or NONE.\n\n"
            "Rules:\n"
            "- ID=0 for candy, fudge, sweets, snacks, treats or something sweet.\n"
            "- ID=1 for pens, writing, notes, signing or writing down a phone number.\n"
            "- ID=2 for quote cards, wisdom, sayings, inspiration, motivation, studying, technology or AI.\n"
            "- ID=3 for pencils, drawing, sketching, art, design, drafting or a random product request.\n"
            "- ID=4 for stickers, labels, decals, decorating, marking or tagging.\n"
            "- ID=5 for clickers, fidget toys, keychains, wristbands, stress relief, anti-stress or relaxing hand toys.\n"
            "- If no container matches, answer NONE.\n\n"
            "RAG context:\n"
            "{context}\n\n"
            "User utterance: \"{text}\"\n"
            "RAG query: \"{query}\"\n\n"
            "Answer with exactly one token: 0, 1, 2, 3, 4, 5 or NONE."
        )

    def parse(self, text: str) -> int | None:
        """Return container ID 0-5, or None if no container matches."""
        if not text:
            return None

        if is_random_product_request(text):
            self.last_query = "losowy produkt olowek pencil random product"
            self.last_candidates = [
                document
                for document in CONTAINER_DOCUMENTS
                if document.id == RANDOM_PRODUCT_CONTAINER_ID
            ]
            return RANDOM_PRODUCT_CONTAINER_ID

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
        audio_visualizer: Any | None = None,
    ) -> None:
        import whisper

        self.record_seconds = record_seconds
        self.save_debug_wave = save_debug_wave
        self.audio_visualizer = audio_visualizer
        print(f"Loading Whisper model ({whisper_model_size})...")
        self.whisper_model = whisper.load_model(whisper_model_size)
        print("Whisper ready!")

        print(f"Checking Ollama ({OLLAMA_MODEL})...")
        self.container_selector = ContainerSelector()
        print("Voice container selector ready!")
        self._publish_status("idle", "Ready for a voice command")

    def _publish_status(self, phase: str, message: str, **payload: Any) -> None:
        if self.audio_visualizer is None:
            return

        self.audio_visualizer.broadcast_status(phase, message, **payload)

    def ask_container_id(self, detected_ids: list[int] | None = None) -> int | None:
        """Ask the operator to record a command and return a selected ID."""
        if detected_ids:
            print(f"Detected marker IDs: {detected_ids}")
        else:
            print("No marker detected yet")

        if self._can_use_web_trigger():
            return self._ask_container_id_from_web(detected_ids=detected_ids)

        while True:
            self._publish_status(
                "idle",
                "Press ENTER and say what you need",
                detected_ids=detected_ids or [],
            )
            raw = input(
                "Press ENTER and say what you need, type an ID, or q to quit: "
            ).strip()

            if raw.lower() in {"q", "quit", "exit"}:
                self._publish_status("stopped", "Stopped")
                return None

            if raw.isdigit():
                container_id = int(raw)
                if not is_known_container_id(container_id):
                    print(f"Unknown container ID={container_id}. Try 0-5.")
                    continue
                self._publish_status(
                    "selected",
                    f"Manually selected ID={container_id}",
                    selected_id=container_id,
                )
                return container_id

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

    def _can_use_web_trigger(self) -> bool:
        return (
            self.audio_visualizer is not None
            and hasattr(self.audio_visualizer, "wait_for_command")
            and hasattr(self.audio_visualizer, "clear_commands")
        )

    def _ask_container_id_from_web(
        self,
        detected_ids: list[int] | None = None,
    ) -> int | None:
        self.audio_visualizer.clear_commands()
        self._publish_status(
            "idle",
            "Ready for a voice command",
            detected_ids=detected_ids or [],
        )
        print("Open the web panel and press Enter or Start Listening.")
        print("Use Ctrl+C in this terminal to quit.")

        while True:
            command = self.audio_visualizer.wait_for_command(timeout=0.25)
            if command is None:
                continue

            command_name = command.get("command")

            if command_name == "quit":
                self._publish_status("stopped", "Stopped")
                return None

            if command_name == "select_id":
                try:
                    container_id = int(command.get("id"))
                except (TypeError, ValueError):
                    continue

                if not is_known_container_id(container_id):
                    self._publish_status(
                        "error",
                        f"Unknown container ID={container_id}",
                    )
                    continue

                self._publish_status(
                    "selected",
                    f"Manually selected ID={container_id}",
                    selected_id=container_id,
                )
                return container_id

            if command_name != "start_listening":
                continue

            result = self.select_once()
            if result is None:
                print("Could not select a container. Try again.\n")
                self._publish_status("idle", "Ready for a voice command")
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
        self._publish_status("listening", "Listening")
        level_callback = None
        if self.audio_visualizer is not None:
            level_callback = self.audio_visualizer.broadcast_audio_level

        recording = record_audio(self.record_seconds, level_callback=level_callback)

        self._publish_status("checking_audio", "Checking recording level")
        max_amp, mean_amp = audio_stats(recording)
        print(f"INFO: Audio levels - max: {max_amp}, mean: {mean_amp:.1f}")

        if is_silence(recording):
            print("WARNING: No speech detected. Try speaking louder or closer to the mic.")
            self._publish_status(
                "error",
                "No speech detected. Try speaking louder",
                max_amplitude=max_amp,
                mean_amplitude=round(mean_amp, 1),
            )
            return None

        self._publish_status("normalizing", "Preparing recording")
        normalized = normalize_audio(recording)
        n_max, n_mean = audio_stats(normalized)
        print(f"INFO: Normalized - max: {n_max}, mean: {n_mean:.1f}")

        if self.save_debug_wave:
            save_wave(WAVE_OUTPUT, normalized)

        whisper_audio = prepare_whisper_audio(recording)
        print("INFO: Transcribing...")
        self._publish_status("transcribing", "Transcribing speech")
        text = transcribe(self.whisper_model, whisper_audio)

        if not text:
            print("WARNING: Could not understand audio.")
            self._publish_status(
                "error",
                "Could not understand the recording",
            )
            return None

        print("INFO: Selecting container...")
        self._publish_status(
            "selecting",
            "Choosing the best container",
            transcript=text,
        )
        container_id = self.container_selector.parse(text)
        if container_id is None:
            print("ERROR: No matching container ID (0-5) detected.")
            self._publish_status(
                "error",
                "No matching container found",
                transcript=text,
                rag_query=self.container_selector.last_query,
            )
            return None

        candidates = [
            {"id": document.id, "name": document.name}
            for document in self.container_selector.last_candidates
        ]
        self._publish_status(
            "selected",
            f"Selected container ID={container_id}",
            selected_id=container_id,
            transcript=text,
            rag_query=self.container_selector.last_query,
            candidates=candidates,
        )

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
    print("  ID=3 - olowki do rysowania")
    print("  ID=4 - naklejki")
    print("  ID=5 - klikacze, breloczki i opaski na nadgarstek")
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
