import numpy as np

from src.speech.utterance_segmenter import (
    MIN_SPEECH_FRAMES,
    PARTIAL_INTERVAL_FRAMES,
    SILENCE_FRAMES_TO_CLOSE_UTTERANCE,
    UtteranceSegmenter,
)


def _frame(value: float) -> np.ndarray:
    """Frame de audio "falso": un array de 1 elemento con un valor
    distinguible, para poder verificar el orden/contenido exacto de lo que
    UtteranceSegmenter concatena, sin necesidad de usar el tamaño real de
    frame de producción (irrelevante para esta lógica)."""
    return np.array([value], dtype=np.float32)


def test_silence_without_prior_speech_returns_none_and_stays_idle():
    segmenter = UtteranceSegmenter()
    assert segmenter.add_frame(_frame(0.0), contains_speech=False) is None
    assert segmenter.accumulated_frames == []


def test_utterance_shorter_than_minimum_is_discarded():
    segmenter = UtteranceSegmenter()
    # Menos frames de voz que MIN_SPEECH_FRAMES.
    speech_frames = max(1, MIN_SPEECH_FRAMES - 1)
    for i in range(speech_frames):
        assert segmenter.add_frame(_frame(i), contains_speech=True) is None

    result = None
    for i in range(SILENCE_FRAMES_TO_CLOSE_UTTERANCE):
        result = segmenter.add_frame(_frame(-1), contains_speech=False)
    assert result is None
    # Se resetea igual, lista para la próxima utterance.
    assert segmenter.accumulated_frames == []


def test_utterance_with_enough_speech_closes_and_excludes_trailing_silence():
    segmenter = UtteranceSegmenter()
    for i in range(MIN_SPEECH_FRAMES):
        assert segmenter.add_frame(_frame(i), contains_speech=True) is None

    result = None
    for i in range(SILENCE_FRAMES_TO_CLOSE_UTTERANCE):
        result = segmenter.add_frame(_frame(-1), contains_speech=False)

    assert result is not None
    # El resultado debe tener exactamente los frames de voz, sin ninguno de
    # los frames de silencio de cierre.
    assert list(result) == list(range(MIN_SPEECH_FRAMES))


def test_brief_silence_mid_utterance_does_not_close_and_audio_is_kept():
    segmenter = UtteranceSegmenter()
    for i in range(MIN_SPEECH_FRAMES):
        assert segmenter.add_frame(_frame(i), contains_speech=True) is None

    # Silencio breve (menos del umbral de cierre) intercalado.
    brief_silence_count = max(1, SILENCE_FRAMES_TO_CLOSE_UTTERANCE - 1)
    for _ in range(brief_silence_count):
        assert segmenter.add_frame(_frame(-2), contains_speech=False) is None

    # Vuelve la voz: el silencio breve NO debe haber cerrado la utterance.
    assert segmenter.add_frame(_frame(999), contains_speech=True) is None
    assert segmenter.consecutive_silence_frame_count == 0

    # Ahora sí cerramos con suficiente silencio.
    result = None
    for _ in range(SILENCE_FRAMES_TO_CLOSE_UTTERANCE):
        result = segmenter.add_frame(_frame(-3), contains_speech=False)

    assert result is not None
    # El audio del silencio breve intercalado (valor -2) debe seguir
    # presente — solo se recorta el silencio de CIERRE.
    assert -2.0 in result
    assert 999.0 in result
    assert -3.0 not in result


def test_reset_after_close_allows_new_utterance():
    segmenter = UtteranceSegmenter()
    for i in range(MIN_SPEECH_FRAMES):
        segmenter.add_frame(_frame(i), contains_speech=True)
    for _ in range(SILENCE_FRAMES_TO_CLOSE_UTTERANCE):
        segmenter.add_frame(_frame(-1), contains_speech=False)

    assert segmenter.accumulated_frames == []
    assert segmenter.speech_frame_count == 0
    assert segmenter.consecutive_silence_frame_count == 0
    assert segmenter.frame_count_at_last_partial_emit == 0


def test_maybe_get_partial_audio_returns_none_when_idle():
    segmenter = UtteranceSegmenter()
    assert segmenter.maybe_get_partial_audio() is None


def test_maybe_get_partial_audio_waits_for_interval_then_emits_once():
    segmenter = UtteranceSegmenter()
    for i in range(PARTIAL_INTERVAL_FRAMES - 1):
        segmenter.add_frame(_frame(i), contains_speech=True)

    # Todavía no pasó suficiente audio nuevo.
    assert segmenter.maybe_get_partial_audio() is None

    segmenter.add_frame(_frame(PARTIAL_INTERVAL_FRAMES - 1), contains_speech=True)
    partial = segmenter.maybe_get_partial_audio()
    assert partial is not None
    assert list(partial) == list(range(PARTIAL_INTERVAL_FRAMES))

    # Sin audio nuevo desde el último parcial: vuelve a dar None.
    assert segmenter.maybe_get_partial_audio() is None


def test_maybe_get_partial_audio_does_not_close_the_utterance():
    segmenter = UtteranceSegmenter()
    for i in range(PARTIAL_INTERVAL_FRAMES):
        segmenter.add_frame(_frame(i), contains_speech=True)

    segmenter.maybe_get_partial_audio()

    # La utterance sigue abierta: el estado interno no debe haberse tocado.
    assert segmenter.speech_frame_count == PARTIAL_INTERVAL_FRAMES
    assert len(segmenter.accumulated_frames) == PARTIAL_INTERVAL_FRAMES
