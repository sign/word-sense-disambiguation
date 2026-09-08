import pytest
import requests_mock

from wsd.env import WORDNET_URL
from wsd.masked_language_model import load_model
from wsd.word_sense_disambiguation import (
    NO_DEFINITIONS_FOUND,
    NONE_OF_THE_ABOVE,
    Definition,
    DisambiguationInput,
    DisambiguationResult,
    WordQuery,
    create_multiple_choice_prompt,
    disambiguate_word_batch,
    get_definitions,
)


@pytest.fixture(autouse=True)
def _clear_definitions_cache():
    from wsd import word_sense_disambiguation

    word_sense_disambiguation._definitions_cache.clear()


def test_word_query_dataclass():
    """Test WordQuery dataclass creation"""
    query = WordQuery(form="bank", pos="n")
    assert query.form == "bank"
    assert query.pos == "n"


def test_definition_dataclass():
    """Test Definition dataclass creation"""
    definition = Definition(
        synset_id="omw-en-1234-n",
        definition="a financial institution"
    )
    assert definition.synset_id == "omw-en-1234-n"
    assert definition.definition == "a financial institution"


def test_disambiguation_result_dataclass():
    """Test DisambiguationResult dataclass creation"""
    result = DisambiguationResult(
        synset_id="omw-en-1234-n",
        definition="a financial institution",
        confidence=0.85
    )
    assert result.synset_id == "omw-en-1234-n"
    assert result.definition == "a financial institution"
    assert result.confidence == 0.85


def test_disambiguation_input_dataclass():
    """Test DisambiguationInput dataclass creation"""
    definitions = [
        Definition(synset_id="omw-en-1234-n", definition="financial institution"),
        Definition(synset_id="omw-en-5678-n", definition="edge of river"),
    ]
    input_obj = DisambiguationInput(
        marked_sentence="I went to the *bank* to withdraw money.",
        definitions=definitions
    )
    assert len(input_obj.definitions) == 2


def test_get_definitions_empty_list():
    """Test get_definitions with empty list"""
    result = get_definitions([])
    assert result == []


def test_get_definitions_success():
    """Test get_definitions with successful API response"""
    with requests_mock.Mocker() as m:
        # Mock the API endpoint
        url = f"{WORDNET_URL}/lexicons/omw-en:1.4/definitions"
        mock_response = {
            "data": [
                {
                    "definitions": {
                        "omw-en-1234-n": "a financial institution",
                        "omw-en-5678-n": "the edge of a river"
                    }
                },
                {
                    "definitions": {
                        "omw-en-9999-v": "to run quickly"
                    }
                }
            ]
        }
        m.post(url, json=mock_response)

        queries = [
            WordQuery(form="bank", pos="n"),
            WordQuery(form="run", pos="v")
        ]
        results = get_definitions(queries)

        assert len(results) == 2
        assert len(results[0]) == 2
        assert len(results[1]) == 1
        assert results[0][0].synset_id == "omw-en-1234-n"
        assert results[0][0].definition == "a financial institution"


def test_get_definitions_api_error():
    """Test get_definitions with API error"""
    with requests_mock.Mocker() as m:
        url = f"{WORDNET_URL}/lexicons/omw-en:1.4/definitions"
        m.post(url, status_code=500)

        queries = [WordQuery(form="bank", pos="n")]
        results = get_definitions(queries)

        # Should return empty list for failed query
        assert len(results) == 1
        assert results[0] == []


def test_create_multiple_choice_prompt():
    """Test create_multiple_choice_prompt function"""
    components = load_model()
    definitions = [
        Definition(synset_id="omw-en-1234-n", definition="a financial institution"),
        Definition(synset_id="omw-en-5678-n", definition="the edge of a river"),
    ]

    prompt = create_multiple_choice_prompt(
        components.tokenizer.mask_token,
        "I went to the *bank*.",
        definitions,
        components.tokenizer,
    )

    from wsd.letters import NOTA_LETTER_INDEX, build_letters
    nota_letter = build_letters(components.tokenizer).letters[NOTA_LETTER_INDEX]
    assert "bank" in prompt.lower()
    assert "A. a financial institution" in prompt
    assert "B. the edge of a river" in prompt
    assert f"{nota_letter}. {NONE_OF_THE_ABOVE}" in prompt
    assert components.tokenizer.mask_token in prompt


def test_result_from_probs():
    """Option i is letter i; "none of the above" is the fixed last letter; confidence is renormalized."""
    from wsd.letters import NOTA_LETTER_INDEX
    from wsd.word_sense_disambiguation import _result_from_probs
    probs = [0.0] * 128
    probs[0], probs[1], probs[NOTA_LETTER_INDEX] = 0.8, 0.1, 0.1
    definitions = [Definition("a", "definition 1"), Definition("b", "definition 2")]
    result = _result_from_probs(probs, definitions)
    assert (result.synset_id, result.confidence) == ("a", pytest.approx(0.8))
    probs[NOTA_LETTER_INDEX] = 0.9
    assert _result_from_probs(probs, definitions).definition == NONE_OF_THE_ABOVE


def test_disambiguate_word_no_definitions():
    """Test disambiguate_word with no definitions"""
    result = disambiguate_word_batch([DisambiguationInput("This is a *test*.", [])])[0]

    assert result.synset_id == NO_DEFINITIONS_FOUND
    assert result.definition == ""
    assert result.confidence == 0.0


def test_disambiguate_word_with_definitions():
    """Test disambiguate_word with real definitions"""
    definitions = [
        Definition(synset_id="omw-en-1234-n", definition="a financial institution"),
        Definition(synset_id="omw-en-5678-n", definition="the edge of a river"),
    ]

    result = disambiguate_word_batch([DisambiguationInput("I went to the *bank* to withdraw money.", definitions)])[0]

    # Should return a valid result
    assert isinstance(result, DisambiguationResult)
    assert isinstance(result.synset_id, str)
    assert isinstance(result.definition, str)
    assert 0.0 <= result.confidence <= 1.0


def test_disambiguate_word_batch_empty():
    """Test disambiguate_word_batch with empty list"""
    results = disambiguate_word_batch([])
    assert results == []


def test_disambiguate_word_batch_no_definitions():
    """Test disambiguate_word_batch when inputs have no definitions"""
    batch_data = [
        DisambiguationInput(marked_sentence="This is a *test*.", definitions=[]),
        DisambiguationInput(marked_sentence="This is an *example*.", definitions=[]),
    ]

    results = disambiguate_word_batch(batch_data)

    assert len(results) == 2
    for result in results:
        assert result.synset_id == NO_DEFINITIONS_FOUND
        assert result.confidence == 0.0


def test_disambiguate_word_batch_with_definitions():
    """Test disambiguate_word_batch with real definitions"""
    definitions1 = [
        Definition(synset_id="omw-en-1234-n", definition="a financial institution"),
        Definition(synset_id="omw-en-5678-n", definition="the edge of a river"),
    ]
    definitions2 = [
        Definition(synset_id="omw-en-9999-v", definition="to move quickly on foot"),
    ]

    batch_data = [
        DisambiguationInput(
            marked_sentence="I went to the *bank* to withdraw money.",
            definitions=definitions1
        ),
        DisambiguationInput(
            marked_sentence="I need to *run* to catch the bus.",
            definitions=definitions2
        ),
    ]

    results = disambiguate_word_batch(batch_data)

    assert len(results) == 2
    for result in results:
        assert isinstance(result, DisambiguationResult)
        assert isinstance(result.synset_id, str)
        assert result.synset_id != NO_DEFINITIONS_FOUND
        assert 0.0 <= result.confidence <= 1.0


def test_disambiguate_word_batch_mixed():
    """Test batch processing with mix of valid and empty definitions"""
    definitions = [
        Definition(synset_id="omw-en-1234-n", definition="a financial institution"),
    ]

    batch_data = [
        DisambiguationInput(
            marked_sentence="I went to the *bank*.",
            definitions=definitions
        ),
        DisambiguationInput(
            marked_sentence="This is *xyz*.",
            definitions=[]
        ),
    ]

    results = disambiguate_word_batch(batch_data)

    assert len(results) == 2
    assert results[0].synset_id != NO_DEFINITIONS_FOUND
    assert results[1].synset_id == NO_DEFINITIONS_FOUND


def test_constants():
    """Test that constants are defined correctly"""
    assert isinstance(NO_DEFINITIONS_FOUND, str)
    assert isinstance(NONE_OF_THE_ABOVE, str)
    assert len(NO_DEFINITIONS_FOUND) > 0
    assert len(NONE_OF_THE_ABOVE) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
