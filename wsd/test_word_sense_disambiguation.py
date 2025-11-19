import pytest
import requests_mock

from wsd.word_sense_disambiguation import (
    WordQuery,
    Definition,
    DisambiguationInput,
    DisambiguationResult,
    get_definitions,
    get_definitions_single,
    create_marked_sentence,
    create_multiple_choice_prompt,
    get_choice_probabilities,
    disambiguate_word,
    disambiguate_word_batch,
    NO_DEFINITIONS_FOUND,
    NONE_OF_THE_ABOVE,
)
from wsd.env import WORDNET_URL
from wsd.masked_language_model import load_model


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
        word="bank",
        marked_sentence="I went to the *bank* to withdraw money.",
        definitions=definitions
    )
    assert input_obj.word == "bank"
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


def test_get_definitions_single():
    """Test get_definitions_single convenience function"""
    with requests_mock.Mocker() as m:
        url = f"{WORDNET_URL}/lexicons/omw-en:1.4/definitions"
        mock_response = {
            "data": [
                {
                    "definitions": {
                        "omw-en-1234-n": "a financial institution"
                    }
                }
            ]
        }
        m.post(url, json=mock_response)

        results = get_definitions_single("bank", "n")

        assert len(results) == 1
        assert results[0].synset_id == "omw-en-1234-n"


def test_create_marked_sentence(monkeypatch):
    """Test create_marked_sentence function"""
    import spacy
    from unittest.mock import Mock

    # Create a mock spacy doc
    nlp = spacy.blank("en")
    doc = nlp("I went to the bank")

    marked = create_marked_sentence(doc, 4)  # Mark "bank"
    assert "*bank*" in marked
    assert "I went to the" in marked


def test_create_multiple_choice_prompt():
    """Test create_multiple_choice_prompt function"""
    components = load_model()
    definitions = [
        Definition(synset_id="omw-en-1234-n", definition="a financial institution"),
        Definition(synset_id="omw-en-5678-n", definition="the edge of a river"),
    ]

    prompt = create_multiple_choice_prompt(
        "bank",
        components.tokenizer.mask_token,
        "I went to the *bank*.",
        definitions
    )

    assert "bank" in prompt.lower()
    assert "A: a financial institution" in prompt
    assert "B: the edge of a river" in prompt
    assert f"C: {NONE_OF_THE_ABOVE}" in prompt
    assert components.tokenizer.mask_token in prompt


def test_get_choice_probabilities():
    """Test get_choice_probabilities function"""
    import torch
    components = load_model()

    # Create mock probabilities
    vocab_size = components.tokenizer.vocab_size
    probs = torch.zeros(vocab_size)

    # Set high probability for 'A' token
    a_token_id = components.tokenizer.convert_tokens_to_ids("A")
    if a_token_id is not None:
        probs[a_token_id] = 0.8

    definitions = [
        Definition(synset_id="omw-en-1234-n", definition="definition 1"),
        Definition(synset_id="omw-en-5678-n", definition="definition 2"),
    ]

    choice_probs = get_choice_probabilities(components.tokenizer, probs, definitions)

    # Should return probabilities for A, B, and C (none of the above)
    assert len(choice_probs) == 3
    assert all(isinstance(p, float) for p in choice_probs)


def test_disambiguate_word_no_definitions():
    """Test disambiguate_word with no definitions"""
    result = disambiguate_word("test", "This is a *test*.", [])

    assert result.synset_id == NO_DEFINITIONS_FOUND
    assert result.definition == ""
    assert result.confidence == 0.0


def test_disambiguate_word_with_definitions():
    """Test disambiguate_word with real definitions"""
    definitions = [
        Definition(synset_id="omw-en-1234-n", definition="a financial institution"),
        Definition(synset_id="omw-en-5678-n", definition="the edge of a river"),
    ]

    result = disambiguate_word("bank", "I went to the *bank* to withdraw money.", definitions)

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
        DisambiguationInput(word="test", marked_sentence="This is a *test*.", definitions=[]),
        DisambiguationInput(word="example", marked_sentence="This is an *example*.", definitions=[]),
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
            word="bank",
            marked_sentence="I went to the *bank* to withdraw money.",
            definitions=definitions1
        ),
        DisambiguationInput(
            word="run",
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
            word="bank",
            marked_sentence="I went to the *bank*.",
            definitions=definitions
        ),
        DisambiguationInput(
            word="xyz",
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
