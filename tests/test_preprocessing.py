from src.preprocessing import preprocess, tokenize


def test_preprocess_replaces_mentions():
    example = {"text": "@johndoe check this out"}
    result = preprocess(example)
    assert result["text"] == "@user check this out"


def test_preprocess_replaces_urls():
    example = {"text": "see http://example.com now"}
    result = preprocess(example)
    assert result["text"] == "see http now"


def test_preprocess_leaves_plain_text_untouched():
    example = {"text": "just a normal sentence"}
    result = preprocess(example)
    assert result["text"] == "just a normal sentence"


def test_tokenize_returns_expected_keys():
    batch = {"text": ["hello world", "another sentence"]}
    encoded = tokenize(batch)
    assert "input_ids" in encoded
    assert "attention_mask" in encoded
    assert len(encoded["input_ids"]) == 2
