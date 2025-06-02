import pytest
from unittest.mock import Mock
from document_analyser.services.ai import AIService
from langchain_core.documents import Document

def test_retrieve_answer_returns_expected_output():
    # --- Setup test data ---
    question = "What is the company name?"
    docs = [Document(page_content="The company name is AI.")]

    # --- Mock the LLM and structured output ---
    mock_llm = Mock()
    mock_structured_llm = Mock()
    mock_llm.with_structured_output.return_value = mock_structured_llm

    # Mock the structured LLM to return a dict as expected by Output
    mock_structured_llm.invoke.return_value = {"answer": "AI"}

    # --- Instantiate AIService ---
    service = AIService(llm=mock_llm)

    # --- Act ---
    result = service.retrieve_answer(question, docs)

    # --- Assert ---
    assert result == "AI"
    mock_llm.with_structured_output.assert_called_once()
    mock_structured_llm.invoke.assert_called_once()

def test_retrieve_answer_returns_none_when_no_output():
    # --- Setup ---
    question = "What is the founder's name?"
    docs = [Document(page_content="No info about founders.")]
    
    mock_llm = Mock()
    mock_structured_llm = Mock()
    mock_llm.with_structured_output.return_value = mock_structured_llm
    mock_structured_llm.invoke.return_value = None  # Simulate no result

    service = AIService(llm=mock_llm)
    result = service.retrieve_answer(question, docs)

    assert result is None
