import pytest
from unittest.mock import AsyncMock, Mock
from document_analyser.models.file import File
from uuid import uuid4
from document_analyser.services.document_parser import DocumentService
from langchain_core.documents import Document
from sqlmodel import Session

@pytest.mark.asyncio
async def test_save_document():

    
    # --- Setup fake file ---
    sample_pdf_content = b"%PDF-1.4\n..."
    file = File(id=uuid4(), name="test.pdf", content=sample_pdf_content)

    # --- Mock dependencies ---
    mock_vector_store = AsyncMock()
    mock_text_splitter = Mock()
    mock_text_splitter.split_documents.return_value = [Document(page_content="Some text", metadata={})]
    mock_file_repository = Mock()
    mock_file_repository.create_file.return_value = file
    mock_session = Mock(spec=Session)


    # Create an instance of DocumentService
    service = DocumentService(
        vector_store=mock_vector_store,
        text_splitter=mock_text_splitter,
        file_repository=mock_file_repository,
    )

    # Patch the internal document conversion method
    service._DocumentService__convert_file_to_documents = Mock(
        return_value=[Document(page_content="Some text", metadata={})]
    )

    # --- Run the method ---
    saved_file = await service.save(mock_session, file)

    # --- Assertions ---
    mock_file_repository.create_file.assert_called_once_with(mock_session, file)
    mock_vector_store.add_documents.assert_awaited_once()
    assert saved_file == file

@pytest.mark.asyncio
async def test_search_documents():
    
    # --- Setup ---
    mock_vector_store = AsyncMock()

    # Fake document returned by the similarity search
    expected_docs = [Document(page_content="Relevant content", metadata={"file_id": str(uuid4())})]
    mock_vector_store.similarity_search.return_value = expected_docs

    # Create service with only the dependency you need
    service = DocumentService(
        vector_store=mock_vector_store,
        text_splitter=None,       # Not used here
        file_repository=None,     # Not used here
    )

    # --- Case 1: With file_ids ---
    file_id_1 = uuid4()
    file_id_2 = uuid4()
    query = "test query"
    results = await service.search(query, file_ids=[file_id_1, file_id_2])

    # Check correct arguments passed to the vector store
    mock_vector_store.similarity_search.assert_awaited_with(
        query,
        filter={"file_id": {"$in": [str(file_id_1), str(file_id_2)]}},
    )
    assert results == expected_docs

    # --- Case 2: Without file_ids ---
    mock_vector_store.similarity_search.reset_mock()
    results = await service.search(query)

    mock_vector_store.similarity_search.assert_awaited_with(query, filter=None)
    assert results == expected_docs