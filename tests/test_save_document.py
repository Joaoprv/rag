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
