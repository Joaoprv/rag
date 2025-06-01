from dataclasses import dataclass
from langchain.vectorstores import VectorStore
from langchain.text_splitter import TextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
import tempfile
from sqlmodel import Session
from document_analyser.models.file import File
from document_analyser.repository.file import FileRepository

@dataclass
class DocumentService:


    vector_store: VectorStore
    text_splitter: TextSplitter
    file_repository: FileRepository

    async def save(self, session: Session, file: File) -> File:
        """"
        Save a file to the vector store.

        Args:
            file (File): The file to save.

        Returns"""

        file = self.file_repository.create_file(session, file)

        documents = self.__convert_file_to_documents(file)
        splits = self.text_splitter.split_documents(documents)
        self.__add_metadata_to_documents(splits, file)
        await self.vector_store.add_documents(splits)

        return file
    
    def __add_metadata_to_documents(self, documents: list[Document], file: File) -> None:
        """
        Add metadata to documents.

        Args:
            documents (list[Document]): The documents to add metadata to.
            file (File): The file containing the metadata.
        """
        for document in documents:
            document.metadata["file_id"] = str(file.id)
            document.metadata["file_name"] = file.name
        

    def __convert_file_to_documents(self, file: File) -> list[Document]:
        """
        Convert a file to a list of documents.

        Args:
            file (File): The file to convert.

        Returns:
            list[Document]: A list of documents.
        """
        with tempfile.NamedTemporaryFile(suffix=".pdf",delete=True) as temp_file:
            temp_file.write(file.content)
            temp_file.flush()

            loader = PyPDFLoader(temp_file.name)
            return loader.load()