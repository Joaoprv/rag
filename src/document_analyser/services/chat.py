from dataclasses import dataclass
from document_analyser.schemas import ChatCreate
from document_analyser.repository import ChatRepository, MessageRepository
from document_analyser.models import Chat, Message, SenderType
from sqlmodel import Session, select
from document_analyser.schemas import MessageCreate
from document_analyser.services.ai import AIService
from document_analyser.services.document_parser import DocumentService
from uuid import UUID
from typing import Sequence


@dataclass
class ChatService:
    chat_repository: ChatRepository
    message_repository: MessageRepository
    ai_svc: AIService
    document_svc: DocumentService

    def create_chat(self, session: Session, chat_create: ChatCreate):
        chat = Chat(name="New Chat", files=[])
        return self.chat_repository.create(session, chat, chat_create.file_ids)

    def find_all_chats(self, session: Session):
        return self.chat_repository.find_all(session)

    def get_chat(self, session: Session, chat_id: UUID):
        chat = session.exec(select(Chat).where(Chat.id == chat_id)).one_or_none()
        if not chat:
            raise ValueError(f"Chat with ID {chat_id} does not exist.")
        return chat

    async def send_message(
        self, session: Session, chat_id: UUID, message_create: MessageCreate
    ):
        human_message = Message(
            content=message_create.content,
            chat_id=chat_id,
            sender_type=SenderType.HUMAN,
        )

        chat = self.get_chat(session, chat_id)
        docs = await self.document_svc.search(
            human_message.content, [file.id for file in chat.files]
        )

        answer = self.ai_svc.retrieve_answer(
            human_message.content,
            docs,
        )
        if not answer:
            answer = "N/A"

        ai_message = Message(content=answer, chat_id=chat_id, sender_type=SenderType.AI)

        self.message_repository.save_messages(session, human_message, ai_message)

        return ai_message

    def find_messages(self, session: Session, chat_id: UUID) -> Sequence[Message]:
        return self.message_repository.find_by_chat_id(session, chat_id)