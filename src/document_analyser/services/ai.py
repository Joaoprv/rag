from langchain_core.language_models import BaseChatModel
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "Always return a JSON object with this structure: {\"answer\": \"your answer\"} "
            "If you do not know the value of an attribute, use null.",
        ),
        ("system", "{data}"),
        ("human", "{text}"),
    ]
)

class Output(BaseModel):
    answer: str | None = Field(
        default=None,
        description="Answer on the question",
    )


class AIService:

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def retrieve_answer(self, question: str, docs: list[Document]) -> str | None:
        """
        Retrieve an answer to a question based on the provided documents.

        Args:
            question (str): The question to answer.
            docs (list[Document]): The documents to search for the answer.

        Returns:
            str | None: The answer to the question, or None if no answer could be found.
        """
        data = "\n\n".join(doc.page_content for doc in docs)
        prompt = prompt_template.invoke({"text": question, "data": data})
        llm_result = self.llm.invoke(prompt)

        if hasattr(llm_result, "content"):
            content = llm_result.content
        else:
            content = str(llm_result)

        try:
            return Output.model_validate_json(content).answer
        except Exception as e:
            print(f"[AIService] Failed to parse response: {content}\nError: {e}")
            return None