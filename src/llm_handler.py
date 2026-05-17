"""
LLM Handler - Manages OpenAI chat completion for RAG-based Q&A.
Handles the generation of responses based on retrieved context.
"""

import os
from typing import List, Tuple, Optional
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


# Concise adaptive prompt for document-based Q&A
ANALYSIS_PROMPT = """You are a concise document analysis assistant.

RULES:
1. For greetings or general questions (hello, how are you, what can you do): respond briefly and conversationally. Do NOT use document context.
2. For document questions: answer ONLY what is asked. Be direct and concise. No extra explanations or summaries.
3. Never show what is "not mentioned" or "missing" from documents unless the user specifically asks about it.
4. Cite document source only when directly relevant to the answer.
5. If the answer is not in the documents, simply say "The documents don't contain this information."

Context from documents:
{context}

Question: {question}

Answer:"""


class LLMQuestionAnswerer:
    """
    Handles question answering using OpenAI's models with RAG context.
    """

    def __init__(
        self,
        model_name: str = "gpt-4",
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ):
        """
        Initialize the LLM question answerer.

        Args:
            model_name: OpenAI model name (gpt-4, gpt-3.5-turbo, etc.)
            temperature: Controls randomness (0 = deterministic, 1 = creative)
            max_tokens: Maximum tokens in response
        """
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.prompt_template = ChatPromptTemplate.from_template(
            ANALYSIS_PROMPT
        )

    def format_context(self, search_results: List[Tuple[Document, float]]) -> str:
        """
        Format retrieved documents into a context string for the LLM.

        Args:
            search_results: List of (Document, score) tuples from similarity search

        Returns:
            Formatted context string
        """
        context_parts = []
        for i, (doc, score) in enumerate(search_results, 1):
            source = doc.metadata.get("source", doc.metadata.get("file_name", "Unknown"))
            page = doc.metadata.get("page", "N/A")
            context_parts.append(
                f"[Document {i} - Source: {source}, Page: {page}]\n{doc.page_content}\n"
            )

        return "\n\n".join(context_parts)

    def generate_response(
        self,
        question: str,
        context_docs: List[Tuple[Document, float]],
    ) -> str:
        """
        Generate a response based on the question and retrieved context.

        Args:
            question: The user's question
            context_docs: Retrieved relevant documents

        Returns:
            Generated answer string
        """
        # If no context found
        if not context_docs:
            return (
                "I couldn't find any relevant information in the uploaded documents "
                "to answer your question. Please make sure you have uploaded and processed "
                "the relevant documents first, or try rephrasing your question."
            )

        # Format context from retrieved documents
        context = self.format_context(context_docs)

        # Generate response using the LLM
        messages = self.prompt_template.format_messages(
            context=context,
            question=question,
        )

        response = self.llm.invoke(messages)
        return response.content

    def generate_streaming_response(
        self,
        question: str,
        context_docs: List[Tuple[Document, float]],
    ):
        """
        Generate a streaming response based on the question and retrieved context.

        Args:
            question: The user's question
            context_docs: Retrieved relevant documents

        Yields:
            Chunks of the generated response
        """
        if not context_docs:
            yield (
                "I couldn't find any relevant information in the uploaded documents "
                "to answer your question. Please make sure you have uploaded and processed "
                "the relevant documents first, or try rephrasing your question."
            )
            return

        # Format context
        context = self.format_context(context_docs)

        # Generate streaming response
        messages = self.prompt_template.format_messages(
            context=context,
            question=question,
        )

        for chunk in self.llm.stream(messages):
            yield chunk.content