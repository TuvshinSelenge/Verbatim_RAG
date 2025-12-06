"""
KRLabs Verbatim RAG Implementation
Source: https://github.com/KRLabsOrg/verbatim-rag

This is a standalone version of the KRLabs verbatim answer generation system.
Key features:
1. Extracts EXACT verbatim spans from documents
2. Verifies spans exist in source text
3. Uses template-based composition with placeholders
4. Supports both LLM and ModernBERT-based extraction
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import openai
except ImportError:
    raise ImportError("OpenAI package required: pip install openai")


# =============================================================================
# MODELS
# =============================================================================

@dataclass
class Highlight:
    """A highlighted span in a document."""
    text: str
    start: int
    end: int


@dataclass
class Citation:
    """A citation reference."""
    text: str
    doc_index: int
    highlight_index: int
    number: Optional[int] = None
    type: Optional[str] = None  # "display" or "reference"


@dataclass
class DocumentWithHighlights:
    """A document with highlighted spans."""
    content: str
    highlights: List[Highlight] = field(default_factory=list)
    title: str = ""
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StructuredAnswer:
    """An answer with citations."""
    text: str
    citations: List[Citation] = field(default_factory=list)


@dataclass
class QueryResponse:
    """Complete response to a query."""
    question: str
    answer: str
    structured_answer: StructuredAnswer
    documents: List[DocumentWithHighlights] = field(default_factory=list)


# =============================================================================
# LLM CLIENT
# =============================================================================

class LLMClient:
    """
    Centralized LLM interaction handler.

    Provides unified interface for OpenAI API calls including:
    - Span extraction from documents
    - Structured extraction with template placeholders
    - Template generation
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        api_base: str = "https://api.openai.com/v1",
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"
        self.client = openai.OpenAI(base_url=api_base, api_key=self.api_key)
        self.async_client = openai.AsyncOpenAI(base_url=api_base, api_key=self.api_key)

    def complete(
        self,
        prompt: str,
        json_mode: bool = False,
        temperature: Optional[float] = None
    ) -> str:
        """Synchronous text completion."""
        messages = [{"role": "user", "content": prompt}]
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
        }

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    async def complete_async(
        self,
        prompt: str,
        json_mode: bool = False,
        temperature: Optional[float] = None
    ) -> str:
        """Asynchronous text completion."""
        messages = [{"role": "user", "content": prompt}]
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
        }

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = await self.async_client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    def extract_spans(
        self,
        question: str,
        documents: Dict[str, str]
    ) -> Dict[str, List[str]]:
        """
        Extract verbatim spans from multiple documents.

        Args:
            question: The user's question
            documents: Dict mapping doc IDs to document text

        Returns:
            Dict mapping doc IDs to lists of extracted spans
        """
        prompt = self._build_extraction_prompt(question, documents)
        try:
            response = self.complete(prompt, json_mode=True)
            return json.loads(response)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Span extraction failed: {e}")
            return {doc_id: [] for doc_id in documents.keys()}

    async def extract_spans_async(
        self,
        question: str,
        documents: Dict[str, str]
    ) -> Dict[str, List[str]]:
        """Async span extraction from documents."""
        prompt = self._build_extraction_prompt(question, documents)
        try:
            response = await self.complete_async(prompt, json_mode=True)
            return json.loads(response)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Async span extraction failed: {e}")
            return {doc_id: [] for doc_id in documents.keys()}

    def extract_relevant_spans(self, question: str, document_text: str) -> List[str]:
        """Extract spans from a single document."""
        result = self.extract_spans(question, {"doc": document_text})
        return result.get("doc", [])

    async def extract_relevant_spans_async(
        self,
        question: str,
        document_text: str
    ) -> List[str]:
        """Async extraction from a single document."""
        result = await self.extract_spans_async(question, {"doc": document_text})
        return result.get("doc", [])

    def extract_structured(
        self,
        question: str,
        template: str,
        placeholders: Dict[str, str],
        documents: List[str],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract spans organized by template placeholders.

        Args:
            question: The user's question
            template: Template with placeholders like [METHODOLOGY]
            placeholders: Dict mapping placeholder names to hints
            documents: List of document texts

        Returns:
            Dict mapping placeholder names to lists of {text, doc} objects
        """
        prompt = self._build_structured_extraction_prompt(
            question, template, placeholders, documents
        )
        try:
            response = self.complete(prompt, json_mode=True)
            return self._normalize_structured_response(
                json.loads(response), placeholders
            )
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Structured extraction failed: {e}")
            return {name: [] for name in placeholders.keys()}

    def generate_template(
        self,
        question: str,
        spans: List[str],
        citation_count: int,
        use_per_fact: bool = True,
    ) -> str:
        """
        Generate a contextual template for the given question and spans.

        Args:
            question: The user's question
            spans: List of spans that will fill the template
            citation_count: Number of citation-only spans
            use_per_fact: Whether to use per-fact placeholders

        Returns:
            Generated template string with placeholders
        """
        if use_per_fact and len(spans) <= 8:
            prompt = self._build_per_fact_template_prompt(
                question, spans, citation_count
            )
        else:
            prompt = self._build_aggregate_template_prompt(
                question, spans, citation_count
            )

        try:
            return self.complete(prompt, temperature=self.temperature)
        except Exception as e:
            print(f"Template generation failed: {e}")
            return self._fallback_template(citation_count > 0)

    def _build_extraction_prompt(self, question: str, documents: Dict[str, str]) -> str:
        """Build the prompt for batch span extraction."""
        return f"""Extract EXACT verbatim text spans from multiple documents that answer the question.

# Rules
1. Extract **only** text that explicitly addresses the question
2. Never paraphrase, modify, or add to the original text
3. Preserve original wording, capitalization, and punctuation
4. Order spans within each document by relevance - MOST RELEVANT FIRST
5. Include complete sentences or paragraphs for context

# Output Format
Return a JSON object mapping document IDs to span arrays ordered by relevance:
{{
    "doc_0": ["most relevant span", "next most relevant span"],
    "doc_1": ["most relevant from doc 1"],
    "doc_2": []
}}

If no relevant information in a document, use empty array.

# Your Task
Question: {question}

Documents:
{json.dumps(documents, indent=2)}

Extract verbatim spans from each document:"""

    def _build_structured_extraction_prompt(
        self,
        question: str,
        template: str,
        placeholders: Dict[str, str],
        documents: List[str],
    ) -> str:
        """Build prompt for structured extraction with document attribution."""
        placeholder_spec = "\n".join(
            f"- {name}: {hint}" for name, hint in placeholders.items()
        )
        docs_text = "\n\n---\n\n".join(
            f"[Document {i}]\n{doc}" for i, doc in enumerate(documents)
        )

        return f"""Extract verbatim spans from the documents for each placeholder in the template.

Question: {question}

Template to fill:
{template}

Placeholders to extract for:
{placeholder_spec}

Documents:
{docs_text}

Instructions:
1. For each placeholder, find EXACT verbatim quotes from the documents
2. Copy text exactly - no paraphrasing or modification
3. For each span, include which document it came from (0-indexed)
4. Return a JSON object mapping placeholder names to arrays of objects with "text" and "doc" fields
5. If no relevant information for a placeholder, use an empty array

Return ONLY valid JSON like:
{{
    "METHODOLOGY": [{{"text": "exact quote about methods...", "doc": 0}}],
    "RESULTS": [{{"text": "exact quote about results...", "doc": 1}}]
}}"""

    def _build_per_fact_template_prompt(
        self, question: str, spans: List[str], citation_count: int
    ) -> str:
        """Build prompt for per-fact template generation."""
        span_lines = []
        for i, span in enumerate(spans, start=1):
            clean = span.replace("\n", " ").strip()[:100]
            span_lines.append(f"{i}. {clean}...")
        spans_block = "\n".join(span_lines)

        return f"""Generate a response template for this Q&A scenario:

Question: {question}

Content that will be inserted into the template:
- Total verbatim facts to show (display facts): {len(spans)}
- Full list of verbatim facts:
{spans_block}
- Additional citation-only facts (only numbers, no text shown): {citation_count}

Template strategy rules:
- Use per-fact placeholders [FACT_1]..[FACT_{len(spans)}] each exactly once.
- If citation-only facts exist, you MAY place [CITATION_REFS] exactly once where their numbers should appear, otherwise omit it.

Instructions:
- Intro: 1 concise sentence tying question to facts.
- Then present each fact in a structured way (bulleted list or numbered list).
- DO NOT invent content beyond connective phrases; never summarize or paraphrase inside placeholders.
- No duplicate placeholders; no placeholder inside a heading alone.

Template requirements:
- Use only placeholders plus minimal connective prose (no actual span text).
- {"Include [CITATION_REFS] once" if citation_count > 0 else "Do NOT include [CITATION_REFS]"}.

Return ONLY the template text (no explanation)."""

    def _build_aggregate_template_prompt(
        self, question: str, spans: List[str], citation_count: int
    ) -> str:
        """Build prompt for aggregate template generation."""
        span_preview = " | ".join(span[:50] + "..." for span in spans[:3])

        return f"""Generate a response template for this Q&A scenario:

Question: {question}

Content that will be inserted into the template:
- Total verbatim facts to show (display facts): {len(spans)}
- Preview of content: {span_preview}
- Additional citation-only facts (only numbers, no text shown): {citation_count}

Template strategy rules:
- Use [DISPLAY_SPANS] exactly once for the aggregate of all verbatim spans.
- If citation-only facts exist, you MAY place [CITATION_REFS] exactly once.

Instructions:
- Intro: 1 concise sentence tying question to spans.
- Provide a section header then include the aggregate placeholder.
- Do NOT invent or paraphrase span content; placeholders stand in for verbatim content only.

Template requirements:
- Must contain [DISPLAY_SPANS].
- {"Include [CITATION_REFS] once" if citation_count > 0 else "Do NOT include [CITATION_REFS]"}.

Return ONLY the template text (no explanation)."""

    def _fallback_template(self, has_citations: bool = False) -> str:
        """Return a simple fallback template when generation fails."""
        template = """## Response

Based on the available documents:

[DISPLAY_SPANS]"""

        if has_citations:
            template += "\n\n**Additional References:** [CITATION_REFS]"

        return template

    def _normalize_structured_response(
        self, response: Dict, placeholders: Dict[str, str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Normalize LLM response to ensure consistent format."""
        result = {}
        for name in placeholders.keys():
            items = response.get(name, [])
            normalized = []
            for item in items:
                if isinstance(item, str):
                    normalized.append({"text": item, "doc": 0})
                elif isinstance(item, dict) and "text" in item:
                    normalized.append({"text": item["text"], "doc": item.get("doc", 0)})
            result[name] = normalized
        return result


# =============================================================================
# SPAN EXTRACTORS
# =============================================================================

class SpanExtractor(ABC):
    """Abstract base class for span extractors."""

    @abstractmethod
    def extract_spans(
        self, question: str, search_results: List[Any]
    ) -> Dict[str, List[str]]:
        """
        Extract relevant spans from search results.

        Args:
            question: The query or question
            search_results: List of search results to extract from

        Returns:
            Dictionary mapping result text to list of relevant spans
        """
        raise NotImplementedError

    async def extract_spans_async(
        self, question: str, search_results: List[Any]
    ) -> Dict[str, List[str]]:
        """Default async implementation that delegates to sync version."""
        import asyncio
        return await asyncio.to_thread(self.extract_spans, question, search_results)


class LLMSpanExtractor(SpanExtractor):
    """Extract spans using an LLM with batch processing support."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        model: str = "gpt-4o-mini",
        extraction_mode: str = "auto",  # "batch", "individual", "auto"
        max_display_spans: int = 5,
        batch_size: int = 5,
    ):
        self.llm_client = llm_client or LLMClient(model)
        self.extraction_mode = extraction_mode
        self.max_display_spans = max_display_spans
        self.batch_size = batch_size

    def extract_spans(
        self, question: str, search_results: List[Any]
    ) -> Dict[str, List[str]]:
        """Extract spans using LLM with mode selection."""
        if not search_results:
            return {}

        should_batch = self.extraction_mode == "batch" or (
            self.extraction_mode == "auto" and len(search_results) <= self.batch_size
        )

        if should_batch:
            return self._extract_spans_batch(question, search_results)
        else:
            return self._extract_spans_individual(question, search_results)

    async def extract_spans_async(
        self, question: str, search_results: List[Any]
    ) -> Dict[str, List[str]]:
        """Async version of span extraction."""
        if not search_results:
            return {}

        should_batch = self.extraction_mode == "batch" or (
            self.extraction_mode == "auto" and len(search_results) <= self.batch_size
        )

        if should_batch:
            return await self._extract_spans_batch_async(question, search_results)
        else:
            return await self._extract_spans_individual_async(question, search_results)

    def _extract_spans_batch(
        self, question: str, search_results: List[Any]
    ) -> Dict[str, List[str]]:
        """Extract spans from multiple documents using batch processing."""
        print("Extracting spans (batch mode)...")

        top_results = search_results[: self.batch_size]
        documents_text = {}
        for i, result in enumerate(top_results):
            documents_text[f"doc_{i}"] = getattr(result, "text", str(result))

        try:
            extracted_data = self.llm_client.extract_spans(question, documents_text)
            verified_spans = {}

            for i, result in enumerate(top_results):
                doc_key = f"doc_{i}"
                result_text = getattr(result, "text", str(result))
                if doc_key in extracted_data:
                    verified = self._verify_spans(extracted_data[doc_key], result_text)
                    verified_spans[result_text] = verified
                else:
                    verified_spans[result_text] = []

            for i in range(self.batch_size, len(search_results)):
                verified_spans[getattr(search_results[i], "text", str(search_results[i]))] = []

            return verified_spans

        except Exception as e:
            print(f"Batch extraction failed, falling back to individual: {e}")
            return self._extract_spans_individual(question, search_results)

    async def _extract_spans_batch_async(
        self, question: str, search_results: List[Any]
    ) -> Dict[str, List[str]]:
        """Async batch extraction."""
        print("Extracting spans (async batch mode)...")

        top_results = search_results[: self.batch_size]
        documents_text = {}
        for i, result in enumerate(top_results):
            documents_text[f"doc_{i}"] = getattr(result, "text", str(result))

        try:
            extracted_data = await self.llm_client.extract_spans_async(
                question, documents_text
            )
            verified_spans = {}

            for i, result in enumerate(top_results):
                doc_key = f"doc_{i}"
                result_text = getattr(result, "text", str(result))
                if doc_key in extracted_data:
                    verified = self._verify_spans(extracted_data[doc_key], result_text)
                    verified_spans[result_text] = verified
                else:
                    verified_spans[result_text] = []

            for i in range(self.batch_size, len(search_results)):
                verified_spans[getattr(search_results[i], "text", str(search_results[i]))] = []

            return verified_spans

        except Exception as e:
            print(f"Async batch extraction failed: {e}")
            return await self._extract_spans_individual_async(question, search_results)

    def _extract_spans_individual(
        self, question: str, search_results: List[Any]
    ) -> Dict[str, List[str]]:
        """Extract spans from documents individually."""
        print("Extracting spans (individual mode)...")
        all_spans = {}

        for result in search_results:
            result_text = getattr(result, "text", str(result))
            try:
                extracted_spans = self.llm_client.extract_relevant_spans(
                    question, result_text
                )
                verified = self._verify_spans(extracted_spans, result_text)
                all_spans[result_text] = verified
            except Exception as e:
                print(f"Individual extraction failed for document: {e}")
                all_spans[result_text] = []

        return all_spans

    async def _extract_spans_individual_async(
        self, question: str, search_results: List[Any]
    ) -> Dict[str, List[str]]:
        """Async individual extraction."""
        print("Extracting spans (async individual mode)...")
        all_spans = {}

        for result in search_results:
            result_text = getattr(result, "text", str(result))
            try:
                extracted_spans = await self.llm_client.extract_relevant_spans_async(
                    question, result_text
                )
                verified = self._verify_spans(extracted_spans, result_text)
                all_spans[result_text] = verified
            except Exception as e:
                print(f"Async individual extraction failed: {e}")
                all_spans[result_text] = []

        return all_spans

    def _verify_spans(self, spans: List[str], document_text: str) -> List[str]:
        """
        Verify that extracted spans actually exist in the document text.
        This is the KEY to preventing hallucination!
        """
        verified = []
        for span in spans:
            if span.strip() and span.strip() in document_text:
                verified.append(span.strip())
            else:
                print(f"Warning: Span not found verbatim in document: '{span[:100]}...'")
        return verified


class ModelSpanExtractor(SpanExtractor):
    """Extract spans using a fine-tuned ModernBERT model."""

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        threshold: float = 0.5,
    ):
        import torch
        from transformers import AutoTokenizer

        self.model_path = model_path
        self.threshold = threshold
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading model from {model_path}...")

        # You need to have the QAModel class from KRLabs
        # For now, this is a placeholder - you'd need their model code
        try:
            from verbatim_core.extractor_models.model import QAModel
            from verbatim_core.extractor_models.dataset import (
                QADataset, Sentence as DatasetSentence,
                Document as DatasetDocument, QASample
            )

            self.QADataset = QADataset
            self.DatasetSentence = DatasetSentence
            self.DatasetDocument = DatasetDocument
            self.QASample = QASample

            self.model = QAModel.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        except ImportError:
            raise ImportError(
                "ModelSpanExtractor requires verbatim_core package. "
                "Use LLMSpanExtractor instead or install verbatim-rag."
            )

    def _split_into_sentences(self, text: str) -> List[str]:
        """Simple sentence splitting."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def extract_spans(
        self, question: str, search_results: List[Any]
    ) -> Dict[str, List[str]]:
        """Extract spans using the trained model."""
        relevant_spans = {}

        for result in search_results:
            raw_text = getattr(result, "text", str(result))
            raw_sentences = self._split_into_sentences(raw_text)

            if not raw_sentences:
                relevant_spans[raw_text] = []
                continue

            dataset_sentences = [
                self.DatasetSentence(text=sent, relevant=False, sentence_id=f"s{i}")
                for i, sent in enumerate(raw_sentences)
            ]
            dataset_doc = self.DatasetDocument(sentences=dataset_sentences)

            qa_sample = self.QASample(
                question=question,
                documents=[dataset_doc],
                split="test",
                dataset_name="inference",
                task_type="qa",
            )

            dataset = self.QADataset([qa_sample], self.tokenizer, max_length=512)
            if len(dataset) == 0:
                relevant_spans[raw_text] = []
                continue

            encoding = dataset[0]
            input_ids = encoding["input_ids"].unsqueeze(0).to(self.device)
            attention_mask = encoding["attention_mask"].unsqueeze(0).to(self.device)

            with self._torch.no_grad():
                predictions = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    sentence_boundaries=[encoding["sentence_boundaries"]],
                )

            spans = []
            if len(predictions) > 0 and len(predictions[0]) > 0:
                sentence_preds = self._torch.nn.functional.softmax(predictions[0], dim=1)
                for i, pred in enumerate(sentence_preds):
                    if i < len(raw_sentences) and pred[1] > self.threshold:
                        spans.append(raw_sentences[i])

            relevant_spans[raw_text] = spans

        return relevant_spans


# =============================================================================
# RESPONSE BUILDER
# =============================================================================

class ResponseBuilder:
    """Builds structured query responses with highlights and citations."""

    def build_response(
        self,
        question: str,
        answer: str,
        search_results: List[Any],
        relevant_spans: Dict[str, List[str]],
        display_span_count: Optional[int] = None,
    ) -> QueryResponse:
        """Build a complete QueryResponse from components."""
        documents_with_highlights = []
        all_citations = []
        current_citation_number = 1

        for result_index, result in enumerate(search_results):
            result_content = getattr(result, "text", str(result))
            highlights = []
            spans_for_doc = relevant_spans.get(result_content, [])

            if spans_for_doc:
                highlights = self._create_highlights(result_content, spans_for_doc)

                for highlight_index, highlight in enumerate(highlights):
                    is_display = (
                        display_span_count is None
                        or current_citation_number <= display_span_count
                    )

                    all_citations.append(
                        Citation(
                            text=highlight.text,
                            doc_index=result_index,
                            highlight_index=highlight_index,
                            number=current_citation_number,
                            type="display" if is_display else "reference",
                        )
                    )
                    current_citation_number += 1

            metadata = getattr(result, "metadata", {})
            documents_with_highlights.append(
                DocumentWithHighlights(
                    content=result_content,
                    highlights=highlights,
                    title=getattr(result, "title", "") or metadata.get("title", ""),
                    source=getattr(result, "source", "") or metadata.get("source", ""),
                    metadata=metadata,
                )
            )

        structured_answer = StructuredAnswer(text=answer, citations=all_citations)

        return QueryResponse(
            question=question,
            answer=answer,
            structured_answer=structured_answer,
            documents=documents_with_highlights,
        )

    def _create_highlights(self, doc_content: str, spans: List[str]) -> List[Highlight]:
        """Create highlight objects for spans in document content."""
        highlights: List[Highlight] = []
        highlighted_regions: Set[Tuple[int, int]] = set()

        for span in spans:
            start = 0
            while True:
                start = doc_content.find(span, start)
                if start == -1:
                    break

                end = start + len(span)

                if not self._has_overlap(start, end, highlighted_regions):
                    highlights.append(Highlight(text=span, start=start, end=end))
                    highlighted_regions.add((start, end))

                start = end

        return highlights

    def _has_overlap(self, start: int, end: int, regions: Set[Tuple[int, int]]) -> bool:
        """Check if a text region overlaps with existing highlighted regions."""
        for region_start, region_end in regions:
            if start < region_end and end > region_start:
                return True
        return False

    def clean_answer(self, answer: str) -> str:
        """Clean up generated answer text."""
        if not answer:
            return ""

        if answer.startswith('"') and answer.endswith('"'):
            answer = answer[1:-1]
        elif answer.startswith("'") and answer.endswith("'"):
            answer = answer[1:-1]

        answer = answer.replace("\\n", "\n")
        answer = re.sub(r" {2,}", " ", answer)
        answer = re.sub(r"\n{3,}", "\n\n", answer)

        return answer.strip()


# =============================================================================
# TEMPLATE MANAGER (Simplified)
# =============================================================================

class TemplateManager:
    """Manages template generation and filling."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        default_mode: str = "contextual",  # "static", "contextual", "random"
    ):
        self.llm_client = llm_client or LLMClient()
        self.default_mode = default_mode
        self.current_mode = default_mode

    def get_template(
        self,
        question: str,
        display_spans: List[str],
        citation_count: int
    ) -> str:
        """Generate or select a template for the response."""
        if self.current_mode == "static":
            return self._static_template(citation_count > 0)
        else:
            return self.llm_client.generate_template(
                question, display_spans, citation_count
            )

    def fill_template(
        self,
        template: str,
        display_spans: List[Dict],
        citation_spans: List[Dict],
    ) -> str:
        """Fill template with display spans and citation references."""
        result = template

        # Handle per-fact placeholders [FACT_1], [FACT_2], etc.
        for i, span_info in enumerate(display_spans, start=1):
            placeholder = f"[FACT_{i}]"
            span_text = span_info.get("text", "") if isinstance(span_info, dict) else str(span_info)
            result = result.replace(placeholder, f'"{span_text}"')

        # Handle aggregate placeholders
        if "[DISPLAY_SPANS]" in result:
            spans_text = "\n\n".join(
                f'• "{s.get("text", s) if isinstance(s, dict) else s}"'
                for s in display_spans
            )
            result = result.replace("[DISPLAY_SPANS]", spans_text)

        # Handle citation references
        if "[CITATION_REFS]" in result and citation_spans:
            start_num = len(display_spans) + 1
            refs = ", ".join(
                f"[{start_num + i}]" for i in range(len(citation_spans))
            )
            result = result.replace("[CITATION_REFS]", refs)
        else:
            result = result.replace("[CITATION_REFS]", "")

        return result

    def process(
        self,
        question: str,
        display_spans: List[Dict],
        citation_spans: List[Dict],
    ) -> str:
        """Full process: generate template and fill it."""
        span_texts = [
            s.get("text", s) if isinstance(s, dict) else str(s)
            for s in display_spans
        ]
        template = self.get_template(question, span_texts, len(citation_spans))
        return self.fill_template(template, display_spans, citation_spans)

    def _static_template(self, has_citations: bool = False) -> str:
        """Return a simple static template."""
        template = """Based on the available documents:

[DISPLAY_SPANS]"""

        if has_citations:
            template += "\n\n**Additional References:** [CITATION_REFS]"

        return template


# =============================================================================
# MAIN VERBATIM RAG CLASS
# =============================================================================

class KRLabsVerbatimRAG:
    """
    A RAG system that prevents hallucination by ensuring all generated content
    is explicitly derived from source documents.

    This is the main orchestrator that coordinates:
    1. Document retrieval (via index)
    2. Span extraction (via extractor)
    3. Template generation and filling
    4. Response building
    """

    def __init__(
        self,
        index: Any,  # Your VerbatimIndex or any index with .query() method
        model: str = "gpt-4o-mini",
        k: int = 5,
        extractor: Optional[SpanExtractor] = None,
        max_display_spans: int = 5,
        template_mode: str = "contextual",
        extraction_mode: str = "auto",
        llm_client: Optional[LLMClient] = None,
    ):
        self.index = index
        self.k = k
        self.max_display_spans = max_display_spans

        # Centralized LLM client
        self.llm_client = llm_client or LLMClient(model)

        # Initialize extractor
        self.extractor = extractor or LLMSpanExtractor(
            llm_client=self.llm_client,
            extraction_mode=extraction_mode,
            max_display_spans=max_display_spans,
        )

        # Template manager
        self.template_manager = TemplateManager(
            llm_client=self.llm_client,
            default_mode=template_mode,
        )

        # Response builder
        self.response_builder = ResponseBuilder()

    def query(
        self,
        question: str,
        k: Optional[int] = None,
        filter: Optional[str] = None,
    ) -> QueryResponse:
        """
        Process a query through the Verbatim RAG system.

        Args:
            question: The user's question
            k: Number of documents to retrieve
            filter: Optional filter for document search

        Returns:
            QueryResponse with answer and citations
        """
        # Step 1: Retrieve documents
        k = k or self.k
        search_results = self.index.query(text=question, k=k, filter=filter)

        # Step 2: Extract verbatim spans
        print("Extracting relevant spans...")
        all_relevant_spans = self.extractor.extract_spans(question, search_results)

        # Step 3: Rank and split spans
        print("Processing spans...")
        display_spans, citation_spans = self._rank_and_split_spans(all_relevant_spans)

        # Step 4: Generate response using template
        print("Generating response...")
        answer = self.template_manager.process(question, display_spans, citation_spans)

        # Step 5: Clean and build response
        answer = self.response_builder.clean_answer(answer)

        return self.response_builder.build_response(
            question=question,
            answer=answer,
            search_results=search_results,
            relevant_spans=all_relevant_spans,
            display_span_count=len(display_spans),
        )

    def _rank_and_split_spans(
        self, relevant_spans: Dict[str, List[str]]
    ) -> Tuple[List[Dict], List[Dict]]:
        """Split spans into display vs citation-only."""
        all_spans = []
        for doc_text, spans in relevant_spans.items():
            for span in spans:
                all_spans.append({"text": span, "doc_text": doc_text})

        display_spans = all_spans[: self.max_display_spans]
        citation_spans = all_spans[self.max_display_spans :]

        return display_spans, citation_spans


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def extract_verbatim_spans(
    question: str,
    documents: List[str],
    model: str = "gpt-4o-mini",
    verify: bool = True,
) -> Dict[str, List[str]]:
    """
    Extract verbatim spans from documents using LLM.

    This is the core function you can use standalone.

    Args:
        question: The question to answer
        documents: List of document texts
        model: OpenAI model to use
        verify: Whether to verify spans exist in source

    Returns:
        Dict mapping document text to list of extracted spans
    """
    llm_client = LLMClient(model=model)

    # Create document dict
    doc_dict = {f"doc_{i}": doc for i, doc in enumerate(documents)}

    # Extract spans
    extracted = llm_client.extract_spans(question, doc_dict)

    # Map back and verify
    result = {}
    for i, doc in enumerate(documents):
        doc_key = f"doc_{i}"
        spans = extracted.get(doc_key, [])

        if verify:
            # Only keep spans that exist verbatim in the document
            verified = [s for s in spans if s.strip() in doc]
            result[doc] = verified
        else:
            result[doc] = spans

    return result


def compose_verbatim_answer(
    question: str,
    spans: List[str],
    model: str = "gpt-4o-mini",
    use_template: bool = True,
) -> str:
    """
    Compose an answer using only verbatim spans.

    Args:
        question: The original question
        spans: List of verbatim spans to use
        model: OpenAI model to use
        use_template: Whether to use template-based generation

    Returns:
        Composed answer using only the provided spans
    """
    llm_client = LLMClient(model=model)
    template_manager = TemplateManager(llm_client=llm_client)

    display_spans = [{"text": s} for s in spans]

    if use_template:
        return template_manager.process(question, display_spans, [])
    else:
        # Simple concatenation
        return "\n\n".join(f'• "{s}"' for s in spans)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example: Standalone span extraction and answer composition

    documents = [
        "Raiffeisen Bank International AG (RBI) is headquartered in Vienna, Austria. "
        "The bank was founded in 1927 and serves customers across Central and Eastern Europe.",

        "RBI maintains comprehensive risk management policies. The bank's risk framework "
        "includes credit risk, market risk, and operational risk management procedures.",
    ]

    question = "Where is Raiffeisen Bank headquartered?"

    # Extract spans
    print("Extracting spans...")
    spans_by_doc = extract_verbatim_spans(question, documents)

    print("\nExtracted spans:")
    for doc, spans in spans_by_doc.items():
        for span in spans:
            print(f'  "{span}"')

    # Compose answer
    all_spans = [s for spans in spans_by_doc.values() for s in spans]
    if all_spans:
        print("\nComposing answer...")
        answer = compose_verbatim_answer(question, all_spans)
        print(f"\nAnswer:\n{answer}")
    else:
        print("\nNo relevant spans found.")
