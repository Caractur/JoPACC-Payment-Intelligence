"""Activity 3 - retrieval-augmented Q&A over supplied JoPACC reports.

The implementation follows ``Activity_3.ipynb``: page-aware PDF extraction,
an OCR fallback, overlapping chunks, sentence-transformer embeddings, cosine
similarity, global retrieval, a relevance threshold, and a grounded MiniMax
response through OpenRouter.
"""

from __future__ import annotations

import os
import re
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_REPORTS_DIR = BASE_DIR / "Reports"
INDEX_CACHE_PATH = BASE_DIR / "rag_index_cache.pkl"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "minimax/minimax-m3:free")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

MONTH_NUMBERS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
MINIMUM_RELEVANCE_SCORE = 0.45


@dataclass
class ReportIndex:
    chunks: list[dict]
    vectors: Any
    encoder: Any
    ocr_reports: list[str] = field(default_factory=list)
    report_count: int = 0
    page_count: int = 0


def clean_text(text: str) -> str:
    """Remove control characters and normalise extraction whitespace."""
    text = text.replace("\x00", " ").replace("\x7f", " ")
    return re.sub(r"\s+", " ", text).strip()


def get_report_month(filename: str) -> str:
    """Derive YYYY-MM report metadata from the supplied filenames."""
    lower_name = filename.lower()
    month_name = next((name for name in MONTH_NUMBERS if name in lower_name), None)
    year_match = re.search(r"(20\d{2})", lower_name)
    if month_name and year_match:
        return f"{year_match.group(1)}-{MONTH_NUMBERS[month_name]:02d}"
    return "Unknown"


def _ocr_page(pdf_document: Any, page_index: int, ocr_engine: Any) -> str:
    image = pdf_document[page_index].render(scale=1.5).to_numpy()
    result, _ = ocr_engine(image)
    return clean_text(" ".join(line[1] for line in (result or [])))


def _ocr_document(path: Path) -> list[str]:
    """OCR a report when optional notebook dependencies are installed."""
    try:
        import pypdfium2 as pdfium
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        return []

    document = pdfium.PdfDocument(str(path))
    engine = RapidOCR()
    try:
        return [_ocr_page(document, index, engine) for index in range(len(document))]
    finally:
        document.close()


def extract_report_pages(
    reports_dir: str | Path = DEFAULT_REPORTS_DIR,
    minimum_report_characters: int = 1_000,
) -> tuple[list[dict], list[str]]:
    """Extract report text while preserving report, month, and page metadata."""
    reports_dir = Path(reports_dir)
    if not reports_dir.exists():
        raise FileNotFoundError(f"JoPACC reports folder was not found: {reports_dir}")

    pages: list[dict] = []
    ocr_reports: list[str] = []
    pdf_paths = sorted(reports_dir.glob("*.pdf"))
    if not pdf_paths:
        raise ValueError(f"No PDF reports were found in {reports_dir}")

    for path in pdf_paths:
        reader = PdfReader(path)
        page_texts = [clean_text(page.extract_text() or "") for page in reader.pages]

        if sum(len(text) for text in page_texts) < minimum_report_characters:
            ocr_texts = _ocr_document(path)
            if ocr_texts:
                page_texts = ocr_texts
                ocr_reports.append(path.name)

        report_month = get_report_month(path.name)
        for page_number, page_text in enumerate(page_texts, start=1):
            if page_text:
                pages.append(
                    {
                        "source": path.name,
                        "month": report_month,
                        "page": page_number,
                        "text": page_text,
                    }
                )
    return pages, ocr_reports


def chunk_text(text: str, max_words: int = 200, overlap: int = 40) -> list[str]:
    """Split page text into overlapping word windows as in the notebook."""
    if max_words <= 0 or overlap < 0 or overlap >= max_words:
        raise ValueError("Chunk size must be positive and overlap smaller than it.")
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks


def _make_chunks(pages: list[dict]) -> list[dict]:
    chunks = []
    for page in pages:
        for chunk in chunk_text(page["text"]):
            chunks.append(
                {
                    "source": page["source"],
                    "month": page["month"],
                    "page": page["page"],
                    "text": chunk,
                }
            )
    return chunks


def build_report_index(
    reports_dir: str | Path = DEFAULT_REPORTS_DIR,
) -> ReportIndex:
    """Extract, chunk, and vectorise the supplied JoPACC reports."""
    reports_dir = Path(reports_dir)
    use_cache = reports_dir.resolve() == DEFAULT_REPORTS_DIR.resolve()

    if use_cache and INDEX_CACHE_PATH.exists():
        try:
            with INDEX_CACHE_PATH.open("rb") as cache_file:
                cached = pickle.load(cache_file)
            return ReportIndex(
                chunks=cached["chunks"],
                vectors=cached["vectors"],
                encoder=SentenceTransformer("all-MiniLM-L6-v2"),
                ocr_reports=cached["ocr_reports"],
                report_count=cached["report_count"],
                page_count=cached["page_count"],
            )
        except (OSError, EOFError, pickle.UnpicklingError, KeyError):
            pass

    pages, ocr_reports = extract_report_pages(reports_dir)
    chunks = _make_chunks(pages)
    if not chunks:
        raise ValueError("No readable report text was extracted.")

    chunk_texts = [chunk["text"] for chunk in chunks]
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = encoder.encode(chunk_texts)
    index = ReportIndex(
        chunks=chunks,
        vectors=vectors,
        encoder=encoder,
        ocr_reports=ocr_reports,
        report_count=len({page["source"] for page in pages}),
        page_count=len(pages),
    )

    if use_cache:
        with INDEX_CACHE_PATH.open("wb") as cache_file:
            pickle.dump(
                {
                    "chunks": index.chunks,
                    "vectors": index.vectors,
                    "ocr_reports": index.ocr_reports,
                    "report_count": index.report_count,
                    "page_count": index.page_count,
                },
                cache_file,
            )

    return index


_REPORT_INDEX: ReportIndex | None = None


def get_report_index() -> ReportIndex:
    """Build the index on first use so the other pages work without the reports."""
    global _REPORT_INDEX
    if _REPORT_INDEX is None:
        _REPORT_INDEX = build_report_index()
    return _REPORT_INDEX


def retrieve_chunks(
    question: str,
    top_k: int = 5,
) -> list[dict]:
    """Retrieve the globally highest-ranked evidence for the original question."""
    if not question.strip():
        return []
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    index = get_report_index()
    query_vector = index.encoder.encode([question])
    similarities = cosine_similarity(query_vector, index.vectors)[0]
    result_count = min(top_k, len(index.chunks))
    top_indices = similarities.argsort()[-result_count:][::-1]

    results = []
    for chunk_index in top_indices:
        chunk_index = int(chunk_index)
        results.append(
            {
                "score": float(similarities[chunk_index]),
                "source": index.chunks[chunk_index]["source"],
                "month": index.chunks[chunk_index]["month"],
                "page": index.chunks[chunk_index]["page"],
                "text": index.chunks[chunk_index]["text"],
            }
        )
    return results


def _generate_with_openrouter(
    question: str,
    retrieved: list[dict],
    evidence_is_relevant: bool,
) -> str:
    if not OPENROUTER_API_KEY:
        raise ValueError(
            "Set the OPENROUTER_API_KEY environment variable to ask report questions."
        )
    context_parts = []
    for result in retrieved:
        context_parts.append(
            f"SOURCE: {result['source']} | MONTH: {result['month']} | "
            f"PAGE: {result['page']}\n{result['text']}"
        )
    context = "\n\n".join(context_parts)
    evidence_status = "SUFFICIENT" if evidence_is_relevant else "INSUFFICIENT"
    prompt = f"""
Use ONLY the supplied context to answer the question.

- First use ordinary conversational judgement to decide whether the user's meaning is clear enough to answer.
- Do not require a month, year, payment system or indicator unless that detail is genuinely necessary to understand what the user means.
- Treat clear broad requests, such as summarising overall payments throughout 2025, as complete questions and answer them from the available evidence.
- For a clear year-wide or collection-wide summary, synthesise the relevant themes, figures and changes present in the retrieved reports. The context does not need to contain every report for you to provide a useful evidence-based overview, but clearly describe the evidence's coverage and never invent information for missing periods.
- Tolerate minor spelling, grammar and wording mistakes when the intended meaning is still clear.
- If the sentence is genuinely incomplete, has multiple plausible meanings, or is difficult to decipher, briefly identify the specific unclear or missing detail and ask one concise clarification question. Do not guess the user's meaning.
- Apply these clarification instructions before considering the retrieval status.
- Treat RETRIEVAL STATUS and the relevance score as useful guidance, not as an automatic reason to refuse. Always inspect the passages themselves.
- When any retrieved passage contains information relevant to the question, provide the most useful answer that can be supported by that evidence.
- For summaries, synthesise the available relevant content instead of refusing merely because the context does not cover every page, indicator or month.
- Ignore passages from an irrelevant reporting period when other retrieved passages match the requested period.
- Lead with the useful findings. Mention evidence limitations briefly only when they materially affect the answer.
- Reply exactly "This was not found in the JoPACC reports we have." only when none of the retrieved passages contains usable evidence for the question.
- Do not guess missing details or use external knowledge.
- In JoPACC report terminology, CliQ figures are labelled as transfers issued by banks or transactions through bank accounts, while JoMoPay figures are labelled as transfers issued by mobile wallets. Use the retrieved passages for every reported number.
- If values differ by month, identify every month clearly.
- Write two or three short, professional paragraphs.

RETRIEVAL STATUS: {evidence_status}
TOP RELEVANCE SCORE: {retrieved[0]['score']:.3f}
MINIMUM RELEVANCE SCORE: {MINIMUM_RELEVANCE_SCORE:.2f}

Context:
{context}

Question: {question}

Answer:
    """
    response = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://127.0.0.1:5000",
            "X-OpenRouter-Title": "Flask App",
        },
        json={
            "model": OPENROUTER_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        },
        timeout=120,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    if not isinstance(content, str) or not content.strip():
        raise ValueError("The answer came back empty, please try asking again.")
    return content.strip()


def rag_query(
    question: str,
    top_k: int = 5,
) -> tuple[str, list[dict]]:
    """Answer one question and return the exact supporting evidence passages."""
    question = clean_text(question)
    if not question:
        return "Please write a question about the JoPACC reports first.", []
    if len(question) > 500:
        return "Please make the question shorter, 500 characters at most.", []

    retrieved = retrieve_chunks(question, top_k=top_k)
    if not retrieved:
        return "This was not found in the JoPACC reports we have.", []

    evidence_is_relevant = retrieved[0]["score"] >= MINIMUM_RELEVANCE_SCORE
    generated_answer = _generate_with_openrouter(
        question,
        retrieved,
        evidence_is_relevant,
    )
    return generated_answer, retrieved


if __name__ == "__main__":
    index = get_report_index()
    print(
        f"Indexed {index.report_count} reports, {index.page_count} pages, "
        f"and {len(index.chunks)} chunks."
    )
