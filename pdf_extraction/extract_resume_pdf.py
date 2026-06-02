import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv


@dataclass
class PageText:
    page_number: int
    text: str
    source: str


@dataclass
class ExtractionResult:
    file_name: str
    extractor_used: str
    is_likely_scanned: bool
    page_count: int
    combined_text: str
    section_slices: Dict[str, str]
    diagnostics: Dict[str, str]


SECTION_ALIASES = {
    "summary": "summary",
    "profile": "summary",
    "about": "summary",
    "skills": "skills",
    "technical skills": "technical_skills",
    "soft skills": "soft_skills",
    "experience": "experience",
    "work experience": "experience",
    "employment": "experience",
    "education": "education",
    "projects": "projects",
    "selected projects": "projects",
    "certifications": "certifications",
    "it vedant certifications": "certifications",
    "achievements": "achievements",
}


def _normalize_header(value: str) -> str:
    return re.sub(r"[^a-z ]", "", value.lower()).strip()


def _extract_with_pymupdf(pdf_path: Path) -> Tuple[List[PageText], str]:
    import fitz

    doc = fitz.open(pdf_path)
    pages: List[PageText] = []
    for idx, page in enumerate(doc, start=1):
        text = page.get_text("text") or ""
        pages.append(PageText(page_number=idx, text=text.strip(), source="pymupdf"))
    doc.close()
    return pages, "pymupdf"


def _extract_with_pdfplumber(pdf_path: Path) -> Tuple[List[PageText], str]:
    import pdfplumber

    pages: List[PageText] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append(PageText(page_number=idx, text=text.strip(), source="pdfplumber"))
    return pages, "pdfplumber"


def _extract_with_pypdf(pdf_path: Path) -> Tuple[List[PageText], str]:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    pages: List[PageText] = []
    for idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(PageText(page_number=idx, text=text.strip(), source="pypdf"))
    return pages, "pypdf"


def _looks_scanned(pages: List[PageText]) -> bool:
    if not pages:
        return True
    total_chars = sum(len(p.text) for p in pages)
    avg_chars = total_chars / max(len(pages), 1)
    return avg_chars < 60


def _ocr_fallback(pdf_path: Path) -> Tuple[List[PageText], str]:
    import fitz
    import pytesseract
    from PIL import Image, ImageEnhance, ImageFilter

    doc = fitz.open(pdf_path)
    pages: List[PageText] = []

    for idx, page in enumerate(doc, start=1):
        pix = page.get_pixmap(dpi=300)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        image = image.convert("L")
        image = image.filter(ImageFilter.MedianFilter(size=3))
        image = ImageEnhance.Contrast(image).enhance(1.8)
        text = pytesseract.image_to_string(image, config="--oem 3 --psm 6") or ""
        pages.append(PageText(page_number=idx, text=text.strip(), source="ocr_pytesseract"))

    doc.close()
    return pages, "ocr_pytesseract"


def _quality_score(text: str) -> int:
    if not text.strip():
        return 0
    score = len(text)
    score += len(re.findall(r"\b(SUMMARY|EDUCATION|EXPERIENCE|PROJECTS|SKILLS)\b", text, flags=re.I)) * 120
    score += text.count(".") * 10
    score -= len(re.findall(r"[A-Z]{20,}", text)) * 30
    score -= len(re.findall(r"[A-Z]{4,}[A-Z]{4,}", text)) * 15
    return score


def _segment_sections(text: str) -> Dict[str, str]:
    if not text.strip():
        return {"full_text": ""}

    lines = [ln.strip() for ln in re.sub(r"\r", "", text).splitlines() if ln.strip()]
    if not lines:
        return {"full_text": text}

    candidates = sorted(SECTION_ALIASES.keys(), key=len, reverse=True)
    header_hits: List[Tuple[int, str]] = []
    for idx, line in enumerate(lines):
        normalized_line = _normalize_header(line)
        direct = SECTION_ALIASES.get(normalized_line)
        if direct:
            header_hits.append((idx, direct))
            continue

        if len(normalized_line.split()) <= 8:
            for cand in candidates:
                if re.search(rf"\b{re.escape(cand)}\b", normalized_line):
                    header_hits.append((idx, SECTION_ALIASES[cand]))

    if not header_hits:
        return {"full_text": text}

    early_hits = [hit for hit in header_hits if hit[0] <= 20]
    if len(early_hits) >= 4:
        return {"full_text": text}

    section_slices: Dict[str, str] = {}
    dedup_hits: List[Tuple[int, str]] = []
    for hit in header_hits:
        if not dedup_hits or dedup_hits[-1] != hit:
            dedup_hits.append(hit)

    for i, (start_idx, header) in enumerate(dedup_hits):
        end_idx = dedup_hits[i + 1][0] if i + 1 < len(dedup_hits) else len(lines)
        content = "\n".join(lines[start_idx + 1 : end_idx]).strip()
        if content:
            existing = section_slices.get(header, "")
            section_slices[header] = f"{existing}\n{content}".strip() if existing else content
    return section_slices or {"full_text": text}


def _parse_with_groq(raw_resume_text: str, model: str) -> Dict:
    from groq import Groq

    if not raw_resume_text.strip():
        return {"error": "empty_resume_text"}

    client = Groq(api_key=os.getenv("GROQ_API"))
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a resume parser. Output only valid JSON."},
            {"role": "user", "content": raw_resume_text},
        ],
    )
    return json.loads(response.choices[0].message.content)


def extract_pdf(pdf_path: Path) -> ExtractionResult:
    errors: Dict[str, str] = {}
    pages: List[PageText] = []
    extractor_used = ""
    best_score = -1

    for extractor in (_extract_with_pymupdf, _extract_with_pdfplumber, _extract_with_pypdf):
        name = extractor.__name__.replace("_extract_with_", "")
        try:
            candidate_pages, candidate_name = extractor(pdf_path)
            candidate_text = "\n".join(page.text for page in candidate_pages if page.text)
            score = _quality_score(candidate_text)
            if candidate_name == "pdfplumber":
                score += 20
            if score > best_score:
                pages = candidate_pages
                extractor_used = candidate_name
                best_score = score
        except Exception as exc:
            errors[name] = str(exc)

    if not pages or not any(page.text for page in pages) or _looks_scanned(pages):
        try:
            ocr_pages, ocr_name = _ocr_fallback(pdf_path)
            if ocr_pages and any(page.text for page in ocr_pages):
                ocr_text = "\n".join(page.text for page in ocr_pages if page.text)
                ocr_score = _quality_score(ocr_text)
                if ocr_score >= best_score:
                    extractor_used = ocr_name
                    pages = ocr_pages
                    best_score = ocr_score
        except Exception as exc:
            errors["ocr_pytesseract"] = str(exc)

    combined_text = "\n\n".join(f"[Page {page.page_number}]\n{page.text}" for page in pages if page.text).strip()
    scanned = _looks_scanned(pages)
    section_slices = _segment_sections(combined_text) if combined_text else {"full_text": ""}
    diagnostics: Dict[str, str] = {}
    if errors:
        diagnostics["fallback_errors"] = json.dumps(errors)
    if scanned:
        diagnostics["scan_hint"] = "This PDF may be scanned/image-only. OCR fallback was attempted."

    return ExtractionResult(
        file_name=pdf_path.name,
        extractor_used=extractor_used or "none",
        is_likely_scanned=scanned,
        page_count=len(pages),
        combined_text=combined_text,
        section_slices=section_slices,
        diagnostics=diagnostics,
    )


def main() -> None:
    load_dotenv(Path("pdf_extraction") / ".env")
    parser = argparse.ArgumentParser(description="Resume PDF extractor with fallback chain.")
    parser.add_argument("--input-dir", default=str(Path("test_files")))
    parser.add_argument("--output-dir", default=str(Path("pdf_extraction") / "output"))
    parser.add_argument("--use-groq", action="store_true")
    parser.add_argument("--groq-model", default="llama-3.3-70b-versatile")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir = output_dir / "parsed"
    if args.use_groq:
        parsed_dir.mkdir(parents=True, exist_ok=True)

    for pdf_path in sorted(input_dir.glob("*.pdf")):
        result = extract_pdf(pdf_path)
        output_path = output_dir / f"{pdf_path.stem}.json"
        output_path.write_text(json.dumps(asdict(result), indent=2, ensure_ascii=True), encoding="utf-8")
        if args.use_groq:
            parsed = _parse_with_groq(result.combined_text, args.groq_model)
            parsed_path = parsed_dir / f"{pdf_path.stem}_parsed.json"
            parsed_path.write_text(json.dumps(parsed, indent=2, ensure_ascii=True), encoding="utf-8")


if __name__ == "__main__":
    main()
