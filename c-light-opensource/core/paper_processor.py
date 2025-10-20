"""
Paper Processing with Full PDF Extraction
Extracts full text from PDFs, chunks into sections, tracks citations
"""

import logging
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import PyPDF2

from .base_types import Paper

logger = logging.getLogger(__name__)


@dataclass
class PaperChunk:
    """A chunk of text from a paper with source tracking"""
    paper_id: str
    chunk_id: str  # paper_id + chunk number

    # Content
    text: str

    # Source tracking for citations
    section: Optional[str] = None  # Introduction, Methods, Results, etc.
    page_start: Optional[int] = None
    page_end: Optional[int] = None

    # Metadata
    paper_title: str = ""
    authors: List[str] = None

    def __post_init__(self):
        if self.authors is None:
            self.authors = []

    def get_citation(self) -> str:
        """Get formatted citation for this chunk"""
        authors_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            authors_str += " et al."

        location = ""
        if self.section:
            location = f", {self.section} section"
        if self.page_start:
            if self.page_end and self.page_end != self.page_start:
                location += f", pp. {self.page_start}-{self.page_end}"
            else:
                location += f", p. {self.page_start}"

        return f"{authors_str}. \"{self.paper_title}\"{location}"


class PaperProcessor:
    """
    Processes papers: extracts full text, chunks intelligently, tracks sources

    This ensures every piece of knowledge can be traced back to the source paper
    """

    def __init__(
        self,
        chunk_size: int = 1000,  # tokens (roughly 750 words)
        chunk_overlap: int = 200,  # overlap between chunks
        extract_sections: bool = True  # Try to identify paper sections
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.extract_sections = extract_sections

        # Section detection patterns
        self.section_patterns = [
            (r'\n\s*ABSTRACT\s*\n', 'abstract'),
            (r'\n\s*(?:1\.?\s+)?INTRODUCTION\s*\n', 'introduction'),
            (r'\n\s*(?:2\.?\s+)?(?:METHODS|METHODOLOGY|MATERIALS AND METHODS)\s*\n', 'methods'),
            (r'\n\s*(?:3\.?\s+)?RESULTS\s*\n', 'results'),
            (r'\n\s*(?:4\.?\s+)?DISCUSSION\s*\n', 'discussion'),
            (r'\n\s*(?:5\.?\s+)?CONCLUSION\s*\n', 'conclusion'),
            (r'\n\s*REFERENCES\s*\n', 'references'),
        ]

    def process_paper(self, paper: Paper, pdf_path: Optional[Path] = None) -> List[PaperChunk]:
        """
        Process a paper into chunks with full source tracking

        Args:
            paper: Paper object (has abstract at minimum)
            pdf_path: Optional path to PDF file for full text

        Returns:
            List of PaperChunk objects, each with source tracking
        """
        chunks = []

        # If PDF available, extract full text
        if pdf_path and pdf_path.exists():
            logger.info(f"Extracting full text from PDF: {paper.paper_id}")
            full_text, page_texts = self._extract_pdf_text(pdf_path)

            if full_text:
                # Detect sections if requested
                if self.extract_sections:
                    sections = self._detect_sections(full_text)
                    chunks.extend(self._chunk_with_sections(
                        paper, full_text, sections, page_texts
                    ))
                else:
                    # Simple chunking without section detection
                    chunks.extend(self._chunk_text(
                        paper, full_text, page_texts
                    ))
            else:
                logger.warning(f"Could not extract text from PDF for {paper.paper_id}, using abstract only")
                chunks.append(self._create_abstract_chunk(paper))

        else:
            # No PDF, use abstract only
            logger.debug(f"No PDF for {paper.paper_id}, using abstract only")
            chunks.append(self._create_abstract_chunk(paper))

        logger.info(f"Processed {paper.paper_id} into {len(chunks)} chunks")

        return chunks

    def _extract_pdf_text(self, pdf_path: Path) -> Tuple[str, Dict[int, str]]:
        """
        Extract text from PDF

        Returns:
            (full_text, page_texts_dict)
        """
        try:
            full_text = []
            page_texts = {}

            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)

                for page_num, page in enumerate(reader.pages, start=1):
                    text = page.extract_text()
                    if text:
                        page_texts[page_num] = text
                        full_text.append(text)

            return "\n\n".join(full_text), page_texts

        except Exception as e:
            logger.error(f"Error extracting PDF: {e}")
            return "", {}

    def _detect_sections(self, text: str) -> List[Tuple[str, int]]:
        """
        Detect paper sections

        Returns:
            List of (section_name, start_position)
        """
        sections = []

        for pattern, section_name in self.section_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                sections.append((section_name, match.start()))

        # Sort by position
        sections.sort(key=lambda x: x[1])

        return sections

    def _chunk_with_sections(
        self,
        paper: Paper,
        text: str,
        sections: List[Tuple[str, int]],
        page_texts: Dict[int, str]
    ) -> List[PaperChunk]:
        """
        Chunk text while preserving section boundaries
        """
        chunks = []

        for i, (section_name, start_pos) in enumerate(sections):
            # Find section end
            if i + 1 < len(sections):
                end_pos = sections[i + 1][1]
            else:
                end_pos = len(text)

            section_text = text[start_pos:end_pos]

            # Chunk this section if it's too long
            if len(section_text) > self.chunk_size * 4:  # Rough character estimate
                section_chunks = self._split_long_section(section_text)
            else:
                section_chunks = [section_text]

            # Create chunks for this section
            for chunk_num, chunk_text in enumerate(section_chunks):
                # Find which pages this chunk spans
                page_start, page_end = self._find_pages(chunk_text, page_texts)

                chunk = PaperChunk(
                    paper_id=paper.paper_id,
                    chunk_id=f"{paper.paper_id}_{section_name}_{chunk_num}",
                    text=chunk_text.strip(),
                    section=section_name,
                    page_start=page_start,
                    page_end=page_end,
                    paper_title=paper.title,
                    authors=paper.authors
                )
                chunks.append(chunk)

        return chunks

    def _chunk_text(
        self,
        paper: Paper,
        text: str,
        page_texts: Dict[int, str]
    ) -> List[PaperChunk]:
        """
        Simple sliding window chunking without section detection
        """
        chunks = []

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)

        current_chunk = []
        current_length = 0
        chunk_num = 0

        for sentence in sentences:
            sentence_length = len(sentence.split())

            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Create chunk
                chunk_text = " ".join(current_chunk)
                page_start, page_end = self._find_pages(chunk_text, page_texts)

                chunk = PaperChunk(
                    paper_id=paper.paper_id,
                    chunk_id=f"{paper.paper_id}_chunk_{chunk_num}",
                    text=chunk_text,
                    page_start=page_start,
                    page_end=page_end,
                    paper_title=paper.title,
                    authors=paper.authors
                )
                chunks.append(chunk)

                # Start new chunk with overlap
                overlap_sentences = current_chunk[-3:] if len(current_chunk) > 3 else current_chunk
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s.split()) for s in current_chunk)
                chunk_num += 1
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            page_start, page_end = self._find_pages(chunk_text, page_texts)

            chunk = PaperChunk(
                paper_id=paper.paper_id,
                chunk_id=f"{paper.paper_id}_chunk_{chunk_num}",
                text=chunk_text,
                page_start=page_start,
                page_end=page_end,
                paper_title=paper.title,
                authors=paper.authors
            )
            chunks.append(chunk)

        return chunks

    def _split_long_section(self, text: str) -> List[str]:
        """Split a long section into smaller chunks"""
        # Simple sentence-based splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence.split())

            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                # Keep some overlap
                current_chunk = current_chunk[-2:] + [sentence]
                current_length = sum(len(s.split()) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _find_pages(self, chunk_text: str, page_texts: Dict[int, str]) -> Tuple[Optional[int], Optional[int]]:
        """
        Find which pages a chunk of text spans

        Returns:
            (page_start, page_end) or (None, None) if not found
        """
        if not page_texts:
            return None, None

        # Take first 100 chars of chunk to search
        search_text = chunk_text[:100]

        found_pages = []
        for page_num, page_text in page_texts.items():
            if search_text in page_text:
                found_pages.append(page_num)

        if found_pages:
            return min(found_pages), max(found_pages)

        return None, None

    def _create_abstract_chunk(self, paper: Paper) -> PaperChunk:
        """Create a single chunk from abstract when no PDF available"""
        return PaperChunk(
            paper_id=paper.paper_id,
            chunk_id=f"{paper.paper_id}_abstract",
            text=paper.abstract,
            section="abstract",
            paper_title=paper.title,
            authors=paper.authors
        )
