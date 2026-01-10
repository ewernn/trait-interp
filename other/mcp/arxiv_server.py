#!/usr/bin/env python3
"""
MCP server for fetching arXiv papers as plain text.

Input: arXiv ID (e.g., "2502.16681" or "2502.16681v1")
Output: Full paper text extracted from HTML

Usage:
    Add to ~/.claude.json via: claude mcp add arxiv python /path/to/arxiv_server.py
"""

import io
import re
import time
import requests
from bs4 import BeautifulSoup
from mcp.server.fastmcp import FastMCP
from pypdf import PdfReader

mcp = FastMCP("arxiv")

# Rate limiting: arXiv asks for 1 request per 3 seconds
_last_request_time = 0

# Cache fetched papers to avoid re-fetching for chunked reads
_paper_cache: dict[str, str] = {}

def _rate_limit():
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < 3:
        time.sleep(3 - elapsed)
    _last_request_time = time.time()

def _clean_arxiv_id(arxiv_id: str) -> str:
    """Extract arxiv ID from various input formats."""
    # Handle full URLs
    if "arxiv.org" in arxiv_id:
        match = re.search(r'(\d{4}\.\d{4,5})(v\d+)?', arxiv_id)
        if match:
            return match.group(0)
    # Handle bare IDs
    match = re.match(r'^(\d{4}\.\d{4,5})(v\d+)?$', arxiv_id.strip())
    if match:
        return match.group(0)
    return arxiv_id.strip()

def _extract_text_from_html(html: str) -> str:
    """Extract clean text from arXiv HTML."""
    soup = BeautifulSoup(html, 'html.parser')

    # Remove script, style, nav elements
    for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
        tag.decompose()

    # Try to find main article content
    article = soup.find('article') or soup.find('main') or soup.find('div', class_='ltx_page_content')

    if article:
        content = article
    else:
        content = soup.body or soup

    # Extract text with some structure preservation
    lines = []

    for element in content.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'li', 'figcaption', 'td', 'th']):
        text = element.get_text(separator=' ', strip=True)
        if not text:
            continue

        # Add markdown-style headers
        if element.name == 'h1':
            lines.append(f"\n# {text}\n")
        elif element.name == 'h2':
            lines.append(f"\n## {text}\n")
        elif element.name == 'h3':
            lines.append(f"\n### {text}\n")
        elif element.name == 'h4':
            lines.append(f"\n#### {text}\n")
        elif element.name == 'li':
            lines.append(f"- {text}")
        else:
            lines.append(text)

    return '\n'.join(lines)

def _extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    lines = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            lines.append(text)
    return '\n\n'.join(lines)

def _fetch_paper(arxiv_id: str) -> str:
    """Fetch paper, using cache if available. Tries HTML first, falls back to PDF."""
    clean_id = _clean_arxiv_id(arxiv_id)

    if clean_id in _paper_cache:
        return _paper_cache[clean_id]

    headers = {
        'User-Agent': 'arxiv-mcp-server/1.0 (research tool; respects rate limits)'
    }

    # Try HTML first (better formatting)
    _rate_limit()
    html_url = f"https://arxiv.org/html/{clean_id}"
    try:
        response = requests.get(html_url, headers=headers, timeout=30)
        if response.status_code == 200:
            text = _extract_text_from_html(response.text)
            if len(text) >= 500:
                full_text = f"# arXiv:{clean_id}\nSource: {html_url}\n\n{text}"
                _paper_cache[clean_id] = full_text
                return full_text
    except requests.exceptions.RequestException:
        pass  # Fall through to PDF

    # Fall back to PDF
    _rate_limit()
    pdf_url = f"https://arxiv.org/pdf/{clean_id}.pdf"
    try:
        response = requests.get(pdf_url, headers=headers, timeout=60)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            return f"Error: Paper {clean_id} not found on arXiv."
        return f"Error fetching PDF: {e}"
    except requests.exceptions.RequestException as e:
        return f"Error fetching PDF: {e}"

    try:
        text = _extract_text_from_pdf(response.content)
    except Exception as e:
        return f"Error extracting text from PDF: {e}"

    if len(text) < 500:
        return f"Error: Could not extract meaningful content from {pdf_url}."

    full_text = f"# arXiv:{clean_id}\nSource: {pdf_url} (PDF)\n\n{text}"
    _paper_cache[clean_id] = full_text
    return full_text

@mcp.tool()
def fetch_arxiv_paper(arxiv_id: str, chunk: int | None = None, chunk_size: int = 15000) -> str:
    """
    Fetch the full text of an arXiv paper from its HTML version.

    Args:
        arxiv_id: arXiv paper ID (e.g., "2502.16681", "2502.16681v1", or full URL)
        chunk: Which chunk to return (0-indexed). None returns full paper.
        chunk_size: Characters per chunk (default 15000, ~4k tokens)

    Returns:
        Full paper text in markdown-ish format (tries HTML first, falls back to PDF)
    """
    text = _fetch_paper(arxiv_id)

    if text.startswith("Error:"):
        return text

    total_chars = len(text)
    total_chunks = (total_chars + chunk_size - 1) // chunk_size

    if chunk is None:
        # Return full paper with chunk info header
        return f"[Full paper: {total_chars:,} chars, {total_chunks} chunks of {chunk_size:,}]\n\n{text}"

    # Return specific chunk
    start = chunk * chunk_size
    end = start + chunk_size

    if start >= total_chars:
        return f"Error: Chunk {chunk} out of range. Paper has {total_chunks} chunks (0-{total_chunks-1})."

    chunk_text = text[start:end]
    return f"[Chunk {chunk}/{total_chunks-1}, chars {start:,}-{min(end, total_chars):,} of {total_chars:,}]\n\n{chunk_text}"

if __name__ == "__main__":
    mcp.run()
