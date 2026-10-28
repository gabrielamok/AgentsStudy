import copy
import hashlib
import json
import re
from collections.abc import Iterator

from markdown_it import MarkdownIt

from agent.rag.contracts import DocumentChunk, KnowledgeDocument


class MarkdownChunker:
    """Character-bounded chunks with whole-word overlap and repeated heading context.

    Paragraphs and other Markdown blocks stay intact when they fit. Overlap is
    at most the configured character count, never crosses sections, and shrinks
    when needed to fit new content. Unfittable headings or words raise ValueError.
    """

    def __init__(self, max_chunk_size: int = 1200, overlap: int = 100) -> None:
        if type(max_chunk_size) is not int or type(overlap) is not int:
            raise TypeError("max_chunk_size and overlap must be integers")
        if max_chunk_size <= 0 or overlap < 0 or overlap >= max_chunk_size:
            raise ValueError("Require max_chunk_size > 0 and 0 <= overlap < max_chunk_size")
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        self._markdown = MarkdownIt("commonmark")

    def _sections(self, document: KnowledgeDocument) -> Iterator[tuple[list[tuple[int, str]], str]]:
        lines = document.content.splitlines(keepends=True)
        headings: list[tuple[int, str]] = []
        start = 0
        tokens = self._markdown.parse(document.content) if document.metadata.get("file_type") != ".txt" else []
        for position, token in enumerate(tokens):
            if token.type != "heading_open" or token.level != 0 or token.map is None:
                continue
            body = "".join(lines[start:token.map[0]]).strip()
            if body:
                yield headings.copy(), body
            level = int(token.tag[1:])
            text = tokens[position + 1].content.strip() or document.title
            while headings and headings[-1][0] >= level:
                headings.pop()
            headings.append((level, text))
            start = token.map[1]
        body = "".join(lines[start:]).strip()
        if body:
            yield headings.copy(), body

    def _blocks(self, content: str) -> list[str]:
        lines = content.splitlines(keepends=True)
        return [
            block for token in self._markdown.parse(content)
            if token.level == 0 and token.map is not None
            if (block := "".join(lines[token.map[0]:token.map[1]]).strip())
        ]

    def _split_block(self, content: str, capacity: int) -> Iterator[str]:
        words = list(re.finditer(r"\S+", content))
        if any(word.end() - word.start() > capacity for word in words):
            raise ValueError("A word cannot fit within max_chunk_size after heading context")
        start = 0
        while start < len(words):
            end = start + 1
            while end < len(words) and words[end].end() - words[start].start() <= capacity:
                end += 1
            yield content[words[start].start():words[end - 1].end()]
            if end == len(words):
                break
            next_start = end
            while next_start > start + 1:
                candidate = next_start - 1
                if (words[end - 1].end() - words[candidate].start() > self.overlap
                        or words[end].end() - words[candidate].start() > capacity):
                    break
                next_start = candidate
            start = next_start

    def _suffix(self, content: str, capacity: int) -> str:
        budget = min(self.overlap, capacity)
        for word in re.finditer(r"\S+", content):
            if len(content) - word.start() <= budget:
                return content[word.start():]
        return ""

    def _pack(self, blocks: list[str], capacity: int) -> Iterator[str]:
        current = ""
        for block in blocks:
            if len(block) > capacity:
                if current:
                    yield current
                    current = ""
                pieces = iter(self._split_block(block, capacity))
                for piece in pieces:
                    if current:
                        yield current
                    current = piece
                continue
            combined = f"{current}\n\n{block}" if current else block
            if len(combined) <= capacity:
                current = combined
            else:
                yield current
                context = self._suffix(current, capacity - len(block) - 2)
                current = f"{context}\n\n{block}" if context else block
        if current:
            yield current

    def chunk_document(self, document: KnowledgeDocument) -> list[DocumentChunk]:
        """Return deterministic, nonempty chunks; leave the input document unchanged."""
        chunks: list[DocumentChunk] = []
        seen: set[str] = set()
        for headings, body in self._sections(document):
            context = headings if headings and headings[0][0] == 1 else [(1, document.title), *headings]
            heading_path = [text for level, text in context]
            prefix = "\n".join(f"{'#' * level} {text}" for level, text in context) + "\n\n"
            capacity = self.max_chunk_size - len(prefix)
            if capacity <= 0:
                raise ValueError("Heading context leaves no content space within max_chunk_size")
            for content in self._pack(self._blocks(body), capacity):
                rendered = prefix + content
                identity = json.dumps([document.document_id, heading_path, rendered], ensure_ascii=True)
                chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
                if chunk_id in seen:
                    continue
                seen.add(chunk_id)
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id, document_id=document.document_id,
                    source_title=document.title, section=heading_path[-1], content=rendered,
                    metadata={
                        "source_path": document.source_path, "heading_path": heading_path.copy(),
                        "chunk_index": len(chunks), "document_metadata": copy.deepcopy(document.metadata),
                    },
                ))
        return chunks