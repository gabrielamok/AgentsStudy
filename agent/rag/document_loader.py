import hashlib
import logging
import os
import stat
from pathlib import Path
from typing import NoReturn

from markdown_it import MarkdownIt
from pydantic import JsonValue

from agent.rag.contracts import KnowledgeDocument


logger = logging.getLogger(__name__)
DEFAULT_KNOWLEDGE_DIRECTORY = Path(__file__).resolve().parents[2] / "knowledge"
SUPPORTED_EXTENSIONS = frozenset({".md", ".markdown", ".txt"})


class DocumentLoader:
    def __init__(self, knowledge_directory: str | Path = DEFAULT_KNOWLEDGE_DIRECTORY) -> None:
        self.knowledge_directory = Path(knowledge_directory).resolve(strict=True)
        if not self.knowledge_directory.is_dir():
            raise NotADirectoryError(self.knowledge_directory)
        self._markdown = MarkdownIt("commonmark")

    def _contained_path(self, source_path: str | Path) -> Path:
        candidate = Path(source_path)
        if not candidate.is_absolute():
            candidate = self.knowledge_directory / candidate
        resolved = candidate.resolve()
        if not resolved.is_relative_to(self.knowledge_directory):
            raise PermissionError(f"Source is outside the configured knowledge directory: {source_path}")
        return resolved

    def load_document(self, source_path: str | Path) -> KnowledgeDocument | None:
        """Load a contained file; source_path is canonical, IDs use relative paths."""
        path = self._contained_path(source_path)
        mode = path.stat().st_mode
        if stat.S_ISDIR(mode):
            raise IsADirectoryError(path)
        if not stat.S_ISREG(mode):
            logger.warning("Ignoring %s: not a regular file", path)
            return None
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            logger.info("Ignoring %s: unsupported file type %r", path, path.suffix or "<none>")
            return None
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            logger.exception("Cannot read knowledge document %s as UTF-8", path)
            raise

        headings: list[JsonValue] = []
        title = path.stem
        title_found = False
        if path.suffix.lower() != ".txt":
            tokens = self._markdown.parse(content)
            for position, token in enumerate(tokens):
                if token.type == "heading_open":
                    text = tokens[position + 1].content.strip()
                    level = int(token.tag[1:])
                    headings.append({
                        "level": level, "text": text,
                        "line": token.map[0] + 1 if token.map else None,
                    })
                    if level == 1 and text and not title_found:
                        title = text
                        title_found = True

        relative_path = path.relative_to(self.knowledge_directory).as_posix()
        return KnowledgeDocument(
            document_id=hashlib.sha256(relative_path.encode("utf-8")).hexdigest(),
            title=title, source_path=path.as_posix(), content=content,
            metadata={
                "file_name": path.name, "relative_path": relative_path,
                "file_type": path.suffix.lower(), "headings": headings,
            },
        )

    def load_documents(self) -> list[KnowledgeDocument]:
        """Recursively load in stable relative-path order without following directory links."""
        def raise_walk_error(error: OSError) -> NoReturn:
            logger.error("Cannot enumerate knowledge directory: %s", error)
            raise error

        paths: list[Path] = []
        for directory, subdirectories, file_names in os.walk(
            self.knowledge_directory, followlinks=False, onerror=raise_walk_error,
        ):
            current = self._contained_path(directory)
            for name in sorted(subdirectories):
                candidate = current / name
                resolved = self._contained_path(candidate)
                if candidate.is_symlink() or resolved != candidate:
                    logger.warning("Ignoring %s: directory links are not traversed", candidate)
                    subdirectories.remove(name)
            subdirectories.sort()
            paths.extend(current / name for name in file_names)

        documents: list[KnowledgeDocument] = []
        for path in sorted(paths, key=lambda source: source.relative_to(self.knowledge_directory).as_posix()):
            document = self.load_document(path)
            if document is not None:
                documents.append(document)
        return documents