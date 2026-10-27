from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from datamate.io.questions import Question
from datamate.scoring.answer import Answer


@runtime_checkable
class System(Protocol):
    name: str

    def answer(self, question: Question, workdir: Path) -> tuple[Answer, dict[str, Any]]: ...
