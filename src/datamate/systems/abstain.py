from __future__ import annotations

from pathlib import Path
from typing import Any

from datamate.io.questions import Question
from datamate.scoring.answer import Answer


class AlwaysAbstain:
    name: str = "abstain"

    def answer(self, question: Question, workdir: Path) -> tuple[Answer, dict[str, Any]]:
        return Answer.abstain(), {"policy": "always_abstain"}
