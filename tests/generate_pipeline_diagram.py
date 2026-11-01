"""Regenerate the pipeline diagram from Pydantic Graph's built-in renderer."""

from pathlib import Path

from petri.execution.pipeline import pipeline_graph

BEGIN = "<!-- BEGIN GENERATED PIPELINE -->"
END = "<!-- END GENERATED PIPELINE -->"


def diagram_text() -> str:
    return pipeline_graph.render() + "\n"


def diagram_block() -> str:
    return f"{BEGIN}\n```mermaid\n{diagram_text()}```\n{END}"


def update_docs(root: Path) -> None:
    docs = root / "docs"
    (docs / "pipeline-graph.mmd").write_text(diagram_text(), encoding="utf-8")
    page = docs / "pipeline-graph.md"
    text = page.read_text(encoding="utf-8")
    before, remainder = text.split(BEGIN, 1)
    _, after = remainder.split(END, 1)
    page.write_text(before + diagram_block() + after, encoding="utf-8")


if __name__ == "__main__":
    update_docs(Path(__file__).resolve().parents[1])
