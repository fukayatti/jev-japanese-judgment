from pydantic import BaseModel, model_validator


class JevExample(BaseModel):
    """統一Jev形式。全データセットの変換スクリプトはこの形式を出力する。"""

    id: str
    source_dataset: str
    context: str
    question: str
    candidates: list[str]
    label: int
    task_type: str

    @model_validator(mode="after")
    def _check_label_range(self) -> "JevExample":
        if not (0 <= self.label < len(self.candidates)):
            raise ValueError(
                f"label={self.label} is out of range for "
                f"{len(self.candidates)} candidates (id={self.id})"
            )
        return self
