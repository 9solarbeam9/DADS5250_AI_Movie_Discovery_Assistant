from pydantic import BaseModel, Field


class TypeAliases(BaseModel):
    type_id: int = Field(description="The id given for the request type")
    aliases: list[str] = Field(description="Short phrases a resident might use to report this problem")


class AliasBatch(BaseModel):
    items: list[TypeAliases]


class EvalMessage(BaseModel):
    type_id: int = Field(description="The id given for the request type")
    message: str = Field(description="One realistic resident message reporting this problem")


class EvalBatch(BaseModel):
    items: list[EvalMessage]
