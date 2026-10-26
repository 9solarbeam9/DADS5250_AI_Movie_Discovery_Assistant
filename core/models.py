from pydantic import BaseModel, Field


class TypeAliases(BaseModel):
    type_id: int = Field(description="The id given for the request type")
    aliases: list[str] = Field(
        description="Short phrases a resident might use to report this problem"
    )


class AliasBatch(BaseModel):
    items: list[TypeAliases]


class EvalMessage(BaseModel):
    type_id: int = Field(description="The id given for the request type")
    message: str = Field(
        description="One realistic resident message reporting this problem"
    )


class EvalBatch(BaseModel):
    items: list[EvalMessage]


class Classification(BaseModel):
    type: str = Field(
        description="Exact request type name form the provided list of request types"
    )
    department: str = Field(
        description="Department code (PWDx, BTDT, PARK, or other) for the request type"
    )
    queue: str = Field(
        description="Queue name from the provided list of queues for the request type"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="How confident you are this is the right match, from 0.0 to 1.0",
    )
    clarifying_question: str | None = Field(
        default=None,
        description="One short question to ask the resident if confidence is low, to help clarify the request type. If confidence is high, this can be None.",
    )
