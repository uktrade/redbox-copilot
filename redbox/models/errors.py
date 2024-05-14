from pydantic import AnyUrl, BaseModel, Field


class APIErrorDetail(BaseModel):
    """A single API error loosely defined to the RFC 9457 format."""

    parameter: str = Field(description="The name of the query or path parameter that is the source of error.")
    detail: str = Field(
        description=(
            "A granular description on the specific error related to a body property, "
            | "query parameter, path parameters, and/or header."
        )
    )


class APIErrorResponse(BaseModel):
    """An API error object loosely defined to the RFC 9457 format."""

    type: AnyUrl = Field(description="A URI reference that identifies the problem type.")
    status: int = Field(
        description="The HTTP status code generated by the origin server for this occurrence of the problem."
    )
    title: str = Field(description="A short, human-readable summary of the problem type.")
    detail: str = Field(description="A human-readable explanation specific to this occurrence of the problem.")
    errors: list[APIErrorDetail] = Field(
        description="An array of error details to accompany a problem details response."
    )


class APIError404(APIErrorResponse):
    type: AnyUrl = "error/not-found"
    status: int = 404
    title: str = "File not found"
