from pydantic import BaseModel, validator
from pydantic.fields import ModelField


class HyperParameterModel(BaseModel):
    min_count: int
    size: int
    sg: int  # validate choices between [0, 1]
    window: int
    iter: int
    sample: float
    hs: int  # validate choices between [0, 1]
    negative: int
    ns_exponent: float
    workers: int

    @validator("sg", "hs")
    def validate_choices(cls, v: int, field: ModelField) -> int:
        if not v in [0, 1]:
            raise ValueError("Not valid value for {}".format(field.alias))
        return v

