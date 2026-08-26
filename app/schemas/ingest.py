from pydantic import BaseModel, ConfigDict

class IngestResponse (BaseModel):
    source: str
    chunks : int 
    cached  :bool
    total_chunks:int
    total_sources:int

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "source": "example_source.txt",
                "chunks": 10,
                "cached": False,
                "total_chunks": 100,
                "total_sources": 5
            }
        }
    )