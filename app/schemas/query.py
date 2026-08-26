from pydantic import BaseModel, ConfigDict, Field

class SourceChunk(BaseModel):
    text:str
    source: str
    score:float
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "This is a sample chunk of text from the source.",  
                "source": "example_source.txt",
                "score": 0.95
            }
        }   )

class AskRequest(BaseModel):
    query:str
    top_k:int=5

    model_config = ConfigDict(  # better to be understanded from swagger UI
        json_schema_extra={
            "example": {
                "query": "What is the main topic of the podcast?",
                "top_k": 5
            }
        }
    )

class AskResponse(BaseModel):
    query:str
    sources:list[SourceChunk]
    context:str
    total_hits:int

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "What is the main topic of the podcast?",
                "sources": [
                    {
                        "text": "The podcast discusses the latest trends in AI and machine learning.",
                        "source": "podcast_episode_1.mp3",
                        "score": 0.95
                    },
                    {
                        "text": "In this episode, we explore the impact of AI on various industries.",
                        "source": "podcast_episode_2.mp3",
                        "score": 0.90
                    }
                ],
                "context": "The podcast covers topics related to AI, machine learning, and their applications in different sectors.",
                "total_hits": 2
            }
        }
    )