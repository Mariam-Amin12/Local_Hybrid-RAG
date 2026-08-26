from pydantic import BaseModel, ConfigDict

from app.schemas.query import SourceChunk


class ChatMessage(BaseModel):
    role: str 
    content: str 

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "role": "user",
                "content": "What is the main topic of the podcast?"
            }
        }
    )

class ChatRequest(BaseModel):
    query:str
    session_id:str|None = None
    llm_api_key:str
    provider:str="ollama"
    top_k:int=5

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "What is the main topic of the podcast?",
                "session_id": "session_123",
                "llm_api_key": "your_api_key",
                "provider": "ollama",
                "top_k": 5
            }
        }
    )

class ChatResponse(BaseModel):
    answer:str
    session_id:str
    sources:list[SourceChunk]
    total_hits:int

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "answer": "The main topic of the podcast is AI and machine learning.",
                "session_id": "session_123",
                "sources": [
                    {
                        "text": "The podcast discusses the latest trends in AI and machine learning.",
                        "source": "podcast_episode_1.mp3",
                        "score": 0.95
                    }
                ],
                "total_hits": 1
            }
        }
    )