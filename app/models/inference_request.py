from pydantic import BaseModel

class InferenceRequest(BaseModel):
    message: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Hello, how can I assist you today?"
            }
        }