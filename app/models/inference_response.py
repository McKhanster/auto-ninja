from pydantic import BaseModel

class InferenceResponse(BaseModel):
    prediction: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": "Hi there! I'm here to help you with any questions you might have."
            }
        }