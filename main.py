"""

The FastAPI web service. Two endpoints:
  GET  /health  → {"status": "ok"}
  POST /chat    → {"reply": ..., "recommendations": [...], "end_of_conversation": ...}
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
import uvicorn
import agent  # our agent.py


app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])



class Message(BaseModel):
    role: str
    content: str

    @field_validator("role")
    @classmethod
    def check_role(cls, v):
        if v not in ("user", "assistant"):
            raise ValueError("role must be 'user' or 'assistant'")
        return v


class ChatRequest(BaseModel):
    messages: list[Message]


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    # Convert pydantic objects → plain dicts
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # Must have at least one message
    if not messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    # Last message must be from the user (we reply to user, not assistant)
    if messages[-1]["role"] != "user":
        raise HTTPException(status_code=400, detail="Last message must be from 'user'")

    # Cap at 8 messages (spec requirement: max 8 turns)
    messages = messages[-8:]

    # Call our agent
    result = agent.chat(messages)

    return ChatResponse(
        reply=result["reply"],
        recommendations=[
            Recommendation(name=r["name"], url=r["url"], test_type=r["test_type"])
            for r in result["recommendations"]
        ],
        end_of_conversation=result["end_of_conversation"]
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)