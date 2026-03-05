from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langserve import add_routes
import uvicorn
import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv
load_dotenv()

os.environ['groq_api_key'] = os.getenv('groq_api_key')

app = FastAPI(
    titles="Langchain Server",
    version="1.0",
    description="A simple API server"
)

"""
from pydantic import BaseModel

# Models for input and output
class TopicRequest(BaseModel):
    topic: str

class TextResponse(BaseModel):
    result: str
"""

model = ChatOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["groq_api_key"],
    model="llama-3.1-8b-instant"
)

add_routes(
    app,
    model,
    path="/openai"
)

llm = Ollama(model="llama2")

prompt1 = ChatPromptTemplate.from_template("Write me an essay about {topic} in 100 words")
prompt2 = ChatPromptTemplate.from_template("Write me a poem about {topic} in 100 words")

"""
Code for manual /docs
# Endpoint for essay generation
@app.post("/essay", response_model=TextResponse)
async def generate_essay(request: TopicRequest):
    chain = prompt1 | model
    result = chain.invoke({"topic": request.topic})
    return {"result": result.content}

# Endpoint for poem generation
@app.post("/poem", response_model=TextResponse)
async def generate_poem(request: TopicRequest):
    chain = prompt2 | llm
    result = chain.invoke({"topic": request.topic})
    return {"result": result}
"""

add_routes(
    app,
    prompt1|model,
    path="/essay"
)
add_routes(
    app,
    prompt2|llm,
    path="/poem"
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
