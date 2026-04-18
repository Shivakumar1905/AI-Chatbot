from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Templates folder
templates = Jinja2Templates(directory="templates")

# Initialize LLM (IMPORTANT: explicit API key)
llm = ChatOpenAI(
    model="gpt-4o-mini",   # you can change to gpt-3.5-turbo if needed
    temperature=0.7,
    api_key=os.environ["OPENAI_API_KEY"]
)

# Prompt template (NO history → stable)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}"),
])

# Chain
chain = prompt | llm

# Request model
class ChatRequest(BaseModel):
    message: str

# Home route
@app.get("/", response_class=HTMLResponse)
async def get_chat_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Chat endpoint
@app.post("/chat")
async def chat_endpoint(chat_request: ChatRequest):
    try:
        user_message = chat_request.message

        response = chain.invoke({"input": user_message})

        return {"response": response.content}

    except Exception as e:
        # Return actual error for debugging
        return {"error": str(e)}