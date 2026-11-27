import os

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

# Defaults point to your local Ollama. The .env file overrides them.
BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
API_KEY = os.getenv("LLM_API_KEY", "ollama")
# LLM_MODEL can hold several models separated by commas. They are tried in order.
MODELS = [m.strip() for m in os.getenv("LLM_MODEL", "mistral").split(",") if m.strip()]

app = FastAPI(title="French Teacher API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

LANGUAGES = {"en": "English", "tl": "Tagalog"}


class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    explain_lang: str = "en"  # "en" or "tl"


def build_system_prompt(explain_lang: str) -> str:
    lang = LANGUAGES.get(explain_lang, "English")
    return f"""You are a friendly private French teacher for Filipino adults living and working in France (housekeepers, babysitters, caregivers). The student is a beginner.

LANGUAGE: Explain in the language the student writes in (English or Tagalog). If unclear, use {lang}. Teach French.

MOST IMPORTANT RULE: Answer exactly what the student asked. If they ask "how do I say X", give the most natural everyday French for X, then 2 or 3 closely related words or phrases that are useful in the same situation. Never answer with a different, more general word.

STYLE:
- Short answers. No long lectures.
- Show each French phrase on its own line like this: French = meaning (simple pronunciation).
- Say when to use "vous" (formal, with employers) or "tu" (friendly, with children).
- Use real French that locals use at work: cleaning products and tasks, babysitting routines, meals, safety, talking with parents and employers.
- Only sometimes (not in every answer), invite the student to say a phrase aloud or to try a short sentence. Vary how you do it.
- If the student writes French, correct it kindly and show the corrected sentence.
- If you are not sure about a French word, say so instead of inventing one."""


@app.get("/")
def health():
    return {"status": "ok", "models": MODELS, "provider": BASE_URL}


@app.post("/chat")
async def chat(req: ChatRequest):
    messages = [{"role": "system", "content": build_system_prompt(req.explain_lang)}]
    messages += [m.model_dump() for m in req.messages[-12:]]  # keep the last 12 messages
    headers = {"Authorization": f"Bearer {API_KEY}"}

    last_error = "no model configured"
    async with httpx.AsyncClient(timeout=120) as client:
        for model in MODELS:
            payload = {"model": model, "messages": messages, "temperature": 0.3}
            try:
                r = await client.post(f"{BASE_URL}/chat/completions", json=payload, headers=headers)
                r.raise_for_status()
                content = r.json()["choices"][0]["message"].get("content") or ""
                if content.strip():
                    print(f"Answered by: {model}")
                    return {"reply": content}
                last_error = f"{model}: empty reply"
            except httpx.HTTPStatusError as e:
                last_error = f"{model}: error {e.response.status_code}: {e.response.text[:200]}"
            except httpx.HTTPError:
                last_error = f"{model}: not reachable"
            print(f"Failed, trying next: {last_error}")

    raise HTTPException(status_code=502, detail=f"All models failed. Last error: {last_error}")