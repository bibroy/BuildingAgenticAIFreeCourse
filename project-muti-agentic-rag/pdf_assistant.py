import typer
import os
from phi.model.groq import Groq
from phi.agent import Agent
from typing import List, Optional
from phi.assistant import Assistant
from phi.storage.assistant.postgres import PgAssistantStorage
from phi.storage.agent.postgres import PgAgentStorage
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.pgvector import PgVector2

from dotenv import load_dotenv
load_dotenv(override=True)
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=PgVector2(collection="reci", db_url=db_url),
)

try:
    knowledge_base.load(recreate=True)
except Exception as e:
    print(f"Error loading knowledge base: {e}")

storage = PgAgentStorage(table_name="pdf_assistant", db_url=db_url)

def pdf_assistant(new: bool = False, user: str = "user"):
    run_id: Optional[str] = None
    
    # if not new:
    #     existing_run_ids: List[str] = storage.get_run_ids(user)
    #     if len(existing_run_ids) > 0:
    #         run_id = existing_run_ids[0]

    assistant = Agent(
        run_id=run_id,
        user_id=user,
        knowledge_base=knowledge_base,
        storage=storage,
        show_tool_calls=True,
        search_knowledge=True,
        read_chat_history=True,
    )

    if run_id is None:
        run_id = assistant.run_id
        print(f"Started Run: {run_id}\n")
    else:
        print(f"Continuing Run: {run_id}\n")

    assistant.cli_app(markdown=True)

if __name__ == "__main__":
    typer.run(pdf_assistant)