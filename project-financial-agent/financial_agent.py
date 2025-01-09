from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv, find_dotenv
# from utils import dotenv_utils
import os
# import sys
# Get the absolute path of the parent directory of the current file
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
# # Add the parent directory to sys.path
# sys.path.insert(0, parent_dir)
# # loading environment variables
# dotenv_utils()

dotenv_path = find_dotenv()
if dotenv_path: 
    load_result = load_dotenv(dotenv_path,verbose=True, override=True)
    print(f"Dotenv Loaded?: {load_result}, Dotenv path: {dotenv_path}")
    # openai_key = os.getenv("OPENAI_API_KEY")
    # print(f"OpenAI key: {openai_key}")
    groq_key = os.getenv("GROQ_API_KEY")
    print(f"Groq key: {groq_key}")
    phi_key = os.getenv("PHI_API_KEY")
    print(f"Phi key: {phi_key}")
else:
    print("No dotenv file found")

    
# Web search agent
web_search_agent = Agent(
    name='web_search_agent',
    role="search the web for information",
    model=Groq(id= "llama-3.1-70b-versatile"),
    tools = [DuckDuckGo()],
    instructions= ["Always include the source of the information in the answer"],
    show_tool_calls=True,
    markdown= True,
    )

# Financial agent
financial_agent = Agent(
    name='financial_agent',
    #role="provide financial information",
    model=Groq(id= "llama-3.1-70b-versatile"),
    tools = [YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, 
                           company_news=True)],
    instructions= ["Use table to display the data"],
    show_tool_calls=True,
    markdown= True,
    )

multi_ai_agent = Agent(
    model=Groq(id= "llama-3.1-70b-versatile"), 
    team=[web_search_agent, financial_agent],
    instructions= ["Always include the source of the information in the answer, Use table to display the data "],
    show_tool_calls=True,
    markdown= True,
    )

multi_ai_agent.print_response("Summarize analyst recommendations and share the latest news for AAPL")