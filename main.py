import os
import asyncio
from pydantic import BaseModel
from agents import Agent, Runner, WebSearchTool, function_tool, ItemHelpers
from openai import OpenAI, RateLimitError
from dotenv import load_dotenv

# Load environment
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# ====== Tool Functions ======
@function_tool
def search_company_info(query: str) -> str:
    """Web search for company or market info, returns aggregated snippets."""
    # Use built-in WebSearchTool under the hood
    snippets = WebSearchTool().search(query)
    return "\n".join(snippets[:5])  # top 5 results

@function_tool
def fetch_financial_data(ticker: str) -> dict:
    """Fetch basic financial metrics for a given stock ticker."""
    data = client.finance.quote(
        ticker=ticker,
        type="equity",
        market="USA"
    )
    return {"price": data.price, "market_cap": data.market_cap}

@function_tool
def analyze_sentiment(text: str) -> str:
    """Analyze sentiment of provided text with OpenAI."""
    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": f"Analyze sentiment for this text and return POSITIVE, NEGATIVE, or NEUTRAL:\n\n{text}"}
            ],
            max_tokens=10
        )
        return resp.choices[0].message.content.strip()
    except RateLimitError:
        return "ERROR: Rate limit exceeded"

# ====== Pydantic Schema ======
class ResearchReport(BaseModel):
    company: str
    summary: str
    financials: dict
    sentiment: str
    competitors: list[str]

# ====== Market Research Agent ======
market_agent = Agent(
    name="MarketResearchAgent",
    instructions="""
    You are a market research assistant. Given a target company or industry, you will:
    1. Search for company background and market trends.
    2. Fetch basic financial metrics for provided tickers.
    3. Analyze overall sentiment from recent news snippets.
    4. Identify top 3 competitors.
    Provide the output as a ResearchReport object.
    """,
    model="gpt-4o-mini",
    tools=[search_company_info, fetch_financial_data, analyze_sentiment],
    output_type=ResearchReport,
)

# ====== Runner Entry Point ======
async def main():
    target = input("Enter company or industry to research: ")
    items = [{"role": "user", "content": target}]
    result = await Runner.run(market_agent, items)
    report: ResearchReport = result.final_output_as(ResearchReport)
    print("\n=== Market Research Report ===")
    print(f"Company/Industry: {report.company}\n")
    print(f"Summary:\n{report.summary}\n")
    print(f"Financials: {report.financials}\n")
    print(f"Sentiment: {report.sentiment}\n")
    print(f"Competitors: {', '.join(report.competitors)}\n")

if __name__ == "__main__":
    asyncio.run(main())
