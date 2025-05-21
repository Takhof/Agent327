import json
import os
import asyncio
from pydantic import BaseModel
from agents import Agent, AgentOutputSchema, Runner, WebSearchTool, function_tool, ItemHelpers
from openai import OpenAI, RateLimitError
from dotenv import load_dotenv
from typing import Dict, TypedDict, List
import yfinance as yf
from agents import function_tool


# Load environment
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# ====== Tool Functions ======
@function_tool
def search_company_info(query: str) -> str:
    resp = client.responses.create(
        model="gpt-4.1",
        tools=[{"type": "web_search_preview"}],
        input=query
    )
    return resp.output_text


@function_tool
def fetch_financial_data(ticker: str) -> dict:
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "price": info.get("currentPrice"),
        "market_cap": info.get("marketCap")
    }

@function_tool
def analyze_sentiment(text: str) -> str:
    """Analyze sentiment of provided text with OpenAI."""
    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": f"Analyze sentiment for this text and return POSITIVE, NEGATIVE, or NEUTRAL:\n\n{text}"}
            ],
            max_tokens=15
        )
        print(resp.choices[0].message.content.strip())
        return resp.choices[0].message.content.strip()
    except RateLimitError:
        return "ERROR: Rate limit exceeded"


class Financials(BaseModel):
    price: float
    market_cap: float
class ResearchReport(BaseModel):
    company: str
    summary: str
    financials: Financials
    sentiment: str
    competitors: list[str]
    recommendations: str


# ====== Market Research Agent ======
market_agent = Agent(
    name="MarketResearchAgent",
    instructions="""
    You are a market research assistant. Given a target company or industry, you will:
    1. Search for company background and market trends using only the first page.
    2. Fetch basic financial metrics for provided tickers.
    3. Analyze overall sentiment from recent news snippets.
    4. Identify top 3 competitors.
    5. Recommend either investing or staying away from the company/industry. At the end of the recommendation, add a cute emoticon.
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
    result = await Runner.run(market_agent, items, max_turns=11)
    report: ResearchReport = result.final_output_as(ResearchReport)
    print(ResearchReport)
    print("\n=== Market Research Report ===")
    print(f"Company/Industry: {report.company}\n")
    print(f"Summary:\n{report.summary}\n")
    print(f"Financials: {report.financials}\n")
    print(f"Sentiment: {report.sentiment}\n")
    print(f"Competitors: {', '.join(report.competitors)}\n")
    print(f"Recommendations: {report.recommendations}")

if __name__ == "__main__":
    asyncio.run(main())

