import asyncio
from agents import Runner
from agent_setup import market_agent
from tools import fetch_multiple_financials
from models import ResearchReport

async def main():
    target = input("調査したい会社名を入力してね: ")
    result = await Runner.run(market_agent, [{"role": "user", "content": target}], max_turns=20)
    report: ResearchReport = result.final_output_as(ResearchReport)

    try:
        tickers = report.competitors  # ["AAPL", "MSFT", ...]
        extra_financials = await fetch_multiple_financials(tickers)
    except Exception:
        extra_financials = {}

    print("\n=== Market Research Report ===")
    print(report.model_dump_json(indent=2))

if __name__ == "__main__":
    asyncio.run(main())