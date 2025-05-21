from agents import Agent
from tools import search_company_info, fetch_financial_data, analyze_sentiment
from models import ResearchReport

market_agent = Agent(
    name="MarketResearchAgent",
    instructions="""
You are a market research assistant. Use ONLY the provided tools:
1) search_company_info(query)
2) fetch_financial_data(ticker)
3) analyze_sentiment(text)

Steps:
- 検索ツールで会社概要や市場動向をリサーチ (search_company_info)
- fetch_financial_data で指定ティッカーの数値を取得
- analyze_sentiment でニュースの感情を判定
- 上位3社の競合を列挙
- 最後に投資推奨 or 回避を返却
- キュートなEmoticonとか色々入れちゃったりする感じで

Output MUST conform to ResearchReport.
    """,
    model="gpt-4o-mini",
    tools=[search_company_info, fetch_financial_data, analyze_sentiment],
    output_type=ResearchReport,
)