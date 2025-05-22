import asyncio
import logging
import yfinance as yf
from openai import RateLimitError
from agents import function_tool
from config import client, logger

@function_tool
def search_company_info(query: str) -> str:
    try:
        resp = client.responses.create(
            model="gpt-4.1",
            tools=[{"type": "web_search_preview"}],
            input=query
        )
        return resp.output_text
    except Exception as e:
        logger.warning(f"[search_company_info] エラー: {e}")
        return "検索に失敗しました…"

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
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": f"Analyze sentiment:\n\n{text}"}],
                max_tokens=5
            )
            return resp.choices[0].message.content.strip().upper()
        except RateLimitError:
            logger.info("Rate limit でリトライ中…")
            asyncio.sleep(2 ** attempt)
    return "ERROR: Rate limit exceeded"

async def fetch_multiple_financials(tickers: list[str]) -> dict[str, dict]:
    unique_tickers = list(dict.fromkeys(tickers))
    tasks = [asyncio.to_thread(fetch_financial_data, ticker) for ticker in unique_tickers]
    results = await asyncio.gather(*tasks)
    return dict(zip(unique_tickers, results))