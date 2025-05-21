import os
import logging
from dotenv import load_dotenv
from openai import OpenAI

# ログ設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 環境変数読み込み
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY が設定されていません。")
    raise SystemExit("環境変数エラー")

# OpenAIクライアント
client = OpenAI(api_key=OPENAI_API_KEY)