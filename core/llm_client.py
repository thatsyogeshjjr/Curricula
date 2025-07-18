from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
import os

class LLMClient:
    _instance = None

    @staticmethod
    def get_llm(model: str = "deepseek/deepseek-r1:free", temperature: float = 0.7):
        if "DEEPSEEK_API_KEY" in os.environ:
            del os.environ["DEEPSEEK_API_KEY"]
        load_dotenv()
        if not os.getenv("DEEPSEEK_API_KEY"):
            print("Can't find api key")
            return
        if not LLMClient._instance:
            # print(os.getenv("DEEPSEEK_API_KEY"))
            LLMClient._instance = ChatOpenAI(
                openai_api_base="https://openrouter.ai/api/v1",
                
                openai_api_key=os.getenv('DEEPSEEK_API_KEY'),
                model=model,
                temperature=temperature,
            )
        return LLMClient._instance