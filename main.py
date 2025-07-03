from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from core.llm_client import LLMClient
from agents.planner import PlannerAgent
from agents.deduper import DeduperAgent
from agents.json_parser import ParserAgent
# from testing_data import planner_response, deduper_response
from dotenv import load_dotenv
import os


def main():
    load_dotenv()
    # Set up LLM
    llm = ChatOpenAI(
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
        model="deepseek/deepseek-r1-0528:free",
        temperature=0.7,
        
    )

    # Set up PlannerAgent
    planner_agent = PlannerAgent()
    deduper_agent = DeduperAgent()
    parser_agent = ParserAgent()

    # Test it with a simple message
    # response = LLMClient.get_llm().invoke([ HumanMessage(content="Give me a study plan to learn Python in 2 weeks, structure should be a weekly goal list that person can use to track progress")])


    planner_response = planner_agent.run(
        goal="Learn Python",
        duration="2 weeks",
        hours_per_week=10,
        skills="basic programming, problem-solving"
    )

    print("✅ planner response generated")

    deduper_response = deduper_agent.run(
        schedule=planner_response
    )

    print("✅ deduper response generated") 

    parser_response = parser_agent.run(schedule=deduper_response)
    print(parser_response)


if __name__ == "__main__":
    main()
