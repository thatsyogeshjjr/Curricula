import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from core.llm_client import LLMClient

class PlannerAgent:
    def __init__(self):
        # Load prompt template from file
        with open("prompts/planner.txt", "r") as file:
            self.template = file.read()

        self.prompt = PromptTemplate(
            input_variables=["goal", "duration", "hours_per_week", "skills"],
            template=self.template
        )

        self.llm = LLMClient.get_llm()

        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def run(self, goal: str, duration: str, hours_per_week: int, skills: str) -> str:
        return self.chain.run({
            "goal": goal,
            "duration": duration,
            "hours_per_week": hours_per_week,
            "skills": skills
        })
