import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from core.llm_client import LLMClient

class DeduperAgent:
    def __init__(self):
        # Load prompt template from file
        with open("prompts/deduper.txt", "r") as file:
            self.template = file.read()

        self.prompt = PromptTemplate(
            input_variables=["schedule"],
            template=self.template
        )

        self.llm = LLMClient.get_llm()

        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def run(self, schedule: str) -> str:
        return self.chain.invoke({
            "schedule": schedule
        })
