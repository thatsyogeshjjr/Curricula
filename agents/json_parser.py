import os
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain
from core.llm_client import LLMClient

class ParserAgent:
    def __init__(self):
        # Load prompt template from file
        with open("prompts/parser.txt", "r") as file:
            self.template = file.read() 
        
        # Load prompt template from file
        with open("prompts/parser_user.txt", "r") as file:
            self.user_template = file.read() 
        
        self.systemPrompt = SystemMessagePromptTemplate.from_template(self.template)

        self.userPrompt = HumanMessagePromptTemplate.from_template(
            "Plan this out in weekly format and give a json response\n{schedule}"
        )


        self.llm = LLMClient.get_llm()

        # self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

        self.chain = ChatPromptTemplate.from_messages([self.systemPrompt, self.userPrompt]) | self.llm

    def run(self, schedule: str) -> str:
        return self.chain.invoke({
            "schedule": schedule
        })
