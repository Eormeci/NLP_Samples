from dotenv import load_dotenv
from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

load_dotenv()

hub_llm = HuggingFaceHub(repo_id="google/gemma-1.1-7b-it")

prompt = PromptTemplate(
    input_variables=["text-generation"],
    template="Text generation :{text-generation}"
    )

hub_chain = LLMChain(prompt=prompt,llm =hub_llm,verbose=True)
print(hub_chain.run("How does the brain work ? "))

 