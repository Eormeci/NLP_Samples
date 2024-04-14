from dotenv import load_dotenv
from langchain import HuggingFaceHub,LLMChain
from langchain.prompts import PromptTemplate

load_dotenv()

hub_llm = HuggingFaceHub(repo_id="mrm8488/t5-base-finetuned-wikiSQL-sql-to-en")

prompt = PromptTemplate(
    input_variables=["question"],
    template="Translate English to SQL :{question}"
    )

hub_chain = LLMChain(prompt=prompt,llm =hub_llm,verbose=True)
print(hub_chain.run("what is average of respondents using mobile data"))

 