from dotenv import load_dotenv
from langchain import HuggingFaceHub,LLMChain
from langchain.prompts import PromptTemplate

load_dotenv()

hub_llm = HuggingFaceHub(repo_id="facebook/bart-large-cnn")

a = """
    Jams, also known as preserves or fruit spreads, a
    re sweet spreads made from fruit and sugar. The process 
    of making jam involves cooking fruit with sugar to create a thick,
    flavorful spread that can be enjoyed on toast, pastries, or as a topping for yogurt and ice cream. 
    Common fruits used in jam-making include strawberries, raspberries, apricots, and peaches, although 
    virtually any fruit can be used. Some jams may also include additional ingredients such as pectin, a 
    natural thickening agent, or lemon juice for added tartness. Jams are often enjoyed as a breakfast staple
    or as a sweet treat throughout the day, and they can be found in a variety of flavors and textures to suit
    different tastes and preferences.
"""

prompt = PromptTemplate(
    input_variables=["summarization"],
    template="Summarize the text :{summarization}"
    )

hub_chain = LLMChain(prompt=prompt,llm =hub_llm,verbose=True)
print(hub_chain.run(a))

 