from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")

template = """
You are an expert in student health and well-being. 
You are interested in understanding what makes students test well, 
and what habits and experiences are detrimental to their test scores.

Students were given a survey that asked for basic demographic information,
study habits, eating habits, their activity level, and other factors, along
with their test scores.

Here are the results of the survey: {survey_results}

You will be asked to analyze the survey results, summarize overall scores, and 
the main factors that correlate with high and low test scores. 

Here is the question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model



while True:
    print("\n\n--------------------------------")
    question = input("Ask your question (or type 'exit' to quit): ")
    print("\n\n")
    if question.lower() == "exit":
        break
    scores = retriever.invoke(question)
    result = chain.invoke({"survey_results": scores, "question": question})

    print(result)