from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_google_genai import GoogleGenerativeAI

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user to improve their tweet. Always provide detailed recommendations, including request for length, virality, style, etc."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter techie influencer assistant tasked with writing excelling twitter posts. Generate the best twitter post possible based on the user's input and the reflection provided. If the user provides critique, respond with a revised version of your previous attemps.TWEET: <your_tweet_here>"
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


llm = ChatOllama(model="llama3.1:8b", temperature=0.3)

generate_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm