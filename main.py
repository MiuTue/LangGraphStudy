from dotenv import load_dotenv

load_dotenv()

from graph.graph import app
if __name__ == "__main__":
    print("Hello Advanced Agentic RAG!")
    response = app.invoke(input={"question": "What is agent memory in AI systems?"})
    print("\n---FINAL RESPONSE---")
    print(response['generation'])