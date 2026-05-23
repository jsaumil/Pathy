from langchain_ollama.chat_models import ChatOllama

llm = ChatOllama(
        base_url="https://e685-35-227-50-23.ngrok-free.app",
        model="glm-4.7-flash:latest",
        think = False
    )

result = llm.invoke("what is the capital of italy?")
print(result)
