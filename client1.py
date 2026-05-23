from langchain_ollama.chat_models import ChatOllama
from dotenv import load_dotenv
import os
import json
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage, AIMessage, SystemMessage
from langchain_protocol import Union
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

SERVERS = {
    "filesystem": {
      "transport": "stdio",
      "command": "npx",  # Uses npx to run the node package
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "E:\\SrootAI\\pathy"  # <--- CHANGE THIS to a real folder path!
      ]
    }
}

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

async def main():
    llm = ChatOllama(
        base_url="https://e685-35-227-50-23.ngrok-free.app",
        model="glm-4.7-flash:latest",
        think = False
    )
    client = MultiServerMCPClient(SERVERS)
    tools = await client.get_tools()

    # tools_map = {tool.name: tool for tool in tools}
    # print("Available tools:", list(tools_map.keys()))

    llm_with_tools = llm.bind_tools(tools)

    async def chat_node(state: ChatState):
        """LLM node that may answer or request a tool call."""
        messages = state["messages"]
        result = await llm_with_tools.ainvoke(messages)
        return {"messages": [result]}

    tool_node = ToolNode(tools) if tools else None

    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")
    graph.add_edge("chat_node", END)

    checkpointer = InMemorySaver()

    chatbot = graph.compile(checkpointer=checkpointer)

    config1 = {"configurable":{"thread_id":"1"}}
    result = await chatbot.ainvoke({"messages": [HumanMessage(content="What files do I have in E:\\SrootAI\\pathy?")]},config=config1)
    print("Final result:", result["messages"][-1].content)
    return result

if __name__ == "__main__":
    load_dotenv()
    import asyncio
    asyncio.run(main())
