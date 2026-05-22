from langchain_ollama.chat_models import ChatOllama
from dotenv import load_dotenv
import os
import json
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage, SystemMessage

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

async def main():
    client = MultiServerMCPClient(SERVERS)
    tools = await client.get_tools()

    tools_map = {tool.name: tool for tool in tools}
    print("Available tools:", list(tools_map.keys()))

    llm = ChatOllama(
        base_url="https://acf5-34-125-248-107.ngrok-free.app",
        model="qwen2.5:14b",
        think = False
    )

    llm_with_tools = llm.bind_tools(tools)

    messages = [HumanMessage(content="make a file name a.txt")]

    result = await llm_with_tools.ainvoke([HumanMessage(content="make a file name a.txt")])

    messages= [HumanMessage(content="create a file name a.txt use mcp file system"), result]

    if not getattr(result, "tool_calls", None):
        print("No tool calls made.", result)
        return result
    
    while True:
        result = await llm_with_tools.ainvoke(messages)
        messages.append(result)

        if not result.tool_calls:
            print("No more tool calls made.", result)
            break

        for tool_call in result.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            print(f"Invoking tool: {tool_name} with args: {tool_args}")

            tool = tools_map[tool_name]
            tool_result = await tool.ainvoke(tool_args)

            print(f"Tool result: {tool_result}")

            messages.append(ToolMessage(content=tool_result, tool_name_id=tool_call["id"]))

    return result

if __name__ == "__main__":
    load_dotenv()
    import asyncio
    asyncio.run(main())
