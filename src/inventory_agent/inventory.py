

import asyncio
from dotenv import load_dotenv
import os
from agents import Agent, OpenAIChatCompletionsModel, RunConfig, Runner, function_tool
from openai import AsyncOpenAI


load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")


inventory = ["Paracetamol", "Aspirin", "Ibuprofen"]

@function_tool
def add_item(item: str) -> str:
    """Adds a new pharmaceutical item to the inventory."""
    inventory.append(item)
    return f"Added: {item}"

@function_tool
def delete_item(item: str) -> str:
    """Deletes a pharmaceutical item from the inventory."""
    if item in inventory:
        inventory.remove(item)
        return f"Deleted: {item}"
    return f"{item} not found"

@function_tool
def update_item(old_item: str, new_item: str) -> str:
    """Updates an existing pharmaceutical item to a new one."""
    if old_item in inventory:
        idx = inventory.index(old_item)
        inventory[idx] = new_item
        return f"Updated: {old_item} -> {new_item}"
    return f"{old_item} not found"


external_client: AsyncOpenAI = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True,
)

pharma_agent = Agent(
    name="PharmaInventoryAgent",
    instructions="You are an assistant that manages a pharmaceutical inventory. Use the available tools to add, delete, or update items.",
    tools=[add_item, delete_item, update_item]
)

async def main():
    """Main asynchronous function to run the inventory agent."""
    print("Pharma Inventory Agent Started! (type 'exit' to quit)\n")
    print("Current Inventory:", inventory, "\n")

    while True:
        command = await asyncio.to_thread(input, "Enter command: ")

        if command.lower() in ["exit", "quit"]:
            print("\nFinal Inventory:", inventory)
            break

        result = await Runner.run(
            pharma_agent,
            input=command,
            run_config=config,
        )
        print(result.final_output)
        print("Current Inventory:", inventory)

def start():
    """A synchronous entry point for the pyproject.toml script."""
    asyncio.run(main())

if __name__ == "__main__":
    start()