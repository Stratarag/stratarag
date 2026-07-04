"""Tools, async runs, and streaming."""
import asyncio
import stratarag as mn
from stratarag.llm.echo import EchoProvider


@mn.tool
def get_order_status(order_id: str) -> str:
    """Look up an order's shipping status."""
    return f"Order {order_id} shipped yesterday and arrives Friday."


provider = EchoProvider(script=[
    {"tool": "get_order_status", "args": {"order_id": "A-1042"}},
    "Good news: order A-1042 shipped yesterday and arrives Friday.",
])
agent = mn.Agent(model=provider, tools=[get_order_status])

# streaming: tokens + tool events + final result
for event in agent.stream("Where is my order A-1042?"):
    if event["type"] == "tool":
        print(f"\n[tool {event['name']}] {event['result']}")
    elif event["type"] == "token":
        print(event["text"], end="", flush=True)
    else:
        print("\n--\nconfidence:", event["result"].confidence)

# async
async def main():
    res = await agent.arun("thanks!")
    print("async:", res.output)

asyncio.run(main())
