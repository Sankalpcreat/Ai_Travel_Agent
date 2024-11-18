import os
import requests
import datetime
import operator
import json
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from resend import Emails

from agents.tools.flights_finder import flights_finder
from agents.tools.hotels_finder import hotels_finder
from agents.tools.attractions_finder import attractions_finder

_ = load_dotenv()

CURRENT_YEAR = datetime.datetime.now().year


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


TOOLS_SYSTEM_PROMPT = f"""You are a smart travel agency. Use the tools to look up information.
You are allowed to make multiple calls (either together or in sequence).
Only look up information when you are sure of what you want.
The current year is {CURRENT_YEAR}.
Always include in your output:
- Links to hotel and flight booking websites (if available).
- The logo of the hotel and airline company (if available).
- Prices and currency details for flights and hotels.
- Details of attractions, including their name, type, and distance from the given location.
"""

TOOLS = {
    "flights_finder": flights_finder,
    "hotels_finder": hotels_finder,
    "attractions_finder": attractions_finder,
}

EMAILS_SYSTEM_PROMPT = """Your task is to convert structured markdown-like text into a valid HTML email body.

- The output should be in proper HTML format, ready to be used as the body of an email.
- Include the information from flight, hotel, and attraction searches.

Example:
<!DOCTYPE html>
<html>
<head>
    <title>Travel Information</title>
</head>
<body>
    <h2>Flight Details</h2>
    <!-- Include flight details -->
    <h2>Hotel Details</h2>
    <!-- Include hotel details -->
    <h2>Attraction Details</h2>
    <!-- Include attraction details -->
</body>
</html>
"""


class Agent:
    OLLAMA_URL = "http://localhost:11434/"

    def __init__(self):
        self._tools = TOOLS

        builder = StateGraph(AgentState)
        builder.add_node("call_tools_llm", self.call_tools_llm)
        builder.add_node("email_sender", self.email_sender)
        builder.set_entry_point("call_tools_llm")

        builder.add_edge("call_tools_llm", "email_sender")
        builder.add_edge("email_sender", END)
        memory = MemorySaver()
        self.graph = builder.compile(checkpointer=memory, interrupt_before=["email_sender"])

        print(self.graph.get_graph().draw_mermaid())

    def email_sender(self, state: AgentState):
        print("Sending email using Resend")
        try:
            email_response = self.invoke_llama(
                EMAILS_SYSTEM_PROMPT + "\n\n" + state["messages"][-1].content
            )
            print("Email content:", email_response)

            response = Emails.send(
                api_key=os.getenv("RESEND_API_KEY"),
                from_email=os.getenv("FROM_EMAIL"),
                to=[os.getenv("TO_EMAIL")],
                subject=os.getenv("EMAIL_SUBJECT"),
                html=email_response,
            )
            print("Email sent successfully:", response)
        except Exception as e:
            print(f"Error sending email with Resend: {e}")

    def call_tools_llm(self, state: AgentState):
        messages = state["messages"]
        input_prompt = TOOLS_SYSTEM_PROMPT + "\n\n" + "\n".join(
            [msg.content for msg in messages]
        )
        print(f"Input Prompt to LLM: {input_prompt}")
        response = self.invoke_llama(input_prompt)

        print(f"Raw LLM response: {response}")
        # Since the LLM returns plain text, we can directly create a SystemMessage
        return {"messages": [SystemMessage(content=response)]}

    def invoke_llama(self, prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        payload = {"model": "llama3.2:latest", "prompt": prompt}

        try:
            response = requests.post(
                self.OLLAMA_URL + "api/generate",
                json=payload,
                headers=headers,
                stream=True,
            )
            response.raise_for_status()

            full_response = ''
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode("utf-8")
                    try:
                        json_data = json.loads(decoded_line)
                        full_response += json_data.get('response', '')
                        if json_data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        print(f"JSON decode error for line: {decoded_line}")
                        continue
            return full_response.strip()

        except requests.exceptions.RequestException as e:
            print(f"Error communicating with Llama: {e}")
            return "Error communicating with the language model."
