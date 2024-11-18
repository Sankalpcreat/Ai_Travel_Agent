import datetime
import operator
import os
import requests
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from resend import Resend  

from agents.tools.flights_finder import flights_finder
from agents.tools.hotels_finder import hotels_finder
from agents.tools.attractions_finder import attractions_finder  # Import the new tool

_ = load_dotenv()

CURRENT_YEAR = datetime.datetime.now().year


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


TOOLS_SYSTEM_PROMPT = f"""You are a smart travel agency. Use the tools to look up information.
    You are allowed to make multiple calls (either together or in sequence).
    Only look up information when you are sure of what you want.
    The current year is {CURRENT_YEAR}.
    If you need to look up some information before asking a follow-up question, you are allowed to do that!
    I want to have in your output links to hotels' websites and flights' websites (if possible).
    I also want to include the logo of the hotel and the airline company (if possible).
    In your output always include the price of the flight and the price of the hotel and the currency as well (if possible).
    For attractions, include the name, type, and distance from the provided location.
    """

TOOLS = [flights_finder, hotels_finder, attractions_finder]

EMAILS_SYSTEM_PROMPT = """Your task is to convert structured markdown-like text into a valid HTML email body.

- Do not include a ```html preamble in your response.
- The output should be in proper HTML format, ready to be used as the body of an email.
Here is an example:
<example>
Input:

I want to travel to New York from Madrid from October 1-7. Find me flights and 4-star hotels.

Expected Output:

<!DOCTYPE html>
<html>
<head>
    <title>Flight and Hotel Options</title>
</head>
<body>
    <h2>Flights from Madrid to New York</h2>
    <ol>
        <li>
            <strong>American Airlines</strong><br>
            <strong>Departure:</strong> Adolfo Suárez Madrid–Barajas Airport (MAD) at 10:25 AM<br>
            <strong>Arrival:</strong> John F. Kennedy International Airport (JFK) at 12:25 PM<br>
            <strong>Duration:</strong> 8 hours<br>
            <strong>Aircraft:</strong> Boeing 777<br>
            <strong>Class:</strong> Economy<br>
            <strong>Price:</strong> $702<br>
            <img src="https://www.gstatic.com/flights/airline_logos/70px/AA.png" alt="American Airlines"><br>
            <a href="https://www.google.com/flights">Book on Google Flights</a>
        </li>
    </ol>

    <h2>4-Star Hotels in New York</h2>
    <ol>
        <li>
            <strong>NobleDen Hotel</strong><br>
            <strong>Description:</strong> Modern, polished hotel offering sleek rooms, some with city-view balconies, plus free Wi-Fi.<br>
            <strong>Rate per Night:</strong> $537<br>
            <strong>Total Rate:</strong> $3,223<br>
            <img src="https://lh5.googleusercontent.com/p/AF1QipNDUrPJwBhc9ysDhc8LA822H1ZzapAVa-WDJ2d6=s287-w287-h192-n-k-no-v1" alt="NobleDen Hotel"><br>
            <a href="http://www.nobleden.com/">Visit Website</a>
        </li>
    </ol>
</body>
</html>

</example>
"""


class Agent:
    OLLAMA_URL = "http://localhost:11434/"  # Updated to the correct endpoint

    def __init__(self):
        self._tools = {t.name: t for t in TOOLS}

        builder = StateGraph(AgentState)
        builder.add_node("call_tools_llm", self.call_tools_llm)
        builder.add_node("invoke_tools", self.invoke_tools)
        builder.add_node("email_sender", self.email_sender)
        builder.set_entry_point("call_tools_llm")

        builder.add_conditional_edges(
            "call_tools_llm",
            Agent.exists_action,
            {"more_tools": "invoke_tools", "email_sender": "email_sender"},
        )
        builder.add_edge("invoke_tools", "call_tools_llm")
        builder.add_edge("email_sender", END)
        memory = MemorySaver()
        self.graph = builder.compile(checkpointer=memory, interrupt_before=["email_sender"])

        print(self.graph.get_graph().draw_mermaid())

    @staticmethod
    def exists_action(state: AgentState):
        result = state["messages"][-1]
        if len(result.tool_calls) == 0:
            return "email_sender"
        return "more_tools"

    def email_sender(self, state: AgentState):
        """
        Send an email using Resend API.
        """
        print("Sending email using Resend")
        email_response = self.invoke_llama(
            EMAILS_SYSTEM_PROMPT + "\n\n" + state["messages"][-1].content
        )
        print("Email content:", email_response)

        # Initialize Resend client
        resend = Resend(api_key=os.getenv("RESEND_API_KEY"))

        try:
            # Send the email
            response = resend.emails.send(
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
        response = self.invoke_llama(input_prompt)
        return {"messages": [SystemMessage(content=response)]}

    def invoke_tools(self, state: AgentState):
        tool_calls = state["messages"][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            if t["name"] not in self._tools:  # Check for bad tool name from LLM
                print("\n ....bad tool name....")
                result = "bad tool name, retry"  # Instruct LLM to retry if bad
            else:
                result = self._tools[t["name"]].invoke(t["args"])
            results.append(
                ToolMessage(tool_call_id=t["id"], name=t["name"], content=str(result))
            )
        print("Back to the model!")
        return {"messages": results}

    def invoke_llama(self, prompt: str) -> str:
        """
        Send a prompt to the locally hosted Llama 3.2 API and return the response.
        """
        headers = {"Content-Type": "application/json"}
        payload = {"model": "llama3.2:latest", "prompt": prompt}
        try:
            response = requests.post(self.OLLAMA_URL, json=payload, headers=headers)
            response.raise_for_status()  # Raise an exception for HTTP errors
            return response.json()["response"]  # Update to match the response format
        except requests.exceptions.RequestException as e:
            print(f"Error communicating with Llama: {e}")
            return "Error communicating with the language model."