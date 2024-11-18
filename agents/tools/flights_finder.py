import os
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool
import serpapi


class FlightsInput(BaseModel):
    departure_airport: Optional[str] = Field(description="Departure airport code (IATA).")
    arrival_airport: Optional[str] = Field(description="Arrival airport code (IATA).")
    outbound_date: Optional[str] = Field(description="Outbound date in YYYY-MM-DD format (e.g., '2024-06-22').")
    return_date: Optional[str] = Field(description="Return date in YYYY-MM-DD format (e.g., '2024-06-28').")
    adults: Optional[int] = Field(1, description="Number of adults. Default: 1.")
    children: Optional[int] = Field(0, description="Number of children. Default: 0.")
    infants_in_seat: Optional[int] = Field(0, description="Number of infants with their own seat. Default: 0.")
    infants_on_lap: Optional[int] = Field(0, description="Number of infants on lap. Default: 0.")


class FlightsInputSchema(BaseModel):
    params: FlightsInput


@tool(args_schema=FlightsInputSchema)
def flights_finder(params: FlightsInput):
    """
    Find flights using the Google Flights engine via SerpAPI.

    Args:
        params (FlightsInput): Parameters for flight search, including departure/arrival airports, dates, and passenger details.

    Returns:
        list: A list of flight options with details like airline, price, duration, and booking link.
    """
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        return {"error": "SERPAPI_API_KEY is missing. Please set it in your .env file."}

    query_params = {
        "api_key": api_key,
        "engine": "google_flights",
        "hl": "en",
        "gl": "us",
        "departure_id": params.departure_airport,
        "arrival_id": params.arrival_airport,
        "outbound_date": params.outbound_date,
        "return_date": params.return_date,
        "currency": "USD",
        "adults": params.adults,
        "children": params.children,
        "infants_in_seat": params.infants_in_seat,
        "infants_on_lap": params.infants_on_lap,
        "stops": "1",
    }

    try:
        search = serpapi.search(query_params)
        results = search.data.get("best_flights", [])
        if not results:
            return {"message": "No flights found for the given criteria."}

        return [
            {
                "airline": flight.get("airline"),
                "price": flight.get("price"),
                "departure_time": flight.get("departure_time"),
                "arrival_time": flight.get("arrival_time"),
                "duration": flight.get("duration"),
                "stops": flight.get("stops"),
                "link": flight.get("booking_link"),
            }
            for flight in results
        ]

    except Exception as e:
        return {"error": f"An error occurred while fetching flight data: {str(e)}"}