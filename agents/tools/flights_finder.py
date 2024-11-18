import os
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool
import requests


class FlightsInput(BaseModel):
    """
    Schema for flights finder input parameters.
    """
    departure_airport: Optional[str] = Field(description="Departure airport code (IATA).")
    arrival_airport: Optional[str] = Field(description="Arrival airport code (IATA).")
    outbound_date: Optional[str] = Field(description="Outbound date in YYYY-MM-DD format (e.g., '2024-11-19').")
    return_date: Optional[str] = Field(description="Return date in YYYY-MM-DD format (e.g., '2024-11-25').")
    adults: Optional[int] = Field(1, description="Number of adults. Default: 1.")
    children: Optional[int] = Field(0, description="Number of children. Default: 0.")
    infants_in_seat: Optional[int] = Field(0, description="Number of infants with their own seat. Default: 0.")
    infants_on_lap: Optional[int] = Field(0, description="Number of infants on lap. Default: 0.")
    currency: Optional[str] = Field("USD", description="Currency for prices. Default: USD.")
    travel_class: Optional[int] = Field(1, description="Travel class: 1 (Economy), 2 (Premium), 3 (Business), 4 (First). Default: Economy.")
    stops: Optional[str] = Field("0", description="Number of stops: 0 (Any), 1 (Nonstop), 2 (1 stop or fewer), 3 (2 stops or fewer). Default: Any.")
    max_price: Optional[int] = Field(None, description="Maximum ticket price in specified currency.")
    gl: Optional[str] = Field("us", description="Country localization (e.g., 'us' for United States). Default: 'us'.")
    hl: Optional[str] = Field("en", description="Language localization (e.g., 'en' for English). Default: 'en'.")


class FlightsInputSchema(BaseModel):
    """
    Wrapper for the FlightsInput schema.
    """
    params: FlightsInput


@tool(args_schema=FlightsInputSchema)
def flights_finder(params: FlightsInput):
    """
    Finds flights using the SerpApi Google Flights engine.

    Args:
        params (FlightsInput): Parameters for flight search.

    Returns:
        dict: List of flight options or an error message.
    """
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        return {"error": "SERPAPI_API_KEY is missing. Please set it in your .env file."}

    query_params = {
        "api_key": api_key,
        "engine": "google_flights",
        "departure_id": params.departure_airport,
        "arrival_id": params.arrival_airport,
        "outbound_date": params.outbound_date,
        "return_date": params.return_date,
        "currency": params.currency,
        "adults": params.adults or 1,
        "children": params.children or 0,
        "infants_in_seat": params.infants_in_seat or 0,
        "infants_on_lap": params.infants_on_lap or 0,
        "travel_class": params.travel_class or 1,
        "stops": params.stops or "0",
        "max_price": params.max_price,
        "gl": params.gl,
        "hl": params.hl,
    }

    # Log query for debugging purposes
    print(f"Querying SerpApi with parameters: {query_params}")

    try:
        response = requests.get("https://serpapi.com/search", params=query_params)
        response.raise_for_status()
        data = response.json()

        # Handle error response from SerpApi
        if data.get("search_metadata", {}).get("status") != "Success":
            return {"error": data.get("search_metadata", {}).get("error", "Unknown error occurred.")}

        # Extract flights
        flights = data.get("best_flights", []) or data.get("other_flights", [])
        if not flights:
            return {"message": "No flights found for the given criteria."}

        # Process and return flight data
        return [
            {
                "airline": flight["flights"][0].get("airline"),
                "price": flight.get("price"),
                "departure_time": flight["flights"][0]["departure_airport"].get("time"),
                "arrival_time": flight["flights"][0]["arrival_airport"].get("time"),
                "duration": flight["flights"][0].get("duration"),
                "stops": len(flight.get("layovers", [])),
                "booking_token": flight.get("booking_token"),
                "link": f"https://www.google.com/flights?token={flight.get('booking_token')}",
            }
            for flight in flights
        ]

    except requests.exceptions.RequestException as e:
        return {"error": f"An error occurred while communicating with the API: {str(e)}"}
    except KeyError as e:
        return {"error": f"Unexpected response format. Missing key: {str(e)}"}