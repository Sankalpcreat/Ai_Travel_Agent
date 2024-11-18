import os
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool
import serpapi


class HotelsInput(BaseModel):
    q: str = Field(description="Location of the hotel (e.g., 'Paris, France').")
    check_in_date: str = Field(description="Check-in date. The format is YYYY-MM-DD (e.g., '2024-06-22').")
    check_out_date: str = Field(description="Check-out date. The format is YYYY-MM-DD (e.g., '2024-06-28').")
    sort_by: Optional[str] = Field("highest_rating", description="Sort by: 'highest_rating' or 'lowest_price'. Default: 'highest_rating'.")
    adults: Optional[int] = Field(1, description="Number of adults. Default: 1.")
    children: Optional[int] = Field(0, description="Number of children. Default: 0.")
    rooms: Optional[int] = Field(1, description="Number of rooms. Default: 1.")
    hotel_class: Optional[str] = Field(None, description="Filter by hotel class (e.g., '2', '3', '4' stars). Default: None.")


class HotelsInputSchema(BaseModel):
    params: HotelsInput


@tool(args_schema=HotelsInputSchema)
def hotels_finder(params: HotelsInput):
    """
    Find hotels using the Google Hotels engine via SerpAPI.

    Args:
        params (HotelsInput): Parameters for hotel search, including location, check-in/out dates, and preferences.

    Returns:
        list: A list of hotel options with details like name, price, rating, and booking link.
    """
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        return {"error": "SERPAPI_API_KEY is missing. Please set it in your .env file."}

    query_params = {
        "api_key": api_key,
        "engine": "google_hotels",
        "hl": "en",
        "gl": "us",
        "q": params.q,
        "check_in_date": params.check_in_date,
        "check_out_date": params.check_out_date,
        "currency": "USD",
        "adults": params.adults,
        "children": params.children,
        "rooms": params.rooms,
        "sort_by": params.sort_by,
        "hotel_class": params.hotel_class,
    }

    try:
        search = serpapi.search(query_params)
        results = search.data

        if "properties" not in results:
            return {"message": "No hotels found for the given criteria."}

        return [
            {
                "name": hotel.get("title"),
                "price": hotel.get("price"),
                "rating": hotel.get("rating"),
                "reviews_count": hotel.get("reviews_count"),
                "address": hotel.get("address"),
                "url": hotel.get("link"),
            }
            for hotel in results["properties"][:5]
        ]

    except Exception as e:
        return {"error": f"An error occurred while fetching hotel data:{str(e)}"}