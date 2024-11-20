import os
import requests
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool


class HotelsInput(BaseModel):
    q: str = Field(description="Location or query for the hotel search (e.g., 'Paris, France').")
    check_in_date: str = Field(description="Check-in date in YYYY-MM-DD format (e.g., '2024-11-19').")
    check_out_date: str = Field(description="Check-out date in YYYY-MM-DD format (e.g., '2024-11-25').")
    sort_by: Optional[str] = Field(None, description="Sort results: 'lowest_price', 'highest_rating', or 'most_reviewed'.")
    adults: Optional[int] = Field(2, description="Number of adults. Default: 2.")
    children: Optional[int] = Field(0, description="Number of children. Default: 0.")
    children_ages: Optional[str] = Field(None, description="Ages of children, separated by commas (e.g., '5,10'). Required if children > 0.")
    min_price: Optional[int] = Field(None, description="Minimum price in the currency of search.")
    max_price: Optional[int] = Field(None, description="Maximum price in the currency of search.")
    hotel_class: Optional[str] = Field(None, description="Hotel class filter (e.g., '2', '3', '4', '5').")
    free_cancellation: Optional[bool] = Field(False, description="Filter for free cancellation. Default: False.")
    special_offers: Optional[bool] = Field(False, description="Filter for special offers. Default: False.")
    eco_certified: Optional[bool] = Field(False, description="Filter for eco-certified hotels. Default: False.")
    currency: Optional[str] = Field("USD", description="Currency for price display. Default: 'USD'.")


class HotelsInputSchema(BaseModel):
    params: HotelsInput


@tool(args_schema=HotelsInputSchema)
def hotels_finder(params: HotelsInput):
    """
    Fetch hotel details using SerpApi's Google Hotels API.

    Args:
        params (HotelsInput): Parameters for the hotel search.

    Returns:
        list: A list of hotels with details like name, price, rating, address, and link.
    """
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key: 
        return {"error": "SERPAPI_API_KEY is missing. Please set it in your environment variables."}

    # Mapping sort_by values to SerpApi's expected values
    sort_by_mapping = {
        "lowest_price": "3",
        "highest_rating": "8",
        "most_reviewed": "13",
    }
    sort_by = sort_by_mapping.get(params.sort_by, None)

    query_params = {
        "engine": "google_hotels",
        "q": params.q,
        "check_in_date": params.check_in_date,
        "check_out_date": params.check_out_date,
        "adults": params.adults,
        "children": params.children,
        "currency": params.currency,
        "sort_by": sort_by,
        "min_price": params.min_price,
        "max_price": params.max_price,
        "hotel_class": params.hotel_class,
        "free_cancellation": str(params.free_cancellation).lower(),
        "special_offers": str(params.special_offers).lower(),
        "eco_certified": str(params.eco_certified).lower(),
        "children_ages": params.children_ages,
        "api_key": api_key,
    }

    # Remove None values from the query parameters
    query_params = {k: v for k, v in query_params.items() if v is not None}

    try:
        response = requests.get("https://serpapi.com/search", params=query_params)
        response.raise_for_status()
        results = response.json()

        if "properties" not in results:
            return {"message": "No hotels found for the given criteria."}

        return [
            {
                "name": property.get("name"),
                "description": property.get("description"),
                "address": property.get("address"),
                "price_per_night": property.get("rate_per_night", {}).get("lowest"),
                "total_price": property.get("total_rate", {}).get("lowest"),
                "rating": property.get("overall_rating"),
                "reviews_count": property.get("reviews"),
                "hotel_class": property.get("hotel_class"),
                "url": property.get("link"),
            }
            for property in results["properties"][:10]  # Fetch top 10 results
        ]

    except requests.exceptions.RequestException as e:
        return {"error": f"An error occurred while fetching hotel data: {str(e)}"}