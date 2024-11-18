import requests
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool


class AttractionsInput(BaseModel):
    location: str = Field(description="Location coordinates in 'latitude,longitude' format (e.g., '48.8566,2.3522').")
    radius: int = Field(1000, description="Search radius in meters. Default is 1000 (1 km).")
    category: Optional[str] = Field("tourism", description="Type of attractions (e.g., 'tourism', 'museum', 'park').")


class AttractionsInputSchema(BaseModel):
    params: AttractionsInput


@tool(args_schema=AttractionsInputSchema)
def attractions_finder(params: AttractionsInput):
    """
    Find nearby attractions using OpenStreetMap's Overpass API.

    Args:
        params (AttractionsInput): Parameters for finding attractions, including location coordinates, radius, and category.

    Returns:
        list: A list of attractions with details like name, type, latitude, longitude, and tags.
    """
    overpass_url = "https://overpass-api.de/api/interpreter"
    query = f"""
    [out:json];
    (
      node["{params.category}"](around:{params.radius},{params.location.split(',')[0]},{params.location.split(',')[1]});
      way["{params.category}"](around:{params.radius},{params.location.split(',')[0]},{params.location.split(',')[1]});
      relation["{params.category}"](around:{params.radius},{params.location.split(',')[0]},{params.location.split(',')[1]});
    );
    out center;
    """

    try:
        response = requests.get(overpass_url, params={"data": query})
        response.raise_for_status()
        data = response.json()

        results = []
        for element in data.get("elements", []):
            if "tags" in element:
                results.append({
                    "name": element["tags"].get("name", "Unnamed"),
                    "type": element["tags"].get(params.category, "Unknown"),
                    "lat": element.get("lat", element["center"]["lat"]),
                    "lon": element.get("lon", element["center"]["lon"]),
                    "details": element["tags"]
                })

        if not results:
            return {"message": "No attractions found for the given location and criteria."}

        return results

    except requests.exceptions.RequestException as e:
        return {"error": f"An error occurred while fetching attractions: {str(e)}"}