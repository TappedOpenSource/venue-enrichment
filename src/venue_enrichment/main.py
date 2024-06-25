
import os
from dotenv import load_dotenv
from scrapegraphai import telemetry
from scrapegraphai.utils import prettify_exec_info, convert_to_json
from scrapegraphai.graphs import SearchGraph, OmniSearchGraph
from pydantic import BaseModel, Field
from typing import List
import nest_asyncio

nest_asyncio.apply()
telemetry.disable_telemetry()
load_dotenv()

gemini_key = os.environ["GOOGLE_API_KEY"]
openai_key = os.environ["OPENAI_API_KEY"]

graph_config = {
   "llm": {
      "api_key": openai_key,
      "model": "gpt-4-turbo",
   },
   "max_results": 10,
   # "verbose": True,
}

venue_name = "The Anthem"
location = "Washington DC"
prompt = f"tell me everything about the venue {venue_name} in {location}"

class Venue(BaseModel):
    name: str = Field(description="the name of the venue")
    description: str = Field(description="a brief description of the venue")
    # capacity: int = Field(description="the capacity of the venue")
    # genres: List[str] = Field(description="the genres they normally book")
    # address: str = Field(description="the address of the venue")
    email: str = Field(description="the email of the venue")
    phone: str = Field(description="the phone number of the venue")
    website: str = Field(description="the website of the venue")
    facebookUrl: str = Field(description="the facebook page of the venue")
    twitterUrl: str = Field(description="the twitter url of the venue")
    instagramUrl: str = Field(description="the instagram url of the venue")
    logoUrl: str = Field(description="the url of the venue's logo or avatar")
    idealPerformerProfile: str = Field(description="what kind of musicians does the venue normally book")

def run_search_graph():
    search_graph = SearchGraph(
        prompt=prompt,
        config=graph_config,
        schema=Venue
    )

    result = search_graph.run()

    graph_exec_info = search_graph.get_execution_info()
    print(prettify_exec_info(graph_exec_info))

    # print(result)
    convert_to_json(result, "search_g.json")

def run_omni_search_graph():
    omni_search_graph = OmniSearchGraph(
        prompt=prompt,
        config=graph_config,
        schema=Venue
    )

    result = omni_search_graph.run()

    graph_exec_info = omni_search_graph.get_execution_info()
    print(prettify_exec_info(graph_exec_info))

    convert_to_json(result, "omni.json")

if __name__ == "__main__":
    run_search_graph()
    # run_omni_search_graph()
