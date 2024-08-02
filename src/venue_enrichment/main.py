import os
import time
from dotenv import load_dotenv
from scrapegraphai import telemetry
from scrapegraphai.utils import prettify_exec_info
from scrapegraphai.graphs import SearchGraph
from pydantic import BaseModel, Field
from typing import List
import nest_asyncio
import json
from openai import OpenAIError, RateLimitError

nest_asyncio.apply()
telemetry.disable_telemetry()
load_dotenv()

gemini_key = os.environ["GOOGLE_API_KEY"]
openai_key = os.environ["OPENAI_API_KEY"]

completion_token_limit = 1500
total_token_limit = 10900
token_usage = 0
max_retries = 3
retry_count = 0
backoff_factor = 2
initial_wait_time = 10

graph_config = {
    "llm": {
        "api_key": openai_key,
        "model": "gpt-4-turbo",
        "max_tokens": completion_token_limit,
    },
    "max_results": 5,
    "headless": False,
}

venue_name = "The Heights Theatre"
location = "Houston, Texas"
prompt = f"Tell me everything about the venue {venue_name} in {location}"

class Venue(BaseModel):
    name: str = Field(description="the name of the venue")
    description: str = Field(description="a brief description of the venue")
    email: str = Field(description="the email of the venue")
    phone: str = Field(description="the phone number of the venue")
    website: str = Field(description="the website of the venue")
    facebookUrl: str = Field(description="the facebook page of the venue")
    capacity: str = Field(description="the capacity of the venue")
    genres: List[str] = Field(description="the genres they normally book")
    address: str = Field(description="the address of the venue")
    twitterUrl: str = Field(description="the twitter url of the venue")
    instagramUrl: str = Field(description="the instagram url of the venue")
    logoUrl: str = Field(description="the url of the venue's logo or avatar")
    idealPerformerProfile: str = Field(description="what kind of musicians does the venue normally book")

def run_search_graph():
    global token_usage, retry_count

    start_time = time.time()
    while retry_count < max_retries:
        elapsed_time = time.time() - start_time
        if elapsed_time > 60:
            print("Exceeded maximum time of 60 seconds. Exiting.")
            break

        try:
            remaining_tokens = total_token_limit - token_usage
            max_tokens = min(completion_token_limit, remaining_tokens)
            graph_config["llm"]["max_tokens"] = max_tokens

            search_graph = SearchGraph(
                prompt=prompt,
                config=graph_config,
                schema=Venue
            )

            result = search_graph.run()
            print("Result:", result)

            token_usage += max_tokens

            optimized_result = {
                "name": result.get("name", ""),
                "description": result.get("description", ""),
                "email": result.get("email", ""),
                "phone": result.get("phone", ""),
                "website": result.get("website", ""),
                "facebookUrl": result.get("facebookUrl", ""),
                "twitterUrl": result.get("twitterUrl", ""),
                "instagramUrl": result.get("instagramUrl", ""),
                "logoUrl": result.get("logoUrl", ""),
                "idealPerformerProfile": result.get("idealPerformerProfile", "")
            }

            graph_exec_info = search_graph.get_execution_info()
            print(prettify_exec_info(graph_exec_info))

            with open("search_g.json", "w") as outfile:
                json.dump(optimized_result, outfile)

            break
        except RateLimitError as e:
            retry_count += 1
            wait_time = initial_wait_time * (backoff_factor ** retry_count)
            print(f"Rate limit exceeded: {e}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            token_usage = 0
        except OpenAIError as e:
            print(f"An OpenAI error occurred: {e}")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break
    else:
        print("Max retries exceeded. Exiting.")

if __name__ == "__main__":
    run_search_graph()
