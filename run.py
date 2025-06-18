import argparse
import logging
from openbb_agents import agent
import os

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Query the OpenBB agent.")
parser.add_argument(
    "query", metavar="query", type=str, help="The query to send to the agent."
)
parser.add_argument(
    "-v", "--verbose", action="store_true", help="Enable verbose logging."
)
args = parser.parse_args()

# We only import after passing in command line args to have verbosity propagate.

query = args.query
logging.basicConfig(filename='myapp.log', level=logging.INFO)
logger.info('Started')
OPENBB_PAT = os.environ.get('OPENBB_PAT')
def google_search(query):
    """Search for Vietnamese queries using Google Search."""

    import requests
    params = {
    "query": query
    }
    r = requests.post('http://127.0.0.1:8000/search',params=params)
    return r.json()
result = agent.openbb_agent(query, openbb_pat=OPENBB_PAT, extra_tools=[google_search])
logger.info('Finished')

print("============")
print("Final Answer")
print("============")
print(result)
