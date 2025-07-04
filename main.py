from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
load_dotenv()
import os


api_key = os.getenv("AZURE_OPENAI_API_KEY")
tok_url = os.getenv("TOKEN_URL")
gr_type = os.getenv("GRANT_TYPE")
cl_id = os.getenv("CLIENT_ID")
cl_secret = os.getenv("CLIENT_SECRET")
graph_url = os.getenv("GRAPHQL_URL")
end_url = os.getenv("ENDPOINT_URL")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model 
class MemberRequest(BaseModel):
    enterpriseIndividualIdentifier: str
    firstName: str
    state: str

@app.post("/generate-summary")
def generate_summary(request: MemberRequest):
    # Step 1: Get OAuth Token
    token_url = tok_url
    payload = {
        "grant_type": gr_type,
        "client_id": cl_id,
        "client_secret": cl_secret
    }
    token_response = requests.post(token_url, data=payload)

    if token_response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to retrieve access token")
    access_token = token_response.json().get("access_token")

    # Step 2: GraphQL Query with dynamic input
    graphql_url = graph_url
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    query = f"""
    query {{
      getMemberSignals(memberSignalsRequest: {{
        enterpriseIndividualIdentifier: "{request.enterpriseIndividualIdentifier}",
        firstName: "{request.firstName}",
        state: "{request.state}"
      }}) {{
        Member {{
          enterpriseIndividualIdentifier
          dateOfBirth
          firstName
          lastName
          policyNbr
          state
          InsightsInfo {{
            insightsSourceName
            insightsCreateDate
            scenarioInfo {{
              scenarioRuleCd
              scenarioRuleDescription
              scenarioEffectiveDate
              scenarioEndDate
              scenarioFlag
              condition
            }}
          }}
        }}
      }}
    }}
    """

    graphql_response = requests.post(graphql_url, headers=headers, json={"query": query})
    if graphql_response.status_code != 200:
        raise HTTPException(status_code=500, detail="GraphQL query failed")

    members = graphql_response.json().get("data", {}).get("getMemberSignals", {}).get("Member", [])
    if not isinstance(members, list):
        members = [members]

    # Step 3: Process Data
    def format_string(key, value, indent="\t"):
        result = ""
        if isinstance(value, dict):
            for k, v in value.items():
                result += format_string(k, v, indent)
        elif isinstance(value, list):
            result += f"{indent}{key}\n"
            for item in value:
                result += format_string(key, item, indent)
        else:
            result += f"{indent}{key}: {value}\n"
        return result

    def process_data(data):
        member_demographic = [
            "id", "enterpriseindividualidentifier", "dateofbirth", "firstname", "lastname",
            "policynbr", "subscriberid", "state", "cipartitionkey", "optumsegmentid",
            "facetaccountids", "facetaccountid", "memberinfo"
        ]
        record_string = "Below records show the clinical conditions identified for a member...\n"
        mem_string = "Below is the member demographic information...\n"

        for key, value in data.items():
            if key.lower() in member_demographic:
                mem_string += format_string(key, value)
            elif key.lower() == "insightsinfo":
                for item in value:
                    insight_source_name = item.get("insightsSourceName", "")
                    insight_create_date = item.get("insightsCreateDate", "")
                    scenario_info = item.get("scenarioInfo", [])
                    for signal in scenario_info:
                        record_string += "- Record:\n"
                        record_string += f"\tinsightsSourceName: {insight_source_name}\n"
                        record_string += f"\tinsightsCreateDate: {insight_create_date}\n"
                        for k, v in signal.items():
                            record_string += format_string(k, v)
                        record_string += "\n"
            else:
                record_string += format_string(key, value)
        return mem_string, record_string

    def process_entries_from_api(data):
        results = []
        for entry in data:
            mem_string, record_string = process_data(entry)
            id = entry.get("enterpriseIndividualIdentifier", "unknown")
            results.append({
                "id": str(id),
                "member_demographic": mem_string,
                "content": record_string
            })
        return results

    results = process_entries_from_api(members)

    # Step 4: Call Azure OpenAI
    endpoint_url = end_url
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }

    summaries = []
    for result in results:
        prompt = f"""
        The following is a member's demographic and content data:

        Member ID: {result['id']}
        Demographics: {result['member_demographic']}
        Content: {result['content']}

        Generate a concise health summary on the basis of above provided details.
        """
        payload = {
            "messages": [
                {"role": "system", "content": "You are a medical assistant that generates health summaries."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.5,
            "max_tokens": 500
        }
        response = requests.post(endpoint_url, headers=headers, json=payload)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to get response from LLM")
        summary = response.json()["choices"][0]["message"]["content"]
        summaries.append({
            "E_ID": result["id"],
            "summary": summary
        })

    return {"summaries": summaries}
# python -m uvicorn main:app --reload  