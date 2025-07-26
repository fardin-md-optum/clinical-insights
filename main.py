from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
load_dotenv()
import os
import json
from fastapi.responses import StreamingResponse
import numpy as np
import openai
from openai import AzureOpenAI


api_key = os.getenv("AZURE_OPENAI_API_KEY")
tok_url = os.getenv("TOKEN_URL")
gr_type = os.getenv("GRANT_TYPE")
cl_id = os.getenv("CLIENT_ID")
cl_secret = os.getenv("CLIENT_SECRET")
graph_url = os.getenv("GRAPHQL_URL")
end_url = os.getenv("ENDPOINT_URL")
end_point = os.getenv("AZURE_ENDPOINT")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#For Embeddings
def truncate_text(text, max_tokens=8192):
    return text[:max_tokens]

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

client = AzureOpenAI(
    api_key=api_key,
    api_version="2023-05-15",
    azure_endpoint=end_point
)

# Request model 
class MemberRequest(BaseModel):
    enterpriseIndividualIdentifier: str
    firstName: str
    state: str
    prompt: str
    role: str

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
    def stream_openai_response(results, endpoint_url, headers):
        for result in results:
            prompt = f"""
            You are a clinical summarization assistant. Based on the following member data, generate a clear, user-friendly clinical summary that helps a healthcare professional quickly understand the member's health status.

            Member ID: {result['id']}
            Demographics: {result['member_demographic']}
            Content: {result['content']}

            The summary should be tailored for the following role: **{request.role}**
            Also return for what role are you tailoring the response in the member demographics section.

            Instructions:       
            - Adapt the tone, focus, and level of detail based on the specified role. For example:
                - For a **Doctor**, emphasize clinical relevance, treatment implications, and urgent care needs.
                - For a **Manager**, highlight care program performance, risk indicators, and population-level insights.
                - For a **Care Coordinator**, focus on follow-up needs, care gaps, and coordination opportunities.
                - For other roles, adjust accordingly to ensure the summary is actionable and relevant.
            - Present the summary in a narrative, conversational tone suitable for a dashboard or care team briefing.
            - Begin with a brief overview of the member's demographics and general health. In this section also mention the role for which you are tailoring the response.
            - For each section (Member Demographics, Active Conditions, Behavioral Health, Medication Gaps, Care Programs, and Member Insights), write a short paragraph that explains the key points in plain language. Also dynamically generate other sections that might be relevant to the speciifed role.
            - Avoid listing conditions or medications unless necessary—group and describe them instead.
            - Use section headers wrapped in *** (e.g., ***Member Demographics***, ***Active Condition***) to separate sections.
            - Do not include closing remarks or generic statements.
            - Highlight the important parts by making them bold.
            - Include a dedicated section titled ***Historical Health Activity*** that summarizes the member's historical records, including:
              - Hospital admissions or emergency visits
              - Adherence to care programs
              - Medication adherence
              - Regularity of checkups or follow-ups
              
            Additional User Prompt: {request.prompt}

            - Create a separate section with a title to answer the Additional user Prompt with concise explanation for your answer if additional user prompt is not Null(If it's NULL do not generate this section).(also enclose the title in *** for handling purpose)

            Note: Treat null end dates as ongoing conditions. Focus on clinical significance and clarity. Avoid excessive medical jargon. The goal is to make the summary easy to read and immediately informative.
            """

            payload = {
                "messages": [
                    {"role": "system", "content": "You are a medical assistant that generates health summaries."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "top_p" : 1.0,
                "max_tokens": 1000,
                "stream" : True
            }

            embedding1 = client.embeddings.create(
                input=truncate_text(prompt),
                model="text-embedding-ada-002"
            ).data[0].embedding
            
            response = requests.post(endpoint_url, headers=headers, json=payload, stream=True)

            if response.status_code != 200:
                yield "[Error: Failed to get response from LLM]"
                return
            full_response = ""
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode("utf-8")
                    if decoded_line.startswith("data: "):
                        data = decoded_line[6:]
                        if data == "[DONE]":
                            break
                        try:                 
                            chunk = json.loads(data)
                            choices = chunk.get("choices", [])
                            if choices and "delta" in choices[0]:
                                content = choices[0]["delta"].get("content", "")
                                full_response += content
                                for char in content:
                                    yield char
                            # else:
                            #     yield "[Warning: No content in response chunk]"
                        except Exception as e:
                            yield f"[Error parsing chunk: {str(e)}]"

        embedding2 = client.embeddings.create(
            input=truncate_text(full_response),
            model="text-embedding-ada-002"
        ).data[0].embedding
                
        similarity = cosine_similarity(embedding1, embedding2)
        yield f"\n***Cosine Similarity Score***\n {similarity}"
    return StreamingResponse(stream_openai_response(results, endpoint_url, headers), media_type="text/plain")

#Post Request for concise summary   
@app.post("/generate-concise-summary")
def generate_concise_summary(request: MemberRequest):
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
    def stream_openai_response(results, endpoint_url, headers):
        for result in results:
            prompt = f"""
            You are a clinical summarization assistant. Based on the following member data, generate a concise summary of a few lines in paragraph format about the person's health condition.

            The summary should be tailored for the following role: **{request.role}**

            Member ID: {result['id']}
            Demographics: {result['member_demographic']}
            Content: {result['content']}

            Additional User Prompt: {request.prompt}

            Note: Do not include any closing or follow-up statements.
            """

            payload = {
                "messages": [
                    {"role": "system", "content": "You are a medical assistant that generates concise health summaries."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "top_p" : 1.0,
                "max_tokens": 500,
                "stream" : True
            }
            
            response = requests.post(endpoint_url, headers=headers, json=payload, stream=True)

            if response.status_code != 200:
                yield "[Error: Failed to get response from LLM]"
                return
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode("utf-8")
                    if decoded_line.startswith("data: "):
                        data = decoded_line[6:]
                        if data == "[DONE]":
                            break
                        try:                 
                            chunk = json.loads(data)
                            choices = chunk.get("choices", [])
                            if choices and "delta" in choices[0]:
                                content = choices[0]["delta"].get("content", "")
                                for char in content:
                                    yield char
                            # else:
                            #     yield "[Warning: No content in response chunk]"
                        except Exception as e:
                            yield f"[Error parsing chunk: {str(e)}]"
    return StreamingResponse(stream_openai_response(results, endpoint_url, headers), media_type="text/plain")


# python -m uvicorn main:app --reload  

# prompt = f"""
#             You are a clinical summarization assistant. Based on the following member data, generate a clear, user-friendly clinical summary that helps a healthcare professional quickly understand the member's health status.

#             Member ID: {result['id']}
#             Demographics: {result['member_demographic']}
#             Content: {result['content'][:len(result['content'])//4]}

#             Instructions:
#             - Present the summary in a narrative, conversational tone suitable for a dashboard or care team briefing.
#             - Begin with a brief overview of the member's demographics and general health.
#             - For each section (Member Demographics, Active Conditions, Behavioral Health, Medication Gaps, Care Programs, and Member Insights), write a short paragraph that explains the key points in plain language.
#             - Avoid listing conditions or medications unless necessary—group and describe them instead.
#             - Use section headers wrapped in *** (e.g., ***Member Demographics***, ***Active Condition***) to separate sections.
#             - Do not include closing remarks or generic statements.
#             - Highlight the important words by making them bold.
#             - Include a dedicated section titled ***Historical Health Activity*** that summarizes the member's historical records, including:
#               - Hospital admissions or emergency visits
#               - Adherence to care programs
#               - Medication adherence
#               - Regularity of checkups or follow-ups
              
#             Additional User Prompt: {request.prompt}

#             - Create a separate section to answer the Additional user Prompt with concise explanation for your answer, and give it a title too(also enclose the title in *** for handling purpose)

#             Note: Treat null end dates as ongoing conditions. Focus on clinical significance and clarity. Avoid excessive medical jargon. The goal is to make the summary easy to read and immediately informative.
#             """
