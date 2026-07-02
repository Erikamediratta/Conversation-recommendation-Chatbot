import json
import retriever
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq()

SYSTEM = """You are an SHL Assessment Advisor. Help hiring managers find the right tests.

RULES:
1. If the request is vague, ask ONE clarifying question before recommending.
2. Recommend 1 to 10 assessments from the catalog below ONLY.
3. If user refines their request, update recommendations, do not start over.
4. Refuse general HR advice, legal questions, or off topic requests.
5. Return ONLY this JSON, no extra text:

{
  "reply": "your message to the user",
  "recommendations": [
    {"name": "Assessment Name", "url": "https://...", "test_type": "K"}
  ],
  "end_of_conversation": false
}

CATALOG:
{catalog}
"""


def get_user_query(messages):
    # combine all user messages into one search string
    all_user_text = ""
    for message in messages:
        if message["role"] == "user":
            all_user_text = all_user_text + " " + message["content"]
    return all_user_text


def format_catalog(assessments):
    # turn list of assessments into readable text
    text = ""
    for a in assessments:
        text = text + "NAME: " + a["name"] + "\n"
        text = text + "URL: " + a["url"] + "\n"
        text = text + "TYPE: " + ", ".join(a["test_type"]) + "\n"
        text = text + "DESCRIPTION: " + a["description"] + "\n\n"
    return text


def chat(messages):

    #  get all user messages combined
    user_query = get_user_query(messages)

    #  search catalog for top 15 matching assessments
    retrieved = retriever.search_query(user_query, top_k=15)

    # convert assessments to text and put into prompt
    catalog_text  = format_catalog(retrieved)
    system_prompt = SYSTEM.replace("{catalog}", catalog_text)

    # send to Groq
    all_messages = [{"role": "system", "content": system_prompt}] + messages

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=1000,
            messages=all_messages,
        )
        raw_text = response.choices[0].message.content

    except Exception as e:
        print(f"ERROR: {e}")
        return {
            "reply": "Something went wrong, please try again.",
            "recommendations": [],
            "end_of_conversation": False
        }

    #  convert text response to Python dictionary
    try:
        result = json.loads(raw_text)
    except:
        return {
            "reply": "Could not understand response, please try again.",
            "recommendations": [],
            "end_of_conversation": False
        }

    
    return {
        "reply": result.get("reply", "How can I help?"),
        "recommendations": result.get("recommendations", []),
        "end_of_conversation": result.get("end_of_conversation", False)
    }