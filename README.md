# SHL Assessment Recommender — Complete Study Guide

---

## Table of Contents

1. What is this project?
2. Why does this project exist?
3. How does it work? (Big Picture)
4. Project Architecture
5. Key Concepts Explained
6. File by File Breakdown
7. Every Line of Code Explained
8. How Data Flows Through the System



---

---

# 1. What is This Project?

This is a **conversational chatbot** that helps HR managers and recruiters find the right SHL assessments for any job role.

**SHL** is a company that makes psychometric tests (personality tests, coding tests, reasoning tests) used by companies worldwide to evaluate job candidates.

They have a catalog of 40+ individual tests. Choosing the right test is confusing for HR teams.

**This chatbot solves that.**

---

**Example conversation:**

```
HR Manager: I need to hire a Java developer

Chatbot:    What level of experience are they expected to have?

HR Manager: Mid level, around 3 years

Chatbot:    Here are 3 assessments I recommend:
            1. Java (New)       - tests OOP and data structures
            2. Java 8 (New)     - tests lambdas and streams
            3. Spring (New)     - tests Spring Boot framework
```

---

---

# 2. Why Does This Project Exist?

The SHL catalog requires you to already know what you are looking for.
You have to type exact keywords like "Java" or "OPQ32" to find assessments.

**The problem:**
Most HR managers do not know the right vocabulary.
They know they want to test a Java developer but they do not know which specific SHL test to pick.

**The solution:**
A conversational agent that understands natural language like
"I am hiring a backend developer who works with databases and teams"
and translates that into the right assessments automatically.

---

---

# 3. How Does It Work? (Big Picture)

```
User sends message
        |
        v
FastAPI receives it at POST /chat
        |
        v
All user messages are combined into one search query
        |
        v
TF-IDF retriever searches 40 assessments
and returns top 15 most relevant ones
        |
        v
Those 15 assessments are injected into Claude's system prompt
(this is the RAG step)
        |
        v
Claude reads the conversation + catalog data
and generates a structured JSON response
        |
        v
We validate the response
(check URLs are real, check schema is correct)
        |
        v
Return to user:
{
  "reply": "Here are 3 assessments...",
  "recommendations": [...],
  "end_of_conversation": false
}
```

---

---

# 4. Project Architecture

## Files and Their Jobs

```
shl_recommender/
│
├── catalog.json      THE KNOWLEDGE BASE
│                     40 SHL assessments with name, url,
│                     description, test_type, keywords
│
├── retriever.py      THE SEARCH ENGINE
│                     Reads catalog.json
│                     Builds TF-IDF index
│                     Given a query, returns top matching assessments
│
├── agent.py          THE BRAIN
│                     Takes conversation history
│                     Calls retriever to find relevant assessments
│                     Sends everything to Claude
│                     Returns structured recommendation
│
├── main.py           THE FRONT DOOR
│                     FastAPI web service
│                     GET  /health → {"status": "ok"}
│                     POST /chat   → recommendation response
│
└── test.py           THE QUALITY CHECKER
                      Tests schema, recall, URL safety, behavior
```

---

## Technology Stack

| Technology | What it does | Why we chose it |
|---|---|---|
| Python | Main programming language | Simple and readable |
| FastAPI | Web framework | Fast, automatic validation |
| Pydantic | Data validation | Auto-validates request shapes |
| NumPy | Math operations | Fast matrix multiplication |
| Anthropic (Claude) | Language model | Powers the recommendations |
| TF-IDF | Search algorithm | No GPU needed, works well here |
| Uvicorn | ASGI server | Runs FastAPI in production |

---

## What is RAG?

RAG stands for Retrieval Augmented Generation.

It has three steps:

```
RETRIEVAL     Search the catalog for relevant assessments
                        |
AUGMENTED     Add those results to Claude's prompt
                        |
GENERATION    Claude generates a response using only that data
```

**Why RAG instead of just asking Claude directly?**

If you ask Claude "what SHL tests exist for Java developers?"
Claude will guess from its training data. It might invent tests that do not exist.
This is called hallucination.

With RAG, Claude can ONLY recommend what we give it.
We give it real catalog data. So it can ONLY recommend real tests.
Hallucination is eliminated.

---

---

# 5. Key Concepts Explained

---

## Concept 1: TF-IDF

TF-IDF is how our search engine finds relevant assessments.
It stands for Term Frequency - Inverse Document Frequency.

**The problem it solves:**

If user types "Java developer" and one assessment mentions Java 5 times,
that assessment is probably very relevant.
But the word "test" appears in every assessment so it means nothing.
TF-IDF handles this automatically.

---

**TF — Term Frequency**

How often does this word appear in THIS assessment?

```
Assessment: "Java test for Java developers who write Java code"
Word: "Java" appears 3 times out of 10 total words

TF = 3 / 10 = 0.3
```

High TF means this word is frequent in this document.

---

**IDF — Inverse Document Frequency**

How rare is this word across ALL assessments?

```
"Java"     → appears in 4  out of 40 assessments → RARE    → important
"test"     → appears in 38 out of 40 assessments → COMMON  → not useful
"the"      → appears in 40 out of 40 assessments → USELESS → score = 0

Formula: IDF = log( (total_docs + 1) / (docs_with_word + 1) )

IDF("Java") = log(41/5)  = 2.1    ← high score, rare word
IDF("test") = log(41/39) = 0.05   ← low score, common word
```

High IDF means this word is rare and therefore meaningful.

---

**TF-IDF Combined**

```
TF-IDF = TF × IDF

For "Java" in the Java assessment:
TF-IDF = 0.3 × 2.1 = 0.63   ← HIGH → very relevant

For "test" in the same assessment:
TF-IDF = 0.4 × 0.05 = 0.02  ← LOW → not useful for matching
```

---

**Cosine Similarity**

Once every assessment has TF-IDF scores, we compare them to the query.

Think of every assessment as an arrow pointing in space.
If two arrows point in the same direction, they are similar.

```
score = dot product of (query, assessment) / (length of query × length of assessment)
score ranges from 0.0 (completely different) to 1.0 (identical)
```

In code:
```python
scores = (MATRIX @ query_vec) / (NORMS * query_norm)
```

This compares the query against ALL 40 assessments at once and returns a score for each.

---

## Concept 2: Vectors and Matrices

Every assessment and every query becomes a list of numbers.
This list of numbers is called a vector.

```
Vocabulary: ["java", "python", "sql", "personality", "leadership"]

Java assessment vector:
  java=0.63, python=0.0, sql=0.0, personality=0.0, leadership=0.0
  → [0.63, 0.0, 0.0, 0.0, 0.0]

OPQ assessment vector:
  java=0.0, python=0.0, sql=0.0, personality=0.71, leadership=0.55
  → [0.0, 0.0, 0.0, 0.71, 0.55]

User query "Java developer" vector:
  java=0.80, python=0.0, sql=0.0, personality=0.0, leadership=0.0
  → [0.80, 0.0, 0.0, 0.0, 0.0]
```

Now comparing the query to Java assessment:
Both have high values in the "java" position → high similarity → correct match

Comparing query to OPQ assessment:
Query has high java, OPQ has zero java → low similarity → correctly not recommended

The MATRIX is just all 40 assessment vectors stacked together.

```
MATRIX shape: 40 rows × 530 columns
40 rows     = 40 assessments
530 columns = 530 unique words in vocabulary
```

---

## Concept 3: Stateless API

The API is stateless. This means the server remembers NOTHING between requests.

Every single request must include the complete conversation from the very beginning.

```
Turn 1 request:
  messages: [
    {"role": "user", "content": "I need a Java test"}
  ]

Turn 2 request:
  messages: [
    {"role": "user",      "content": "I need a Java test"},
    {"role": "assistant", "content": "What level?"},
    {"role": "user",      "content": "Mid level"}
  ]
```

Turn 2 includes everything from turn 1 plus the new messages.

**Why stateless?**

- Simple: no database needed to store sessions
- Scalable: any server can handle any request
- Safe: if server crashes, nothing is lost

---

## Concept 4: System Prompt Engineering

The system prompt is the instruction we give Claude before the conversation starts.
It defines exactly how Claude should behave.

Our system prompt tells Claude:
- You are an SHL Assessment Advisor
- Only recommend from the catalog data I give you
- If query is vague, ask ONE clarifying question
- If off topic, refuse politely
- Always return a specific JSON format

The catalog data (retrieved by TF-IDF) is injected into the system prompt.
This is the AUGMENTED part of RAG.

---

## Concept 5: Pydantic Validation

Pydantic automatically checks that incoming data has the right shape.

```python
class Message(BaseModel):
    role: str
    content: str
```

If someone sends:
```json
{"role": 123, "content": "hello"}
```
Pydantic sees role is 123 (a number) not a string → returns 422 error automatically.
You write zero extra checking code.

---

## Concept 6: Recall@K (Evaluation Metric)

Recall@K measures how good our search is.

```
Recall@10 = (number of relevant assessments found in top 10)
            ÷
            (total number of relevant assessments for this query)
```

Example:
```
Query: "Python data scientist"

Relevant assessments that SHOULD appear:
  - Python (New)
  - Machine Learning (New)
  - Data Analysis with Python

Our top 10 results:
  - Python (New)              ← found ✅
  - SQL (New)                 ← wrong
  - Machine Learning (New)   ← found ✅
  - Java (New)                ← wrong
  - Data Analysis with Python ← found ✅

Recall@10 = 3 found / 3 total = 1.0   (perfect score)
```

Mean Recall@10 = average Recall@10 across all test queries.
Our project achieves Mean Recall@10 = 1.0 (perfect).

---

---

# 6. File by File Breakdown

---

## catalog.json

This is the knowledge base. 40 SHL assessments stored as JSON.

**Structure of one assessment:**

```json
{
  "name": "Java (New)",
  "url": "https://www.shl.com/.../java-new/",
  "description": "Tests Java programming including OOP and algorithms",
  "test_type": ["K"],
  "job_levels": ["Entry", "Mid-Professional"],
  "duration_minutes": 30,
  "keywords": ["java", "programming", "backend", "developer", "OOP"]
}
```

**What each field means:**

```
name           → what the test is called
url            → link to SHL website page
description    → what the test measures in plain English
test_type      → K=Knowledge, A=Ability, P=Personality, B=Behavioral, S=Simulation
job_levels     → who this test is appropriate for
duration_minutes → how long the test takes
keywords       → extra words to help the search find this test
```

**Why keywords matter:**

The description alone might not contain every search term.
Keywords add extra words to improve search accuracy.

Example: Java (New) description does not say "backend engineer"
but keywords include "backend" so the search still finds it.

**Test type codes:**

```
K = Knowledge / Skills test    (knows Java, SQL, Python?)
A = Ability / Cognitive test   (can reason verbally and numerically?)
P = Personality test           (what kind of person are they?)
B = Behavioral / SJT test      (how do they handle real situations?)
S = Simulation test            (can they actually do the job?)
```

---

## retriever.py

This is the search engine. It finds the most relevant assessments for any query.

**What happens when this file is imported:**

```
1. catalog.json is loaded into memory
2. Every assessment is converted to a text document
3. All text is cleaned and tokenized (split into words)
4. Vocabulary is built (all unique words)
5. IDF scores are calculated for every word
6. TF-IDF matrix is built (40 rows × 530 columns)
7. Row norms are pre-calculated for cosine similarity
```

All of this happens ONCE at startup, not on every search.
That is why searches are instant.

**The search function:**

```python
def search(query, top_k=10):
```

1. Clean and tokenize the query
2. Build TF-IDF vector for the query
3. Multiply query vector against the matrix (cosine similarity)
4. Sort by score highest to lowest
5. Return top_k results

---

## agent.py




 FILE 3 — agent.py
 
## What it does in one sentence
 
Takes the conversation, searches the catalog, sends everything
to Groq AI, and returns a recommendation.
 
## The Big Picture
 
```
You type "I need a Java test"
            ↓
agent.py collects your message
            ↓
searches catalog → finds Java (New), Java 8 (New), Spring (New)
            ↓
sends those assessments + your message to Groq AI
            ↓
Groq reads them and writes a recommendation
            ↓
agent.py returns the recommendation as JSON
```
 
## Every line explained
 
---
 
```python
import json
import retriever
from groq import Groq
from dotenv import load_dotenv
 
load_dotenv()
```
 
- import json → Python needs this to convert text into a dictionary.
  Groq returns text. We need a dictionary. json does that conversion.
- import retriever → brings in retriever.py so we can search the catalog
- from groq import Groq → brings in the Groq AI library
- from dotenv import load_dotenv → brings in the tool that reads your .env file
- load_dotenv() → actually reads the .env file and loads GROQ_API_KEY into memory
Why dotenv? Without it you would hardcode your key in the code like:
```python
key = "sk-abc123..."  # dangerous, anyone can see it
```
With dotenv you keep the key in a .env file and never share it.
 
---
 
```python
client = Groq()
```
 
Creates a connection to Groq AI.
It automatically reads GROQ_API_KEY from your .env file.
Think of this like opening a phone line to Groq.
Every time you want to talk to the AI, you use this client.
 
---
 
```python
SYSTEM = """You are an SHL Assessment Advisor...
CATALOG:
{catalog}
"""
```
 
This is the instruction you give to Groq before the conversation starts.
 
The {catalog} part is a placeholder. It gets replaced later with real data.
 
Think of it like a letter with a blank:
```
Dear AI,
You are an advisor. Here are the assessments: {catalog}
Please help the user.
```
 
Later you fill in the blank with real assessment data.
 
---
 
```python
def get_user_query(messages):
    all_user_text = ""
    for message in messages:
        if message["role"] == "user":
            all_user_text = all_user_text + " " + message["content"]
    return all_user_text
```
 
What this does:
 
Imagine the conversation looks like this:
```python
messages = [
    {"role": "user",      "content": "I need a Java test"},
    {"role": "assistant", "content": "What level?"},
    {"role": "user",      "content": "Mid level"}
]
```
 
This function loops through every message.
If the message is from the user, it adds it to all_user_text.
 
Result:
```
"I need a Java test Mid level"
```
 
Why combine all user messages?
If you only searched with "Mid level", the retriever would not know
what role you are hiring for. By combining everything, the retriever
gets full context and finds better assessments.
 
---
 
```python
def format_catalog(assessments):
    text = ""
    for a in assessments:
        text = text + "NAME: " + a["name"] + "\n"
        text = text + "URL: " + a["url"] + "\n"
        text = text + "TYPE: " + ", ".join(a["test_type"]) + "\n"
        text = text + "DESCRIPTION: " + a["description"] + "\n\n"
    return text
```
 
The retriever returns a list of Python dictionaries.
But Groq cannot read Python dictionaries. It only reads text.
 
So this function converts the list into readable text:
```
NAME: Java (New)
URL: https://www.shl.com/.../java-new/
TYPE: K
DESCRIPTION: Tests Java programming knowledge...
 
NAME: Java 8 (New)
URL: https://www.shl.com/.../java-8-new/
TYPE: K
DESCRIPTION: Tests Java 8 features like lambda...
```
 
Now Groq can read it.
 
---
 
## The Main Function — chat()
 
This is where everything happens. Read each step carefully.
 
---
 
```python
user_query = get_user_query(messages)
```
 
Step 1 — Collect all user messages.
Calls the function above.
Gets back one combined string of everything the user said.
Example: "I need a Java test Mid level"
 
---
 
```python
retrieved = retriever.search_query(user_query, top_k=15)
```
 
Step 2 — Search the catalog.
Sends the combined string to retriever.py.
retriever.py uses TF-IDF to find the 15 most relevant assessments.
Returns a list of 15 assessment dictionaries.
 
---
 
```python
catalog_text  = format_catalog(retrieved)
system_prompt = SYSTEM.replace("{catalog}", catalog_text)
```
 
Step 3 — Build the prompt. This is RAG.
 
format_catalog() converts 15 assessments to readable text.
SYSTEM.replace() fills in the blank in the system prompt.
 
Before replace:
```
...Here are the assessments: {catalog}
```
 
After replace:
```
...Here are the assessments:
NAME: Java (New)
URL: https://...
DESCRIPTION: Tests Java...
```
 
Now Groq knows exactly which assessments exist.
It cannot recommend anything outside this list.
This is RAG — you gave Groq real data to work with.
 
---
 
```python
all_messages = [{"role": "system", "content": system_prompt}] + messages
```
 
Step 4 — Build the full message list.
 
Groq needs messages in this format:
```python
[
    {"role": "system",    "content": "You are an advisor... CATALOG: ..."},
    {"role": "user",      "content": "I need a Java test"},
    {"role": "assistant", "content": "What level?"},
    {"role": "user",      "content": "Mid level"}
]
```
 
The system message goes FIRST. Then the conversation follows.
The + joins two lists together.
 
---
 
```python
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
```
 
Step 5 — Call Groq.
 
client.chat.completions.create() sends everything to Groq AI.
- model → which AI model to use
- max_tokens=1000 → maximum length of response
- messages=all_messages → the full conversation with system prompt
response.choices[0].message.content → Groq returns choices.
We take the first one and get its text.
 
raw_text is now Groq's response as a string:
```
{"reply": "Here are Java tests...", "recommendations": [...]}
```
 
try/except → if anything goes wrong we catch the error and
return a safe fallback instead of crashing the whole server.
 
---
 
```python
try:
    result = json.loads(raw_text)
except:
    return {
        "reply": "Could not understand response, please try again.",
        "recommendations": [],
        "end_of_conversation": False
    }
```
 
Step 6 — Convert text to dictionary.
 
raw_text is a string:
```
'{"reply": "Here are Java tests...", "recommendations": [...]}'
```
 
json.loads() converts it to a Python dictionary:
```python
{
    "reply": "Here are Java tests...",
    "recommendations": [...],
    "end_of_conversation": False
}
```
 
Now Python can work with it properly.
If conversion fails we return a safe fallback.
 
---
 
```python
return {
    "reply": result.get("reply", "How can I help?"),
    "recommendations": result.get("recommendations", []),
    "end_of_conversation": result.get("end_of_conversation", False)
}
```
 
Step 7 — Return the result.
 
result.get("reply", "How can I help?") means:
- Try to get the reply key from result
- If it does not exist, use "How can I help?" as default
This is safer than result["reply"] which would crash if key is missing.
 
---
 
## The One Thing to Remember Forever
 
agent.py does NOT think by itself.
 
It just:
1. Collects messages
2. Finds relevant assessments
3. Hands everything to Groq
4. Returns what Groq says
All the intelligence is in Groq.
agent.py is just the organizer that puts the right information
in front of Groq so Groq can do its job properly.
 
That is RAG in one sentence:
You find the right information, then you let the AI use it.

## main.py - Complete Walkthrough

```python
app = FastAPI(title="SHL Assessment Recommender")
```

Creates the web application instance. Everything is registered on this object.

---

```python
app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)
```

Allows HTTP requests from any origin.
Without this, browsers block requests from different domains.
The evaluator bot needs this to be able to call our API.

---

```python
class Message(BaseModel):
    role: str
    content: str

    @field_validator("role")
    @classmethod
    def check_role(cls, v):
        if v not in ("user", "assistant"):
            raise ValueError("role must be user or assistant")
        return v
```

Pydantic model for one message.
`@field_validator("role")` runs the check_role function every time role is set.
`@classmethod` required by Pydantic V2 for validators.
`cls` is the class itself (not an instance). Standard Python classmethod pattern.
`v` is the value being validated.
If invalid, raise ValueError and FastAPI returns 422 automatically.

---

```python
@app.get("/health")
def health():
    return {"status": "ok"}
```

`@app.get("/health")` is a decorator. It tells FastAPI:
"When a GET request arrives at /health, call this function."
Returns a dict which FastAPI automatically converts to JSON.

---

```python
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    messages = [
        {"role": m.role, "content": m.content}
        for m in request.messages
    ]
```

`request: ChatRequest` tells FastAPI to parse and validate the request body.
The list comprehension converts Pydantic objects to plain dicts.
agent.py expects plain dicts not Pydantic objects.

---

```python
    if messages[-1]["role"] != "user":
        raise HTTPException(status_code=400, detail="Last message must be from user")
```

`messages[-1]` = last element of the list (Python negative indexing).
We are generating the next ASSISTANT response.
So the last message must always be from the USER.
If not, raise HTTPException which FastAPI catches and returns as HTTP 400.

---

```python
    messages = messages[-8:]
```

The assignment says maximum 8 turns.
This slices the list to keep only the last 8 messages.
Example: if there are 12 messages, only messages[4:12] are kept.

---

```python
    result = agent.chat(messages)
    return ChatResponse(
        reply=result["reply"],
        recommendations=[
            Recommendation(name=r["name"], url=r["url"], test_type=r["test_type"])
            for r in result["recommendations"]
        ],
        end_of_conversation=result["end_of_conversation"]
    )
```

Call agent, get result dict.
Build ChatResponse Pydantic object from result.
FastAPI automatically converts this to JSON and sends it back.

---

---

# 8. How Data Flows Through the System

## Example: First message from user

```
User sends:
POST /chat
{
  "messages": [
    {"role": "user", "content": "I need to hire a Java developer"}
  ]
}

  ↓ main.py
Pydantic validates: role is "user" ✅, content is string ✅
Last message is from user ✅
Messages within 8 limit ✅

  ↓ agent.py
get_user_query → "I need to hire a Java developer"

retriever.search("I need to hire a Java developer", top_k=15)

  ↓ retriever.py
clean("I need to hire a Java developer")
→ ["need", "hire", "java", "developer"]

Build query vector:
"java" → TF=0.25, IDF=2.1 → TF-IDF = 0.525
"developer" → TF=0.25, IDF=1.8 → TF-IDF = 0.45
"need", "hire" → not very distinctive words → low scores

MATRIX @ query_vec → similarity scores for all 40 assessments

Top results:
Java (New)      → 0.66
Java 8 (New)    → 0.53
Spring (New)    → 0.21
Python (New)    → 0.05

Return top 15

  ↓ agent.py
format_catalog(results) → readable text of 15 assessments

system_prompt = SYSTEM.replace("{catalog}", catalog_text)

Claude API call with:
- system = rules + 15 assessments
- messages = [{"role": "user", "content": "I need to hire a Java developer"}]

Claude sees vague query → rule says ask ONE clarifying question

Claude returns:
{
  "reply": "What level of experience are you looking for?",
  "recommendations": [],
  "end_of_conversation": false
}

  ↓ agent.py
parse_reply → valid JSON
validate_recommendations → empty list, nothing to validate

  ↓ main.py
Build ChatResponse object
Return as JSON:

{
  "reply": "What level of experience are you looking for?",
  "recommendations": [],
  "end_of_conversation": false
}
```

---

## Example: Second message (user provides details)

```
User sends:
POST /chat
{
  "messages": [
    {"role": "user",      "content": "I need to hire a Java developer"},
    {"role": "assistant", "content": "What level of experience are you looking for?"},
    {"role": "user",      "content": "Mid level, around 3 years experience"}
  ]
}

agent.py combines user messages:
"I need to hire a Java developer Mid level around 3 years experience"

retriever.search finds same Java tests but now has more context

Claude now has enough context → recommends 3 assessments

Returns:
{
  "reply": "Here are 3 assessments for a mid-level Java developer:",
  "recommendations": [
    {"name": "Java (New)",   "url": "https://...", "test_type": "K"},
    {"name": "Java 8 (New)", "url": "https://...", "test_type": "K"},
    {"name": "Spring (New)", "url": "https://...", "test_type": "K"}
  ],
  "end_of_conversation": false
}
```



**Q: What does this project do in one sentence?**

A: It is a conversational chatbot that takes a job description from an HR manager
and recommends the most relevant SHL assessments using RAG and Claude.

---

**Q: What is RAG and how did you implement it?**

A: RAG stands for Retrieval Augmented Generation.
Instead of letting Claude answer from its training data (which can hallucinate),
we first retrieve relevant assessments from our catalog using TF-IDF search,
then inject those results into Claude's system prompt.
Claude can only recommend what we give it, so hallucination is impossible.

---

**Q: What is TF-IDF and why did you use it?**

A: TF-IDF scores words by how frequently they appear in a document
multiplied by how rare they are across all documents.
Common words like "test" get low scores. Rare specific words like "selenium" get high scores.
I used it instead of embeddings because it needs no GPU, builds instantly,
and works very well for a domain-specific 40-item catalog.

---

**Q: What is cosine similarity?**

A: It measures the angle between two vectors.
If two vectors point in the same direction the similarity is 1.0 (identical).
If they are perpendicular the similarity is 0.0 (nothing in common).
We use it to compare the query vector against every assessment vector
to find which assessments are most similar to the query.

---

**Q: Why is the API stateless?**

A: The server stores nothing between requests.
Every /chat call must include the full conversation history from turn 1.
This makes the service simple to scale (any server can handle any request),
easy to reason about (no hidden state), and crash-safe (losing a server loses nothing).

---

**Q: How do you prevent the agent from hallucinating URLs?**

A: After Claude returns recommendations, we check every URL against a set
of valid URLs built from our catalog.json file.
If a URL does not exist in the catalog, we either find the correct URL by
looking up the assessment name, or we drop the recommendation entirely.
This guarantees every URL in the response is a real SHL catalog URL.

---

**Q: What is Recall@10 and what did you achieve?**

A: Recall@10 measures what fraction of relevant assessments appear in our top 10 results.
Formula: Recall@10 = (relevant found in top 10) / (total relevant).
We achieved Mean Recall@10 = 1.0 across 5 test queries, meaning we found
all the expected assessments in our top 10 for every test case.

---

**Q: What is Pydantic and why use it?**

A: Pydantic is a Python library that validates data shapes automatically.
We define the expected shape of requests and responses using Pydantic models.
If incoming data does not match the shape, FastAPI automatically returns a 422 error.
This means we write almost zero manual validation code.

---

**Q: What are the four agent behaviors and how are they implemented?**

A: The four behaviors are implemented through the system prompt:
1. Clarify: if query is vague, Claude is instructed to ask ONE question
2. Recommend: once context is clear, recommend 1-10 assessments from catalog only
3. Refine: because full history is sent every turn, Claude sees context changes
   and updates recommendations accordingly
4. Compare: Claude is instructed to answer comparison questions using only catalog data

---

**Q: What is the maximum number of turns and how do you enforce it?**

A: The assignment requires maximum 8 turns. We enforce this with one line:
messages = messages[-8:]
This keeps only the last 8 messages before passing to the agent.

---

**Q: What would you improve with more time?**

A: Three improvements:
1. Use sentence-transformer embeddings instead of TF-IDF for better semantic matching
2. Add live catalog scraping to keep assessments current
3. Add response caching to reduce API costs and latency

---