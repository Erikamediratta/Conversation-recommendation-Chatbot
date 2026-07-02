import os


import retriever
from fastapi.testclient import TestClient
from main import app
from dotenv import load_dotenv
load_dotenv()

client = TestClient(app)

VALID_URLS = {item["url"] for item in retriever.CATALOG}


# # VALID_URLS looks like:
# {
#   "https://www.shl.com/.../java-new/",
#   "https://www.shl.com/.../python-new/",
#   ...all 40 urls
# }

#TEST 1

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
    print(" /health works")

# client.get("/health") → sends a GET request to our /health endpoint
# r.status_code → HTTP status code. 200 means success. 404 means not found. 500 means crashed.
# assert → if this condition is false, the test FAILS and shows an error
# r.json() → converts the response body to a Python dictionary

# TEST 2

def test_schema():
    # Test 1: bad role should be rejected
    r = client.post("/chat", json={
        "messages": [{"role": "alien", "content": "hi"}]
    })
    assert r.status_code == 422
    print(" Bad role rejected")

# status_code 422Pydantic rejected the request (bad shape)

    # Test 2: last message must be from user
    r = client.post("/chat", json={
        "messages": [
            {"role": "user",      "content": "hello"},
            {"role": "assistant", "content": "hi"}
        ]
    })
    assert r.status_code == 400
    print(" Last message not user rejected")

# TEST 3 

def test_url_safety():
    fake = "https://www.shl.com/fake-test/"
    real = "https://www.shl.com/solutions/products/product-catalog/view/java-new/"

    assert fake not in VALID_URLS
    assert real in VALID_URLS
    print(f"URL safety works ")


# TEST 4  recall@10



def recall_at_k(found_names, expected_names, k=10):
    if not expected_names:
        return 1.0

    # Only look at top k results
    top_k = found_names[:k]

    # Convert to lowercase so "Java (New)" matches "java (new)"
    top_k_lower    = {name.lower() for name in top_k}
    expected_lower = {name.lower() for name in expected_names}

    # Count how many expected ones we found
    hits = len(top_k_lower & expected_lower)

    return hits / len(expected_names)

def test_recall():
    cases = [
        (
            "Java developer backend OOP", #search query
            ["Java (New)", "Java 8 (New)"], #expected assessment
            0.5   # minimum acceptable recall
        ),
        (
            "personality leadership executive",
            ["OPQ32r (Occupational Personality Questionnaire)", "Leadership Report (using OPQ32)"],
            0.5
        ),
        (
            "customer service BPO call center",
            ["Situational Judgement Test - Customer Service", "Call Center Simulation"],
            0.5
        ),
    ]

    for query, expected, min_recall in cases:
        results  = retriever.search_query(query, top_k=10)
        found    = [r["name"] for r in results]
        score    = recall_at_k(found, expected)
        passed   = score >= min_recall
        mark     = "✅" if passed else "❌"
        print(query, score)

# TEST 5


def test_vague_query():
    results = retriever.search_query("I need an assessment", top_k=10)

    max_score = -1

    for r in results:
        score = r.get("score", 0)

        if score > max_score:
            max_score = score

    if max_score < 0.15:
        print(f"Vague query ({max_score})-> Agent will clarify")
    else:
        print(f"Confident match ({max_score}) -> Agent can recommend")



#RUN ALL TESTS

if __name__=="__main__":

    print("HEALTH CHECK")
    test_health()

    print("SCHEMA VALIDATION")
    test_schema()

    print("URL SAFETY")
    test_url_safety()

    print("RECALL@10")
    test_recall()

    print("VAGUE QUERY PROBE")
    test_vague_query()
