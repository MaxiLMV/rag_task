import requests

API_URL = "http://localhost:5000/ask"


# A test for a simple question
def test_basic_question():
    question = "What was the revenue in Q2 FY24?"
    response = requests.post(API_URL, json={"question": question})

    assert response.status_code == 200, "Status code should be 200 OK"

    data = response.json()
    assert "answer" in data, "Response should include 'answer'"
    assert "sources" in data, "Response should include 'sources'"
    answer = data["answer"].lower()
    assert "q2 fy24" in answer or "q2 fiscal year 2024" in answer or "$13,507 million" in answer, \
        "Answer should reference the correct quarter or at least the expected number"


# A test for an empty input
def test_invalid_question():
    response = requests.post(API_URL, json={})
    assert response.status_code == 422 or response.status_code == 400, "Should return a client error for bad input"


# A test for a non-document question
def test_nonexistent_question():
    question = "What is the airspeed velocity of an sparrow?"
    response = requests.post(API_URL, json={"question": question})

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["answer"], str)
    assert isinstance(data["sources"], list)


# A test for multiple questions at the same time
def test_multiple_questions_in_one():
    question = "What was the revenue in Q2 FY24 and what is the gross margin?"
    response = requests.post(API_URL, json={"question": question})

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["answer"], str)
    answer = data["answer"].lower()
    assert "revenue" in answer and "gross margin" in answer, \
        "Answer should mention revenue and gross margin"
    assert isinstance(data["sources"], list)


# A test for a long question
def test_very_long_question():
    long_question = (
        "In the context of NVIDIA's financial performance for the fiscal year 2024, "
        "especially focusing on the second quarter and the comparison with the previous "
        "quarters as well as the same quarter from the previous year, can you explain not only "
        "the reported revenue figures, but also any major announcements made regarding product lines, "
        "strategic partnerships, and shareholder returns such as dividends or share buybacks, while "
        "also providing relevant gross margin and net income values where possible?"
    )

    response = requests.post(API_URL, json={"question": long_question})
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["answer"], str)
    assert isinstance(data["sources"], list)
    assert any(keyword in data["answer"].lower() for keyword in ["revenue", "net income", "dividend", "buyback"]), \
        "Answer should include financial performance info"
