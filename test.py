from query_data import query_rag
from langchain_community.llms.ollama import Ollama

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""


def test_grocery_list():
    assert query_and_validate(
        question="How many items are in my list? (Answer with the number only)",
        expected_response="8",
    )


def test_microbiome():
    assert query_and_validate(
        question="What is the budget of the Human Microbiome Project, in dollars? (Answer with the number only)",
        expected_response="150,000,000",
    )

def test_llm_document():
    assert query_and_validate(
        question="What is the cost of training gpt3, in dollars? (Answer with the number only)",
        expected_response="12,000,000",
    )

def query_and_validate(question: str, expected_response: str):
    response_text = query_rag(question)
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )

    model = Ollama(model="llama3.1")
    evaluation_results_str = model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

    print(prompt)

    if "true" in evaluation_results_str_cleaned:
        # Print response in Green if it is correct.
        print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return True
    elif "false" in evaluation_results_str_cleaned:
        # Print response in Red if it is incorrect.
        print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return False
    else:
        raise ValueError(
            f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )