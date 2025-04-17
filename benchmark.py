import os
import argparse
import time
import json
from typing import List, Dict, Any, Tuple

from src.config import OLLAMA_MODEL, OLLAMA_URL, TOPICS
from src.vector_db import load_vector_db, get_embeddings
from src.chat import get_ollama_client, search_similar_chunks, create_prompt

# --- Test Data ---
# Define test questions and simple ground truth keywords for evaluation
# IMPORTANT: Replace these with actual relevant questions and expected keywords for your data
BENCHMARK_QUESTIONS: Dict[str, List[Dict[str, str]]] = {
    "jfk_files": [
        {"question": "What agency was investigating JFK's assassination?", "keywords": "Warren Commission, FBI, CIA"},
        {"question": "Who was Lee Harvey Oswald?", "keywords": "assassin, alleged, shooter, rifle"},
        # Add more JFK questions
    ],
    "literature": [
        {"question": "Who wrote Hamlet?", "keywords": "Shakespeare, William"},
        {"question": "What is the main theme of 'To Kill a Mockingbird'?", "keywords": "justice, prejudice, innocence, mockingbird"},
        {"question": "What makes the Gospel book of Mark unique among the Gospels?", "keywords": "straightforward, urgent, Jesus's actions"},
        {"question": "What makes the Gospel book of Matthew unique among the Gospels?", "keywords": "structured, formal, teaching focused, Papacy, Pope"},
        {"question": "What makes the Gospel book of Luke unique among the Gospels?", "keywords": "polished, historical, detailed"},
        {"question": "What makes the Gospel book of John unique among the Gospels?", "keywords": "philosophical, symbolic, theological, broad, reflective"},
        {"question": "What was the main event in Matthew 16:16-19?", "keywords": "first Pope, Papacy, ordination, peter is the first pope, Catholic Church"},
        {"question": "What was the name of the first Pope?", "keywords": "Peter, Simon, Simon Peter, Apostle Peter"},
        {"question": "What was the title given to Mary by Gabriel in the Gospel book of Luke and what did it mean?", "keywords": "full of grace, κεχαριτωμένη, kecharitomene, without sin, will not sin"},
        {"question": "What were Jesus's last words on the cross according to Luke?", "keywords": "Father, forgive them, for they know not what they do"},
        # Add more literature questions
    ],
    "math": [
        {"question": "What is the Pythagorean theorem?", "keywords": "a^2, b^2, c^2, right, triangle, hypotenuse"},
        {"question": "What is the derivative of x^2?", "keywords": "2x, power rule, calculus"},
        {"question": "What is x*4 when x=5?", "keywords": "20"},
        {"question": "Solve -11*y - 263*y + 3162 = -88*y for y", "keywords": "17"},
        {"question": "Solve 20*a + 0*a - 10*a = -40 for a.", "keywords": "-4"},
        {"question": "Solve 24 = 1601*c - 1605*c for c.", "keywords": "-6"},
        {"question": "Solve 72 = -25*n + n for n.", "keywords": "-3"},
        {"question": "Solve -10*t = -5*t + 5*t for t.", "keywords": "0"},
        {"question": "Solve -525*u + 566*u - 205 = 0 for u.", "keywords": "5"},
        {"question": "Suppose a function f(x) is defined as f(x) = ax^2 + bx + c. If f(1) = 6, f(2) = 11, and f(3) = 18, find the value of f(4).", "keywords": "Quadratic function, System of equations, Substitution, Evaluate f(4), 27"},
        # Add more math questions
    ],
    "science": [
        {"question": "What is photosynthesis?", "keywords": "plants, sunlight, energy, carbon dioxide, oxygen, chlorophyll"},
        {"question": "What is the theory of relativity?", "keywords": "Einstein, gravity, spacetime, E=mc^2"},
        # Add more science questions
    ],
}

# --- Evaluation Function ---

def evaluate_answer(question: str, answer: str, keywords_str: str) -> float:
    """
    Simple keyword-based evaluation. Returns fraction of keywords found in the answer.
    Case-insensitive.

    Args:
        question: The question asked (for context, not used in this simple eval).
        answer: The LLM's generated answer.
        keywords_str: A comma-separated string of expected keywords.

    Returns:
        Score between 0.0 and 1.0.
    """
    if not keywords_str or not answer:
        return 0.0

    keywords = [kw.strip().lower() for kw in keywords_str.split(',')]
    answer_lower = answer.lower()
    found_count = 0
    for kw in keywords:
        if kw in answer_lower:
            found_count += 1

    return found_count / len(keywords) if keywords else 0.0

# --- Main Benchmark Logic ---

def run_benchmark(specific_topics: List[str] | None = None):
    """
    Runs the benchmark questions against the specified topic databases.
    """
    print("--- Starting Benchmark ---")

    topics_to_run = specific_topics if specific_topics else TOPICS
    results = {}

    print("Initializing models...")
    try:
        embeddings = get_embeddings()
        ollama = get_ollama_client()
    except Exception as e:
        print(f"Error initializing models: {e}. Aborting benchmark.")
        return

    total_start_time = time.time()

    for topic in topics_to_run:
        if topic not in BENCHMARK_QUESTIONS:
            print(f"Warning: No benchmark questions defined for topic '{topic}'. Skipping.")
            continue

        print(f"\n--- Benchmarking Topic: {topic} ---")
        topic_start_time = time.time()
        topic_scores = []
        question_results = []

        try:
            print(f"Loading vector database for topic: {topic}...")
            chroma_db = load_vector_db(topic=topic, embeddings=embeddings)
            print("Database loaded.")
        except Exception as e:
            print(f"Error loading database for topic '{topic}': {e}. Skipping topic.")
            results[topic] = {"error": str(e), "average_score": 0.0, "questions": []}
            continue

        questions = BENCHMARK_QUESTIONS[topic]
        for i, q_data in enumerate(questions):
            question = q_data["question"]
            keywords = q_data["keywords"]
            print(f"  Running Q{i+1}/{len(questions)}: {question[:80]}...")
            q_start_time = time.time()

            try:
                # RAG Pipeline
                similar_chunks = search_similar_chunks(question, chroma_db)
                prompt = create_prompt(question, similar_chunks)

                # Get LLM Answer
                answer = ollama(prompt)
                q_time = time.time() - q_start_time

                # Evaluate
                score = evaluate_answer(question, answer, keywords)
                topic_scores.append(score)

                question_results.append({
                    "question": question,
                    "keywords": keywords,
                    "retrieved_context": similar_chunks,
                    "prompt": prompt,
                    "answer": answer,
                    "score": score,
                    "time_seconds": round(q_time, 2)
                })
                print(f"    Score: {score:.2f}, Time: {q_time:.2f}s")

            except Exception as e:
                print(f"    Error running question {i+1}: {e}")
                topic_scores.append(0.0) # Penalize errors
                question_results.append({
                    "question": question,
                    "keywords": keywords,
                    "error": str(e),
                    "score": 0.0,
                    "time_seconds": round(time.time() - q_start_time, 2)
                })

        topic_time = time.time() - topic_start_time
        average_score = sum(topic_scores) / len(topic_scores) if topic_scores else 0.0
        results[topic] = {
            "average_score": round(average_score, 3),
            "total_time_seconds": round(topic_time, 2),
            "questions": question_results
        }
        print(f"--- Topic '{topic}' finished. Avg Score: {average_score:.3f}, Time: {topic_time:.2f}s ---")

    total_time = time.time() - total_start_time
    print(f"\n--- Benchmark Finished ---")
    print(f"Total Time: {total_time:.2f}s")
    print("\n--- Results Summary ---")
    for topic, res in results.items():
        if "error" in res:
            print(f"Topic: {topic} - ERROR: {res['error']}")
        else:
            print(f"Topic: {topic} - Average Score: {res['average_score']:.3f} ({len(res['questions'])} questions in {res['total_time_seconds']}s)")

    # Save results to JSON
    results_filename = f"benchmark_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(results_filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_filename}")
    except Exception as e:
        print(f"\nError saving results to JSON: {e}")

    return results

# --- Argparse ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG benchmark across specified topics.")
    parser.add_argument(
        "--topics",
        nargs='+',
        choices=TOPICS,
        default=None,
        help="Specific topics to benchmark (default: all topics in config.py)"
    )
    # Add other relevant args if needed (e.g., override OLLAMA_MODEL)

    args = parser.parse_args()

    run_benchmark(specific_topics=args.topics) 