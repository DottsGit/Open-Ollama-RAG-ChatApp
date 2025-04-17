import os
import argparse
import time
import json
from typing import List, Dict, Any, Tuple, Literal

from src.config import OLLAMA_MODEL, OLLAMA_URL, TOPICS
from src.vector_db import load_vector_db, get_embeddings
from src.chat import get_ollama_client, search_similar_chunks, create_prompt

# --- Test Data ---
# Define test questions and simple ground truth keywords for evaluation
# IMPORTANT: Replace these with actual relevant questions and expected keywords for your data
BENCHMARK_QUESTIONS: Dict[str, List[Dict[str, str]]] = {
    "jfk_files": [
        {"question": "What agency was investigating JFK's assassination?", "keywords": "Warren Commission, FBI, CIA"},
        {"question": "Who was Lee Harvey Oswald?", "keywords": "assassin, defector, gunman"},
        {"question": "Who killed JFK?", "keywords": "Lee Harvey Oswald, CIA"},
        # Add more JFK questions
    ],
    "literature": [
        {"question": "Who wrote Hamlet?", "keywords": "Shakespeare, William"},
        {"question": "What is the main theme of 'To Kill a Mockingbird'?", "keywords": "justice, prejudice, innocence, mockingbird"},
        {"question": "What does the Gospel book of Mark focus on?", "keywords": "authoritative, Messiah, miracles, teachings, Jesus, divine, faith"},
        {"question": "What does the Gospel book of Matthew focus on?", "keywords": "Jesus, prophecy, messiah, fulfillment, divine, authority, old covenant, new covenant"},
        {"question": "What does the Gospel book of Luke focus on?", "keywords": "resurrection, miracles, jesus, redemption, Mary"},
        {"question": "What does the Gospel book of John focus on?", "keywords": "eternal life, Jesus, divine, resurrection, faith"},
        {"question": "What was the main event in Matthew 16:16-19?", "keywords": "foundation, rock, Peter, confession"},
        {"question": "What was the name of the first Pope?", "keywords": "Peter, Simon, Simon Peter, Apostle Peter"},
        {"question": "What was the title given to Mary by Gabriel in the Gospel book of Luke and what did it mean?", "keywords": "full of grace, κεχαριτωμένη, without sin, will not sin, highly favored"},
        {"question": "What were Jesus's last words on the cross according to Luke?", "keywords": "Father, forgive them, for they know not what they do"},
        # Add more literature questions
    ],
    "math": [
        {"question": "What is the Pythagorean theorem?", "keywords": "a\u00b2 + b\u00b2 = c\u00b2, right, triangle, hypotenuse"},
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
        {"question": "In what dimension is a graph reconstruction problem strongly NP-complete?", "keywords": "one dimension, two dimensions"},
        {"question": "Who wrote Eﬃcient numerical method to calculate three-tangle of mixe d states?", "keywords": "Kun Cao, Zheng-Wei Zhou, Guang-Can Guo, and Lixin He"},
        {"question": "What is the goal of polyhomeostatic control?", "keywords": "contrast, homeostatic regulation, target distribution, multiple, stable states"},
        {"question": "What did Zhe Chang and Ning Wu find the derivation of the time at which the entanglement rea ches its ﬁrst maximum with respect to the reciprocal transverse ﬁeld?", "keywords": "minimum, critical point, quantum phase transition"},
        {"question": "What is an atom?", "keywords": "protons, neutrons, electrons, nucleus, element, atomic structure"},
        {"question": "What is Newton’s first law of motion?", "keywords": "inertia, force, motion, acceleration, object, physics"},
        {"question": "What is the water cycle?", "keywords": "evaporation, condensation, precipitation, collection, hydrologic cycle, water vapor"},
        {"question": "What is mitosis?", "keywords": "cell division, chromosomes, nucleus, prophase, metaphase, anaphase, telophase"},
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

    # Extract only the part after "LLM Response:" if it exists
    if "LLM Response:" in answer:
        answer = answer.split("LLM Response:", 1)[1].strip()
    
    keywords = [kw.strip().lower() for kw in keywords_str.split(',')]
    answer_lower = answer.lower()
    found_count = 0
    for kw in keywords:
        if kw in answer_lower:
            found_count += 1

    return found_count / len(keywords) if keywords else 0.0

# --- Main Benchmark Logic ---

def run_benchmark(specific_topics: List[str] | None = None, mode: Literal["rag", "no_rag", "both"] = "both"):
    """
    Runs the benchmark questions against the specified topic databases.
    
    Args:
        specific_topics: List of topics to run the benchmark on. If None, all topics are used.
        mode: Whether to run with RAG ("rag"), without RAG ("no_rag"), or both ("both").
    """
    print("--- Starting Benchmark ---")
    print(f"Mode: {mode}")

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
    
    # Determine which modes to run
    modes_to_run = []
    if mode == "both":
        modes_to_run = ["rag", "no_rag"]
    else:
        modes_to_run = [mode]

    for current_mode in modes_to_run:
        print(f"\n=== Running in {current_mode.upper()} mode ===")
        use_rag = current_mode == "rag"
        mode_results = {}

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
                mode_results[topic] = {"error": str(e), "average_score": 0.0, "questions": []}
                continue

            questions = BENCHMARK_QUESTIONS[topic]
            for i, q_data in enumerate(questions):
                question = q_data["question"]
                keywords = q_data["keywords"]
                print(f"  Running Q{i+1}/{len(questions)}: {question[:80]}...")
                q_start_time = time.time()

                try:
                    # RAG Pipeline (optional)
                    similar_chunks = None
                    if use_rag:
                        similar_chunks = search_similar_chunks(question, chroma_db)
                        print(f"    Retrieved {len(similar_chunks)} chunks for context")
                    else:
                        print(f"    Running without RAG context")

                    # Create prompt with or without context
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
                        "retrieved_context": similar_chunks if use_rag else None,
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
            mode_results[topic] = {
                "average_score": round(average_score, 3),
                "total_time_seconds": round(topic_time, 2),
                "questions": question_results
            }
            print(f"--- Topic '{topic}' finished. Avg Score: {average_score:.3f}, Time: {topic_time:.2f}s ---")

        # Store results for this mode
        results[current_mode] = mode_results

    total_time = time.time() - total_start_time
    print(f"\n--- Benchmark Finished ---")
    print(f"Total Time: {total_time:.2f}s")
    print("\n--- Results Summary ---")
    
    # Print summary by mode
    for current_mode in results:
        print(f"\n=== {current_mode.upper()} Mode Results ===")
        mode_results = results[current_mode]
        for topic, res in mode_results.items():
            if "error" in res:
                print(f"Topic: {topic} - ERROR: {res['error']}")
            else:
                print(f"Topic: {topic} - Average Score: {res['average_score']:.3f} ({len(res['questions'])} questions in {res['total_time_seconds']}s)")
    
    # If both modes were run, print comparison
    if len(results) > 1 and "rag" in results and "no_rag" in results:
        print("\n=== RAG vs. No-RAG Comparison ===")
        for topic in topics_to_run:
            if topic in results["rag"] and topic in results["no_rag"]:
                if "error" not in results["rag"][topic] and "error" not in results["no_rag"][topic]:
                    rag_score = results["rag"][topic]["average_score"]
                    no_rag_score = results["no_rag"][topic]["average_score"]
                    diff = rag_score - no_rag_score
                    print(f"Topic: {topic} - RAG: {rag_score:.3f}, No-RAG: {no_rag_score:.3f}, Difference: {diff:+.3f}")

    # Save results to JSON
    results_filename = f"benchmark_results_{mode}_{time.strftime('%Y%m%d_%H%M%S')}.json"
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
    parser.add_argument(
        "--mode",
        choices=["rag", "no_rag", "both"],
        default="both",
        help="Run with RAG, without RAG, or both (default: both)"
    )
    # Add other relevant args if needed (e.g., override OLLAMA_MODEL)

    args = parser.parse_args()

    run_benchmark(specific_topics=args.topics, mode=args.mode) 