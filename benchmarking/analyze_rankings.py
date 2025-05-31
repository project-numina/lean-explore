# benchmarking/analyze_rankings.py

"""Module for analyzing LLM ranking results from evaluation_output.json."""

import json
import os
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List

EVALUATION_OUTPUT_FILE = "evaluation_output.json"
PLACE_MAP = {1: "1st", 2: "2nd", 3: "3rd"}
PLACES_TO_REPORT = ["1st", "2nd", "3rd"]
# Defines the order and keys for displaying tie details
TIE_DETAIL_KEYS = ["solo", "2-way tie", "3-way tie"]


def calculate_rank_statistics(
    evaluation_data: List[Dict[str, Any]]
) -> DefaultDict[str, DefaultDict[str, DefaultDict[str, int]]]:
    """Calculates 1st, 2nd, 3rd place finishes, detailing ties.

    Args:
        evaluation_data: A list of evaluation dictionaries, each containing
                         an 'llm_parsed_ranking' field.

    Returns:
        A defaultdict where keys are engine names, and values are
        defaultdicts mapping place strings ('1st', '2nd', '3rd') to
        another defaultdict. This inner defaultdict maps tie type
        ('total', 'solo', '2-way tie', '3-way tie') to counts.

    Example:
        {'EngineA': {
            '1st': {'total': 10, 'solo': 5, '2-way tie': 5},
            '2nd': {'total': 3, 'solo': 3}
            }
        }
    """
    engine_scores: DefaultDict[
        str, DefaultDict[str, DefaultDict[str, int]]
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for entry in evaluation_data:
        parsed_ranking = entry.get("llm_parsed_ranking")

        if not parsed_ranking:
            original_query = entry.get("original_query", "Unknown query")
            if entry.get("llm_error"):
                print(
                    f"Info: Skipping query '{original_query}' due to "
                    f"LLM error: {entry['llm_error']}"
                )
            elif isinstance(parsed_ranking, list):  # Empty list
                print(
                    f"Info: Skipping query '{original_query}' as it has an "
                    "empty LLM parsed ranking."
                )
            else:  # None or other unexpected type
                print(
                    f"Warning: Skipping query '{original_query}' due to missing or "
                    f"invalid 'llm_parsed_ranking' field: {parsed_ranking}"
                )
            continue

        current_place = 1
        for rank_group in parsed_ranking:
            if not rank_group:
                continue

            if current_place > 3:  # Only track 1st, 2nd, 3rd
                break

            place_key = PLACE_MAP.get(current_place)
            if place_key:
                num_tied_in_group = len(rank_group)
                tie_type_detail_key = ""

                if num_tied_in_group == 1:
                    tie_type_detail_key = "solo"
                elif num_tied_in_group == 2:
                    tie_type_detail_key = "2-way tie"
                elif num_tied_in_group == 3:  # Max for 3 engines scenario
                    tie_type_detail_key = "3-way tie"

                if tie_type_detail_key:
                    for engine_name in rank_group:
                        engine_scores[engine_name][place_key][tie_type_detail_key] += 1
                        engine_scores[engine_name][place_key]["total"] += 1

            current_place += len(rank_group)

    return engine_scores


def display_rank_statistics(
    engine_scores: DefaultDict[str, DefaultDict[str, DefaultDict[str, int]]]
) -> None:
    """Prints the rank statistics for each engine, including tie details.

    Args:
        engine_scores: A dictionary containing the rank counts and tie details
                       for each engine, as returned by `calculate_rank_statistics`.
    """
    if not engine_scores:
        print("No engine scores to display.")
        return

    print("\n--- Search Engine Ranking Statistics ---")
    sorted_engine_names = sorted(engine_scores.keys())

    for engine_name in sorted_engine_names:
        print(f"\nEngine: {engine_name}")

        # Check if the engine has any scored places at all
        has_any_score = any(
            engine_scores[engine_name][place]["total"] > 0 for place in PLACES_TO_REPORT
        )
        if not has_any_score:
            print("  No 1st, 2nd, or 3rd place finishes recorded in this analysis.")
            continue

        for place in PLACES_TO_REPORT:
            # Total count for this specific place (e.g., total 1st places)
            total_for_place = engine_scores[engine_name][place]["total"]
            print(f"  {place} Place: {total_for_place} times")

            if total_for_place > 0:
                # Indented breakdown of how this place was achieved
                for tie_key in TIE_DETAIL_KEYS:
                    count = engine_scores[engine_name][place][tie_key]
                    if count > 0:
                        # Capitalize first letter of tie_key for display
                        display_tie_key = tie_key[0].upper() + tie_key[1:]
                        print(f"    - {display_tie_key}: {count} times")


def main() -> None:
    """Loads evaluation data, calculates, and displays rank statistics."""
    script_dir = os.path.dirname(__file__)
    eval_file_path = os.path.join(script_dir, EVALUATION_OUTPUT_FILE)

    if not os.path.exists(eval_file_path):
        print(f"Error: Evaluation file not found at '{eval_file_path}'.")
        print(
            f"Please ensure '{EVALUATION_OUTPUT_FILE}' exists in the same "
            "directory as this script."
        )
        return

    try:
        with open(eval_file_path, encoding="utf-8") as f:
            evaluation_data = json.load(f)
        if not isinstance(evaluation_data, list):
            print(
                f"Error: Content of '{eval_file_path}' is not a JSON list."
            )
            return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{eval_file_path}'.")
        return
    except Exception as e:
        print(f"Error reading or parsing '{eval_file_path}': {e}")
        return

    if not evaluation_data:
        print(f"No evaluation entries found in '{eval_file_path}'.")
        return

    print(f"Loaded {len(evaluation_data)} evaluation entries for analysis.")
    engine_scores = calculate_rank_statistics(evaluation_data)
    display_rank_statistics(engine_scores)


if __name__ == "__main__":
    main()
