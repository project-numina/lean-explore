import json

def remove_leanexplore_results_from_list(filepath: str) -> None:
    """Reads a JSON file containing a list of objects,
    removes the 'leanexplore_results' key from each object in the list,
    and writes back to the file.

    Args:
        filepath: The path to the JSON file.
    """
    try:
        with open(filepath, 'r') as f:
            data_list = json.load(f)

        if not isinstance(data_list, list):
            print(f"Error: Expected a list of JSON objects in {filepath}, but found {type(data_list)}.")
            # Attempt to process it as a single object as a fallback, like the previous script
            if isinstance(data_list, dict) and 'leanexplore_results' in data_list:
                del data_list['leanexplore_results']
                print(f"Processed as a single object: Successfully removed 'leanexplore_results' from {filepath}")
                with open(filepath, 'w') as f:
                    json.dump(data_list, f, indent=4)
            else:
                print("Could not process the file as a list or a single object with the target key.")
            return

        modified_count = 0
        for item in data_list:
            if isinstance(item, dict) and 'leanexplore_results' in item:
                del item['leanexplore_results']
                modified_count += 1

        if modified_count > 0:
            print(f"Successfully removed 'leanexplore_results' from {modified_count} object(s) in {filepath}")
        else:
            print(f"'leanexplore_results' key not found in any of the objects in {filepath}")

        with open(filepath, 'w') as f:
            json.dump(data_list, f, indent=4)

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}. Ensure it's a valid JSON array.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    file_path = 'benchmarking/search_results.json'
    # Name the script process_results.py as per your terminal commands
    remove_leanexplore_results_from_list(file_path)
