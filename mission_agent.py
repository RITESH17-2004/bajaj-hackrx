import re

# This file contains the specific, correct logic for solving the "Sachin's Parallel World" mission.
# It is only the "brain" and does not contain any tools.

# --- The Agent's Brain ---

def get_next_action(history: list, document_text: str):
    """
    Determines the next action based on the mission steps and the history.
    This version correctly parses the nested API responses.
    """
    print("\n--- AGENT BRAIN: Evaluating mission state ---")
    
    # Consolidate all gathered information from the 'data' field of the results.
    current_state = {}
    for step in history:
        # The internal update for decoded_landmark is not nested in 'data'
        if step.get("tool") == "internal_update":
            current_state.update(step.get("result", {}))
        else:
            result_data = step.get("result", {}).get("data", {})
            current_state.update(result_data)

    # Step 1: Get the Secret City if we haven't already.
    if "city" not in current_state:
        print("Goal: Step 1 - Get the Secret City.")
        url = "https://register.hackrx.in/submissions/myFavouriteCity"
        print(f"Action: Calling the API to get the city: {url}")
        return {"tool": "make_get_request", "args": {"url": url}}

    # Step 2: Decode the City to find the true landmark.
    elif "decoded_landmark" not in current_state:
        api_city = current_state["city"]
        print(f"Goal: Step 2 - Decode the city '{api_city}'.")
        
        # Find all landmarks listed for the given city.
        matches = re.findall(f"([A-Za-z\s]+)\s+{re.escape(api_city)}", document_text)
        
        if not matches:
            return {"error": f"Could not find the city '{api_city}' in the document's landmark table."}

        landmarks = [match.strip() for match in matches]
        print(f"Analysis: Found the following landmarks for '{api_city}': {landmarks}")

        # The new tie-breaking rule.
        special_landmarks = ["Gateway of India", "Taj Mahal", "Eiffel Tower", "Big Ben"]
        chosen_landmark = ""

        for landmark in landmarks:
            if landmark in special_landmarks:
                chosen_landmark = landmark
                print(f"Analysis: Found a special landmark: '{chosen_landmark}'. Prioritizing it.")
                break
        
        # If no special landmark was found, choose the first one.
        if not chosen_landmark:
            chosen_landmark = landmarks[0]
            print(f"Analysis: No special landmarks found. Choosing the first one: '{chosen_landmark}'.")

        return {"update_history": {"decoded_landmark": chosen_landmark}}

    # Step 3: Choose the correct flight path based on the landmark.
    elif "flightNumber" not in current_state:
        landmark = current_state["decoded_landmark"]
        print(f"Goal: Step 3 - Choose the flight path for landmark '{landmark}'.")
        
        flight_path_rules = {
            "Gateway of India": "getFirstCityFlightNumber",
            "Taj Mahal": "getSecondCityFlightNumber",
            "Eiffel Tower": "getThirdCityFlightNumber",
            "Big Ben": "getFourthCityFlightNumber"
        }
        
        endpoint_name = flight_path_rules.get(landmark, "getFifthCityFlightNumber")
        url = f"https://register.hackrx.in/teams/public/flights/{endpoint_name}"
        print(f"Analysis: Based on the rules, the correct endpoint is {endpoint_name}. Calling URL: {url}")
        return {"tool": "make_get_request", "args": {"url": url}}

    # Step 4: Mission Complete.
    else:
        print("Goal: Final Step - Submit the flight number.")
        answer = f"The final flight number is {current_state['flightNumber']}."
        print(f"Action: Provide the final answer.")
        return {"answer": answer}

# --- The Orchestrator ---
# This function now accepts a tools dictionary to work with real functions.

def run_mission_agent(document_text: str, tools: dict):
    history = []
    max_steps = 5

    for step in range(max_steps):
        print(f"\n--- ORCHESTRATOR: Starting Step {step + 1} ---")
        decision = get_next_action(history, document_text)

        if "answer" in decision:
            return decision["answer"]
        if "error" in decision:
            return f"Agent Error: {decision['error']}"
        
        if "update_history" in decision:
            history.append({"tool": "internal_update", "result": decision["update_history"]})
            continue

        tool_name = decision.get("tool")
        tool_args = decision.get("args", {})
        tool_function = tools.get(tool_name)

        if not tool_function:
            return f"Agent Error: Tool '{tool_name}' not found in provided toolset."

        # The real tool is called here.
        result = tool_function(**tool_args)
        history.append({"tool": tool_name, "args": tool_args, "result": result})

    return "Agent reached max steps without finding an answer."