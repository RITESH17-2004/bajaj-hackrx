import logging
import re
import json
import time
from typing import Dict, Any, List, Tuple

from src.llm_interaction_service import LLMInteractionService
from src.query_resolver import QueryResolver
from src.answer_generation_engine import AnswerGenerationEngine
from src.agent_tools import api_call, text_parser

class IntelligentAgent:
    """
    A dynamic agent that uses a RAG-first approach to solve missions.
    It iteratively uses an LLM to decide the next action (e.g., API call, text parsing, document search)
    based on the current goal and relevant information retrieved from documents.
    """
    def __init__(self, llm_service: LLMInteractionService, query_processor: QueryResolver, decision_engine: AnswerGenerationEngine):
        self.llm_service = llm_service
        self.query_processor = query_processor
        self.decision_engine = decision_engine
        self.history = []  # Stores a history of actions and their results
        self.state = {}    # Stores the current state of the mission

    async def run(self, document_chunks: List[Dict], document_text: str) -> str:
        """
        Executes the agent's mission, iteratively taking actions until the goal is achieved or max turns reached.
        """
        logging.info("Starting intelligent agent with RAG-first approach.")
        self.state = {"document_text": document_text}
        goal = "Solve the mission described in the document by figuring out the final answer."
        max_turns = 10

        for i in range(max_turns):
            logging.info(f"[Turn {i+1}/{max_turns}] Current goal: {goal}")
            self.state["current_goal"] = goal

            # Step 1: Use RAG to get relevant information for the current goal from document chunks
            rag_start_time = time.time()
            relevant_info = await self.query_processor.process_query(goal, document_chunks, self.decision_engine)
            rag_end_time = time.time()
            self.state["relevant_info"] = relevant_info
            logging.info(f"RAG retrieval took {rag_end_time - rag_start_time:.2f}s. Info: {relevant_info[:200]}...")

            # Step 2: Use LLM to decide the next action based on current state and RAG results
            action_prompt = self._build_action_prompt()
            llm_start_time = time.time()
            next_action_str = await self.llm_service.llm_engine.generate(action_prompt)
            llm_end_time = time.time()
            logging.info(f"LLM action decision took {llm_end_time - llm_start_time:.2f}s. Action: {next_action_str}")

            # Step 3: Parse and execute the chosen action
            try:
                action, details = self._parse_action(next_action_str)
                logging.info(f"Parsed action: {action}, Details: {details}")

                if action == "api_call":
                    result = api_call(details, self.state)
                    self.state.update(result)
                    self.history.append({"action": "api_call", "result": result})
                    goal = f"The API call returned: {json.dumps(result)}. What is the next step?"
                elif action == "text_parser":
                    result = text_parser(details, self.state)
                    self.state.update(result)
                    self.history.append({"action": "text_parser", "result": result})
                    goal = f"The text parser found: {json.dumps(result)}. What is the next step?"
                elif action == "find_in_document":
                    query = details.get("query")
                    if not query:
                        raise ValueError("Query not found for find_in_document action.")
                    logging.info(f"Intelligent agent performing targeted document search for: '{query}'")
                    found_info = await self.query_processor.process_query(query, document_chunks, self.decision_engine)
                    state_key = f"info_found_for_{query.replace(' ', '_').lower()}"
                    self.state[state_key] = found_info
                    self.history.append({"action": "find_in_document", "details": details, "result": {state_key: found_info}})
                    goal = f"I searched the document for '{query}' and found: {found_info}. What is the next step?"
                elif action == "answer":
                    final_answer = details.get("text", "No answer text provided.")
                    # Clean up the final answer from any extra formatting
                    cleaned_answer = re.sub(r'^text:\s*', '', final_answer).replace('"', '').replace('"', '').strip()
                    logging.info(f"Mission accomplished. Final answer: {cleaned_answer}")
                    return cleaned_answer
                else:
                    logging.warning(f"Unknown action: {action}. Attempting to continue.")
                    goal = f"I tried to perform an unknown action '{action}'. I will try to find the final answer now."

            except Exception as e:
                logging.error(f"Error during action execution: {e}")
                goal = f"An error occurred: {e}. I need to recover and find the final answer."

        return "Agent could not solve the mission within the turn limit."

    def _build_action_prompt(self) -> str:
        """
        Constructs the prompt for the LLM to decide the next action.
        Includes available actions, current goal, and relevant information.
        """
        return f"""You are an expert agent. Your task is to decide the next single action to take to achieve a goal.

        **Available Actions:**
        - `api_call(url: str)`: Call a URL. The URL must be a complete and valid URL.
        - `text_parser(pattern: str)`: Extract information from the document text using a regex pattern.
        - `find_in_document(query: str)`: Search the document for specific information. Use this if you need to find information within the provided document.
        - `answer(text: str)`: Provide the final answer when the mission is complete.

        **Current Goal:**
        {self.state.get('current_goal')}

        **Information from Document:**
        {self.state.get('relevant_info')}

        **Next Action:**
        Respond with ONLY the next action to take in the format `action(parameters)`. For example: `api_call(url="https://example.com")` or `find_in_document(query="what is the landmark for Paris?")`
        """

    def _parse_action(self, action_str: str) -> Tuple[str, Dict[str, Any]]:
        """
        Parses the action and its parameters from the LLM's output string.
        """
        # Regex to find action name and its parameters (e.g., func(param1="value", param2="value"))
        match = re.search(r"(\w+)\((.*)\)", action_str, re.DOTALL)
        
        if not match:
            raise ValueError(f"Could not find a valid action in the string: '{action_str}'. Expected format: action(parameters).")

        action = match.group(1).strip()
        params_str = match.group(2).strip()
        details = {}

        if params_str:
            # Regex to extract key-value pairs (e.g., key="value" or key='value')
            kw_params = re.findall(r"(\w+)\s*=\s*(?P<quote>[<>'\"])(.*?)(?P=quote)", params_str)
            
            if kw_params:
                # Convert extracted key-value tuples into a dictionary
                details = {key: value for key, quote, value in kw_params}
            
            # If no keyword parameters, treat as a single positional argument
            elif '=' not in params_str:
                param_value = params_str.strip("' ")
                if action == 'api_call':
                    details = {"url": param_value}
                elif action == 'text_parser':
                    details = {"pattern": param_value}
                elif action == 'find_in_document':
                    details = {"query": param_value}
                elif action == 'answer':
                    details = {"text": param_value}
        
        return action, details