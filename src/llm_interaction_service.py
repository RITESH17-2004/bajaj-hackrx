import logging
from typing import Dict, Any, List
import json
import re
import asyncio

from src.open_source_llm_engine import OpenSourceLLMEngine

# Prompt for parsing complex instructions into tool calls
INSTRUCTION_PARSING_PROMPT = """
You are an expert at breaking down complex instructions from a document into a sequence of simple tool calls.
Your goal is to create a plan of execution that can be executed by a simple agent to solve the mission described in the document.

**Available Tools:**

*   `api_call(url: str) -> dict`: Makes an API call and returns the JSON response. The `url` parameter must be a valid and complete URL, e.g., `https://register.hackrx.in/api/data`. Ensure the domain name is correct and includes dots (e.g., `register.hackrx.in`). Do not include spaces in the URL.
*   `find_in_document(query: str) -> str`: Searches the document for the answer to a query.
*   `conditional_logic(condition: str, true_action: dict, false_action: dict) -> dict`: Executes different actions based on a condition.

**Your Task:**

Given the following document, create a JSON object that represents the plan of execution.
The JSON object should have a "steps" field, which is a list of tool calls.
Each tool call should have an "action" field and a "details" field.

**Document Text:**
{document_text}

**JSON Plan:**
"""

# Prompt for generating a high-level plan
HIGH_LEVEL_PLAN_PROMPT = """
Given the following document, create a high-level plan to solve the mission. 
The plan should be a list of goals. 
Return the plan as a JSON object with a single key "plan" which is a list of strings.

**Document Text:**
{document_text}

**JSON Plan:**
"""


class LLMInteractionService:
    """
    Provides services for interacting with an LLM, including parsing instructions
    into tool calls and generating high-level plans from document text.
    """
    def __init__(self, llm_engine: OpenSourceLLMEngine):
        self.llm_engine = llm_engine

    async def parse_instructions(self, document_text: str) -> List[Dict[str, Any]]:
        """
        Parses complex instructions from a document into a sequence of tool calls.
        Uses the LLM to understand the mission and generate an executable plan.
        """
        logging.info("Parsing instructions from document using LLM.")

        # Extract relevant parts of the document using the QA pipeline for context
        mission_objective = await self.llm_engine.generate_answer("What is the mission objective?", [({'text': document_text}, 1.0)], {})
        step_by_step_guide = await self.llm_engine.generate_answer("What is the step-by-step guide?", [({'text': document_text}, 1.0)], {})

        # Create a concise prompt with only the essential information for instruction parsing
        prompt = INSTRUCTION_PARSING_PROMPT.format(document_text=f"{mission_objective}\n{step_by_step_guide}")

        try:
            # Generate response from LLM with a timeout
            response = await asyncio.wait_for(self.llm_engine.generate(prompt), timeout=60.0)
            logging.info("LLM response received for instruction parsing.")

            # Sanitize URLs in the LLM response to ensure correctness
            def sanitize_url_match(match):
                scheme = match.group(1)
                domain_part = match.group(2)
                path_part = match.group(3) if match.group(3) else ""

                # Specific fix for "register hackrx in" pattern
                domain_part = domain_part.replace("register hackrx in", "register.hackrx.in")
                domain_part = re.sub(r"register(hackrx\.in)", r"register.\1", domain_part) # Ensure dot

                domain_part = domain_part.replace(" ", "") # Remove any remaining spaces
                path_part = path_part.replace(" ", "") # Remove spaces from path

                return f"{scheme}{domain_part}{path_part}"

            sanitized_response = re.sub(
                r"(https?://)([a-zA-Z0-9\s\.]+)(/[^\s]*)?",
                sanitize_url_match,
                response
            )
            logging.info("LLM response sanitized.")

            # Extract and parse the JSON object from the sanitized response
            json_match = re.search(r"\{{.*\}}", sanitized_response, re.DOTALL)
            if not json_match:
                logging.error("No JSON object found in LLM response for instruction parsing.")
                return []

            parsed_response = json.loads(json_match.group(0))
            return parsed_response.get("steps", [])
        except asyncio.TimeoutError:
            logging.error("LLM request for instruction parsing timed out.")
            return []
        except Exception as e:
            logging.error(f"Failed to parse instructions from LLM response: {e}")
            return []

    async def generate_plan(self, document_text: str) -> List[str]:
        """
        Generates a high-level plan (list of goals) from the document text using the LLM.
        """
        logging.info("Generating high-level plan from document using LLM.")

        prompt = HIGH_LEVEL_PLAN_PROMPT.format(document_text=document_text)

        try:
            response = await asyncio.wait_for(self.llm_engine.generate(prompt), timeout=60.0)
            logging.info("LLM response received for plan generation.")

            # Extract and parse the JSON object from the response
            json_match = re.search(r"\{{{{.*}}}}", response, re.DOTALL)
            if not json_match:
                logging.error("No JSON object found in LLM response for plan generation.")
                return []

            parsed_response = json.loads(json_match.group(0))
            return parsed_response.get("plan", [])
        except asyncio.TimeoutError:
            logging.error("LLM request for plan generation timed out.")
            return []
        except Exception as e:
            logging.error(f"Failed to generate plan from LLM response: {e}")
            return []
