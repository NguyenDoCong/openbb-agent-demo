import logging
import json
import asyncio
from datetime import datetime
from typing import Any, Callable, Dict, List, Union
import os
import google.generativeai as genai
from langchain.vectorstores import VectorStore

from openbb_agents.models import (
    AnsweredSubQuestion,
    SubQuestion,
)
from openbb_agents.prompts import (
    FINAL_RESPONSE_PROMPT_TEMPLATE,
    GENERATE_SUBQUESTION_SYSTEM_PROMPT_TEMPLATE,
    SUBQUESTION_ANSWER_PROMPT,
    TOOL_SEARCH_PROMPT_TEMPLATE,
)

logger = logging.getLogger(__name__)

# Configure Gemini API
gemini_api_key = os.environ.get("GEMINI_API_KEY")
if hasattr(gemini_api_key, "get_secret_value"):
    gemini_api_key = gemini_api_key.get_secret_value() # type: ignore
    
genai.configure(api_key=gemini_api_key)

class GeminiChatModel:
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
    
    def generate_content(self, messages: List[Dict], tools: List[Callable] = None) -> str:
        """Generate content using Gemini API, handling simple function calls internally."""
        try:
            prompt = self._convert_messages_to_prompt(messages)
            
            if not tools:
                # If no tools are provided, the response will always be text.
                response = self.model.generate_content(prompt)
                return response.text

            # When tools are provided, make the API call.
            tool_configs = self._convert_tools_to_gemini_format(tools)
            response = self.model.generate_content(
                prompt,
                tools=tool_configs,
                tool_config={'function_calling_config': {'mode': 'ANY'}}
            )

            # Try to access .text. If it fails, we know a function call is present.
            try:
                return response.text
            except ValueError:
                # This block executes ONLY if response.text raised a ValueError,
                # which indicates a function call is pending.
                logger.info("Model responded with a function call request.")

                # Safeguard: ensure the function call part exists
                if not (response.parts and response.parts[0].function_call):
                    raise ValueError("Response has no text and no recognizable function call.")

                # Extract the function call details
                function_call = response.parts[0].function_call
                tool_name = function_call.name
                
                # Find the corresponding callable function from the provided tools
                tool_to_call = next((t for t in tools if t.__name__ == tool_name), None)

                if tool_to_call is None:
                    raise ValueError(f"Model requested to call an unknown tool: '{tool_name}'")

                # Execute the function with the arguments provided by the model
                tool_args = dict(function_call.args)
                logger.info(f"Executing tool '{tool_name}' with args: {tool_args}")
                result = tool_to_call(**tool_args)
                
                # Return the string result of the tool's execution.
                # This works perfectly for `search_tools` because the output of
                # `llm_query_tool_index` is the exact text we need.
                return str(result)

        except Exception as e:
            logger.error(f"Error generating content: {e}")
            # Re-raise the exception to be handled by the agent's main loop
            raise
    
    async def generate_content_async(self, messages: List[Dict], tools: List[Callable] = None) -> str:
        """Async version of generate_content"""
        # Run the sync version in a thread pool
        return await asyncio.get_event_loop().run_in_executor(
            None, self.generate_content, messages, tools
        )
    
    def _convert_messages_to_prompt(self, messages: List[Dict]) -> str:
        """Convert message format to Gemini prompt"""
        prompt_parts = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                if role == 'system':
                    prompt_parts.append(f"System: {content}")
                elif role == 'user':
                    prompt_parts.append(f"User: {content}")
                elif role == 'assistant':
                    prompt_parts.append(f"Assistant: {content}")
            else:
                # Handle string messages
                prompt_parts.append(str(msg))
        return "\n\n".join(prompt_parts)
    
    def _convert_tools_to_gemini_format(self, tools: List[Callable]) -> List[Dict]:
        """Convert callable functions to Gemini tool format"""
        tool_configs = []
        for tool in tools:
            # Inspect the signature of the tool to get its parameters
            import inspect
            signature = inspect.signature(tool)
            parameters = {}
            required_params = []

            for name, param in signature.parameters.items():
                parameters[name] = {"type": "string"} # Assuming all tool params are strings for simplicity
                if param.default == inspect.Parameter.empty:
                    required_params.append(name)

            tool_config = {
                "function_declarations": [{
                    "name": tool.__name__,
                    "description": tool.__doc__ or f"Function {tool.__name__}",
                    "parameters": {
                        "type": "object",
                        "properties": parameters,
                        "required": required_params
                    }
                }]
            }
            tool_configs.append(tool_config)
        return tool_configs


def generate_final_answer(
    user_query: str,
    answered_subquestions: list[AnsweredSubQuestion],
) -> str:
    model = GeminiChatModel("gemini-2.0-flash")
    
    # Format the prompt
    prompt = FINAL_RESPONSE_PROMPT_TEMPLATE.format(
        user_query=user_query,
        answered_subquestions=answered_subquestions
    )
    
    messages = [{"role": "user", "content": prompt}]
    return model.generate_content(messages)


async def agenerate_final_answer(
    user_query: str,
    answered_subquestions: list[AnsweredSubQuestion],
) -> str:
    model = GeminiChatModel("gemini-2.0-flash")
    
    # Format the prompt
    prompt = FINAL_RESPONSE_PROMPT_TEMPLATE.format(
        user_query=user_query,
        answered_subquestions=answered_subquestions
    )
    
    messages = [{"role": "user", "content": prompt}]
    return await model.generate_content_async(messages)


def generate_subquestion_answer(
    user_query: str,
    subquestion: SubQuestion,
    dependencies: list[AnsweredSubQuestion],
    tools: list[Callable],
) -> AnsweredSubQuestion:
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model = GeminiChatModel("gemini-2.0-flash")
    
    messages = [
        {"role": "system", "content": SUBQUESTION_ANSWER_PROMPT},
        {"role": "user", "content": f"""
            User Query: {user_query}
            Subquestion: {subquestion.question}
            Dependencies: {dependencies}
            Current DateTime: {current_datetime}
        """}
    ]
    
    answer = None
    max_iterations = 5
    iteration = 0
    
    while not answer and iteration < max_iterations:
        try:
            if tools:
                # Handle function calling
                response = model.generate_content(messages, tools)
                
                # Check if response contains function calls
                if _contains_function_calls(response):
                    # Execute function calls and add results to messages
                    function_results = _execute_function_calls(response, tools)
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": f"Function results: {function_results}"})
                else:
                    answer = response
            else:
                answer = model.generate_content(messages)
                
        except Exception as e:
            logger.error(f"Error in subquestion answering: {e}")
            break
            
        iteration += 1
    
    if not answer:
        answer = "Unable to generate answer after maximum iterations."
    
    return AnsweredSubQuestion(subquestion=subquestion, answer=answer)


async def agenerate_subquestion_answer(
    user_query: str,
    subquestion: SubQuestion,
    dependencies: list[AnsweredSubQuestion],
    tools: list[Callable],
) -> AnsweredSubQuestion:
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model = GeminiChatModel("gemini-2.0-flash")
    
    messages = [
        {"role": "system", "content": SUBQUESTION_ANSWER_PROMPT},
        {"role": "user", "content": f"""
User Query: {user_query}
Subquestion: {subquestion.question}
Dependencies: {dependencies}
Current DateTime: {current_datetime}
        """}
    ]
    
    answer = None
    max_iterations = 5
    iteration = 0
    
    while not answer and iteration < max_iterations:
        try:
            if tools:
                # Handle function calling
                response = await model.generate_content_async(messages, tools)
                
                # Check if response contains function calls
                if _contains_function_calls(response):
                    # Execute function calls and add results to messages
                    function_results = await _execute_function_calls_async(response, tools)
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": f"Function results: {function_results}"})
                else:
                    answer = response
            else:
                answer = await model.generate_content_async(messages)
                
        except Exception as e:
            logger.error(f"Error in async subquestion answering: {e}")
            break
            
        iteration += 1
    
    if not answer:
        answer = "Unable to generate answer after maximum iterations."
    
    return AnsweredSubQuestion(subquestion=subquestion, answer=answer)


def generate_subquestions_from_query(user_query: str) -> list[SubQuestion]:
    model = GeminiChatModel("gemini-2.0-flash")
    
    messages = [
        {"role": "system", "content": GENERATE_SUBQUESTION_SYSTEM_PROMPT_TEMPLATE},
        {"role": "user", "content": f"# User query\n{user_query}"}
    ]
    
    response = model.generate_content(messages)
    logger.info(f"Response: {response}")
    
    # Parse response to extract SubQuestion objects
    try:
        # Assuming the response is in JSON format or structured text
        subquestions = _parse_subquestions_response(response)
        return subquestions
    except Exception as e:
        logger.error(f"Error parsing subquestions: {e}")
        return []


async def agenerate_subquestions_from_query(user_query: str) -> list[SubQuestion]:
    model = GeminiChatModel("gemini-2.0-flash")
    
    messages = [
        {"role": "system", "content": GENERATE_SUBQUESTION_SYSTEM_PROMPT_TEMPLATE},
        {"role": "user", "content": f"# User query\n{user_query}"}
    ]
    
    response = await model.generate_content_async(messages)
    
    # Parse response to extract SubQuestion objects
    try:
        subquestions = _parse_subquestions_response(response)
        return subquestions
    except Exception as e:
        logger.error(f"Error parsing async subquestions: {e}")
        return []


def search_tools(
    subquestion: SubQuestion,
    tool_vector_index: VectorStore,
    answered_subquestions: list[AnsweredSubQuestion] | None = None,
) -> list[Callable]:
    def llm_query_tool_index(query: str) -> str:
        """Use natural language to search the tool index for tools."""
        logger.info("Searching tool index for: %s", query)
        results = tool_vector_index.similarity_search(query=query, k=4)
        return "\n".join([r.page_content for r in results])

    model = GeminiChatModel("gemini-2.0-flash")
    
    prompt = TOOL_SEARCH_PROMPT_TEMPLATE.format(
        subquestion=subquestion.question,
        answered_subquestions=answered_subquestions
    )
    
    messages = [{"role": "user", "content": prompt}]
    response = model.generate_content(messages, [llm_query_tool_index])
    tool_names = _parse_tool_names_response(response)
    callables = _get_callables_from_tool_search_results(
        tool_vector_index=tool_vector_index, tool_names=tool_names
    )
    return callables


async def asearch_tools(
    subquestion: SubQuestion,
    tool_vector_index: VectorStore,
    answered_subquestions: list[AnsweredSubQuestion] | None = None,
) -> list[Callable]:
    def llm_query_tool_index(query: str) -> str:
        """Use natural language to search the tool index for tools."""
        logger.info("Searching tool index for: %s", query)
        results = tool_vector_index.similarity_search(query=query, k=4)
        return "\n".join([r.page_content for r in results])

    model = GeminiChatModel("gemini-2.0-flash")
    
    prompt = TOOL_SEARCH_PROMPT_TEMPLATE.format(
        subquestion=subquestion.question,
        answered_subquestions=answered_subquestions
    )
    
    messages = [{"role": "user", "content": prompt}]
    response = await model.generate_content_async(messages, [llm_query_tool_index])
    
    tool_names = _parse_tool_names_response(response)
    logger.info(f"Parsed tool names: {tool_names}")
    callables = _get_callables_from_tool_search_results(
        tool_vector_index=tool_vector_index, tool_names=tool_names
    )
    return callables


def _get_callables_from_tool_search_results(
    tool_vector_index: VectorStore,
    tool_names: list[str],
) -> list[Callable]:
    callables = []
    for tool_name in tool_names:
        for doc in tool_vector_index.docstore._dict.values():  # type: ignore
            try:
                logger.debug(f"Checking tool: {doc.metadata['tool_name']}")
            except Exception as e:
                logger.error(f"Error accessing metadata: {e}")
                continue
            if doc.metadata["tool_name"] == tool_name:
                callables.append(doc.metadata["callable"])
                break
            else:
                logger.debug(f"Tool {tool_name} not found in document: {doc.metadata.get('tool_name', 'Unknown')}")

    logger.info("Number of documents in vector store: %d", len(tool_vector_index.docstore._dict))
    logger.info(f"Found {len(callables)} callables for tool names: {tool_names}")

    return callables


def _contains_function_calls(response: str) -> bool:
    """Check if response contains function calls"""
    # This is a simple check - you might need to implement more sophisticated parsing
    return "function_call" in response.lower() or "tool_call" in response.lower()


def _execute_function_calls(response: str, tools: List[Callable]) -> str:
    """Execute function calls mentioned in response"""
    # This is a placeholder - implement actual function call parsing and execution
    results = []
    for tool in tools:
        try:
            # Simple execution - you'll need to parse the actual function calls from response
            result = tool()
            results.append(f"{tool.__name__}: {result}")
        except Exception as e:
            results.append(f"{tool.__name__}: Error - {e}")
    return "; ".join(results)


async def _execute_function_calls_async(response: str, tools: List[Callable]) -> str:
    """Async version of function call execution"""
    return await asyncio.get_event_loop().run_in_executor(
        None, _execute_function_calls, response, tools
    )

import re

def _parse_subquestions_response(response: str) -> list[SubQuestion]:
    """
    Phân tích phản hồi từ LLM để trích xuất các đối tượng SubQuestion.
    Hàm này đã được sửa lỗi để xử lý các cấu trúc JSON khác nhau (cả object và list)
    và ưu tiên sử dụng ID do LLM cung cấp.
    """
    subquestions = []
    try:
        # Bước 1: Trích xuất chuỗi JSON thuần túy từ trong khối markdown
        json_match = re.search(r'```(json)?(.*)```', response, re.DOTALL)
        if json_match:
            cleaned_response = json_match.group(2).strip()
        else:
            cleaned_response = response.strip()

        # Bước 2: Phân tích cú pháp chuỗi JSON
        data = json.loads(cleaned_response)
        
        question_list = []
        # Bước 3: Trích xuất danh sách câu hỏi từ các cấu trúc có thể có
        # Trường hợp 1: JSON là một object có key 'subquestions'
        if isinstance(data, dict) and 'subquestions' in data and isinstance(data['subquestions'], list):
            question_list = data['subquestions']
        # Trường hợp 2: JSON là một list
        elif isinstance(data, list):
            question_list = data

        # Bước 4: Duyệt qua danh sách câu hỏi đã trích xuất và tạo đối tượng
        for i, item in enumerate(question_list):
            if isinstance(item, dict) and 'question' in item:
                # Ưu tiên sử dụng 'id' từ LLM, nếu không có thì dùng 'id' tự tạo
                q_id = item.get('id', i + 1)
                q_text = item.get('question')
                if q_text:
                    subquestions.append(SubQuestion(id=q_id, question=q_text))
            elif isinstance(item, str): # Xử lý trường hợp item là chuỗi thuần túy
                subquestions.append(SubQuestion(id=i + 1, question=item))

    except (json.JSONDecodeError, TypeError) as e:
        logger.error(f"Lỗi khi phân tích phản hồi JSON của câu hỏi phụ: {e}")
        # Cơ chế dự phòng: Coi toàn bộ phản hồi là một câu hỏi duy nhất
        if cleaned_response and not subquestions:
            subquestions.append(SubQuestion(id=1, question=cleaned_response))
            
    except Exception as e:
        logger.error(f"Lỗi không xác định khi phân tích câu hỏi phụ: {e}")
        # Cơ chế dự phòng cuối cùng
        if cleaned_response and not subquestions:
            subquestions.append(SubQuestion(id=1, question=cleaned_response))

    if not subquestions:
        logger.warning("Không thể phân tích bất kỳ câu hỏi phụ nào từ phản hồi của LLM.")

    return subquestions

# def _parse_tool_names_response(response: str) -> List[str]:
#     """Parse response to extract tool names"""
#     # This is a placeholder - implement actual parsing based on your response format
#     tool_names = []
#     try:
#         # Try to parse as JSON first
#         if response.strip().startswith('['):
#             tool_names = json.loads(response)
#         else:
#             # Parse as text format
#             lines = response.strip().split('\n')
#             for line in lines:
#                 line = line.strip()
#                 if line and not line.startswith('#'):
#                     # Remove numbering and bullets
#                     cleaned_line = line.lstrip('0123456789.- ')
#                     if cleaned_line:
#                         tool_names.append(cleaned_line)
#     except Exception as e:
#         logger.error(f"Error parsing tool names response: {e}")

#     # logger.info(f"Parsed tool names: {tool_names}")
        
#     return tool_names

def _parse_tool_names_response(response):
    if isinstance(response, str):
        try:
            data = json.loads(response)
            if isinstance(data, list):
                items = data
            else:
                items = [data]
        except Exception:
            items = [line.strip() for line in response.splitlines()]
    else:
        items = response

    return [
        s for s in items
        if isinstance(s, str)
        and s.count('.') >= 2
        and ':' not in s
        and not s.lower().startswith('example')
        and not s.lower().startswith('output')
        and not s.lower().startswith('>>>')
        and not s.lower().startswith('get ')
        and s != ''
    ]