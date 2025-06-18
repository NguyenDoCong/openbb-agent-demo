"""Load OpenBB functions at OpenAI tools for function calling in Langchain"""
import logging
import typing
from typing import Any, Callable

from langchain.schema import Document
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.vectorstores import VectorStore
from langchain_openai import OpenAIEmbeddings
from openbb import obb
from pydantic import BaseModel, SecretStr
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from .models import OpenBBFunctionDescription

logger = logging.getLogger(__name__)


def enable_openbb_llm_mode():
    from openbb import obb

    obb.user.preferences.output_type = "llm"  # type: ignore
    obb.system.python_settings.docstring_sections = ["description", "examples"]  # type: ignore
    obb.system.python_settings.docstring_max_length = 1024  # type: ignore

    import openbb

    openbb.build()


enable_openbb_llm_mode()


def _get_openbb_coverage_providers() -> dict:
    return obb.coverage.providers  # type: ignore


def _get_openbb_user_credentials() -> dict:
    return obb.user.credentials.model_dump()  # type: ignore


def _get_openbb_coverage_command_schemas() -> dict:
    return obb.coverage.command_schemas()  # type: ignore


def get_valid_list_of_providers() -> list[str]:
    credentials = _get_openbb_user_credentials()

    # By default we include yfinance, since it doesn't need a key
    valid_providers = ["yfinance"]
    for name, value in credentials.items():
        if value is not None:
            valid_providers.append(name.split("_api_key")[0].split("_token")[0])
    return valid_providers


# def get_valid_openbb_function_names() -> list[str]:
#     valid_providers = get_valid_list_of_providers()
#     valid_function_names = set()
#     for provider in valid_providers:
#         try:
#             valid_function_names |= set(_get_openbb_coverage_providers()[provider])
#         except KeyError:
#             pass
#     return sorted(list(valid_function_names))

def get_valid_openbb_function_names() -> list[str]:
    valid_providers = get_valid_list_of_providers()
    valid_function_names = set()
    openbb_coverage_providers = {}
    try:
        openbb_coverage_providers = _get_openbb_coverage_providers()
    except Exception as e:
        logger.error(f"Error retrieving OpenBB coverage providers: {e}")
        # Depending on desired behavior, you might want to re-raise or return an empty list here.

    for provider in valid_providers:
        if provider in openbb_coverage_providers:
            try:
                valid_function_names |= set(openbb_coverage_providers[provider])
            except TypeError as e:
                logger.error(f"Error processing functions for provider '{provider}': {e}")
        else:
            logger.warning(f"Provider '{provider}' not found in OpenBB coverage.")

    valid_function_names = {name for name in valid_function_names if "." in name and not name.lower().startswith("output") and not name.lower().startswith("example")}
    # logger.info(f"Valid OpenBB function names: {valid_function_names}")         
    return sorted(list(valid_function_names))


def get_valid_openbb_function_descriptions() -> list[OpenBBFunctionDescription]:
    obb_function_descriptions = []
    for obb_function_name in get_valid_openbb_function_names():
        obb_function_descriptions.append(
            map_name_to_openbb_function_description(obb_function_name)
        )
    return obb_function_descriptions


def map_name_to_openbb_function_description(
    obb_function_name: str,
) -> OpenBBFunctionDescription:
    command_schemas = _get_openbb_coverage_command_schemas()
    dict_ = command_schemas[obb_function_name]
    return OpenBBFunctionDescription(
        name=obb_function_name,
        input_model=dict_["input"],
        output_model=dict_["output"],
        callable=dict_["callable"],
    )


def _get_flat_properties_from_pydantic_model_as_str(model: Any) -> str:
    output_str = ""
    schema_properties = model.schema()["properties"]
    for name, props in schema_properties.items():
        description = props.get("description", "")
        output_str += f"{name}: {description}\n"
    return output_str


def make_vector_index_description(
    openbb_function_description: OpenBBFunctionDescription,
) -> str:
    output_str = ""
    output_str += openbb_function_description.name
    output_str += "\n"
    output_str += openbb_function_description.callable.__doc__
    output_str += "\nOutputs:\n"
    output_str += _get_flat_properties_from_pydantic_model_as_str(
        openbb_function_description.output_model
    )
    return output_str


# def build_vector_index_from_openbb_function_descriptions(
#     openbb_function_descriptions: list[OpenBBFunctionDescription],
# ) -> VectorStore:
#     documents = []
#     for function_description in openbb_function_descriptions:
#         documents.append(
#             Document(
#                 page_content=make_vector_index_description(function_description),
#                 metadata={
#                     "callable": function_description.callable,
#                     "tool_name": function_description.name,
#                 },
#             )
#         )
#     vector_store = FAISS.from_documents(documents, embedding=OpenAIEmbeddings())
#     return vector_store

import os
def build_vector_index_from_openbb_function_descriptions(
    openbb_function_descriptions: list[OpenBBFunctionDescription],
) -> VectorStore:
    """
    Build a vector index from OpenBB function descriptions.
    
    Args:
        openbb_function_descriptions: List of OpenBBFunctionDescription objects containing function descriptions.
        
    Returns:
        VectorStore: FAISS vector store with embedded documents.
    """
    # Convert OpenBBFunctionDescription objects to Document objects
    documents = [
        Document(
            page_content=make_vector_index_description(description),
            metadata={
                "callable": description.callable,
                "tool_name": description.name,
            },
        )
        for description in openbb_function_descriptions
    ]

    if not documents:
        raise ValueError("No documents to build vector index from.")

    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if google_api_key is not None:
        if isinstance(google_api_key, SecretStr):
            google_api_key = google_api_key.get_secret_value()
        google_api_key = str(google_api_key)  # Ensure it's a string
    else:
        raise ValueError("GOOGLE_API_KEY environment variable is not set or is empty")        

    # Use Google's embedding model instead of OpenAI
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=google_api_key
    )
    
    vector_store = FAISS.from_documents(documents, embedding=embeddings)
    logger.info("Vector store length: %d", len(vector_store.docstore._dict))
    return vector_store

def build_openbb_tool_vector_index() -> VectorStore:
    logger.info("Building OpenBB tool vector index...")
    return build_vector_index_from_openbb_function_descriptions(
        get_valid_openbb_function_descriptions()
    )


def _tool_has_unique_name(vector_store: VectorStore, tool: Callable) -> bool:
    for stored_tool in vector_store.docstore._dict.values():
        if stored_tool.metadata["tool_name"] == tool.__name__:
            return False
    return True


def _get_output_type_hint(tool: Callable) -> type | BaseModel | None:
    try:
        return typing.get_type_hints(tool)["return"]
    except KeyError:
        return None


def append_tools_to_vector_index(
    vector_store: VectorStore, tools: list[Callable]
) -> VectorStore:
    logger.info("Adding user-specified tools to vector index...")

    for tool in tools:
        if not _tool_has_unique_name(vector_store=vector_store, tool=tool):
            logger.warning(
                f"Skipping: a tool with name {tool.__name__} already exists in vector index."  # noqa: E501
            )
        else:
            page_content = ""
            page_content += tool.__name__
            page_content += "\n"

            if tool.__doc__:
                page_content += tool.__doc__

            output_type = _get_output_type_hint(tool)
            if output_type is not None:
                page_content += "\nOutput:\n"

                if issubclass(output_type, BaseModel):
                    page_content += _get_flat_properties_from_pydantic_model_as_str(
                        output_type
                    )
                else:
                    page_content += output_type.__name__

            metadata = {
                "callable": tool,
                "tool_name": tool.__name__,
            }
            vector_store.add_documents(
                [Document(page_content=page_content, metadata=metadata)]
            )
    return vector_store
