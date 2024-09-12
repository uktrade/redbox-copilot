import logging
import sys
from operator import itemgetter
from typing import Annotated

from fastapi import Depends
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnablePassthrough,
    chain,
)
from langchain_core.runnables.config import RunnableConfig
from langchain_core.vectorstores import VectorStoreRetriever
from redbox.api.format import format_documents
from redbox.api.runnables import (
    filter_by_elbow,
    make_chat_prompt_from_messages_runnable,
    resize_documents,
)
from redbox.models import ChatRoute, Settings
from redbox.models.errors import NoDocumentSelected
from tiktoken import Encoding

from core_api import dependencies

# === Logging ===

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger()


def build_chat_chain(
    llm: Annotated[ChatLiteLLM, Depends(dependencies.get_llm)],
    tokeniser: Annotated[Encoding, Depends(dependencies.get_tokeniser)],
    env: Annotated[Settings, Depends(dependencies.get_env)],
) -> Runnable:
    return (
        make_chat_prompt_from_messages_runnable(
            system_prompt=env.ai.chat_system_prompt,
            question_prompt=env.ai.chat_question_prompt,
            input_token_budget=env.ai.context_window_size - env.llm_max_tokens,
            tokeniser=tokeniser,
        )
        | llm
        | {
            "response": StrOutputParser(),
            "route_name": RunnableLambda(lambda _: ChatRoute.chat.value),
        }
    )


def build_coach_chain(
    llm: Annotated[ChatLiteLLM, Depends(dependencies.get_llm)],
    tokeniser: Annotated[Encoding, Depends(dependencies.get_tokeniser)],
    env: Annotated[Settings, Depends(dependencies.get_env)],
) -> Runnable:
    coach_prompt = (
        # "You are an AI assistant called Redbox tasked with analysing a users chat history and to improve the users prompts."
        "You are the '@coach' assistant for Redbox users, providing helpful feedback to improve the quality of their prompts."
        # "Your goal is to provide guidance on how the user could improve their prompts in the chat history."
        "When a user uses the @coach keyword,"
        "you should analyze the chat history to identify why their prompt(s) may not have achieved the desired result."
        "Your goal is to:\n"
        "1. **Identify issues** with the original prompt. For example: lack of clarity, insufficient context, "
        "overly broad or narrow focus, poorly used keyword route (@chat, @summarise, @search).\n"
        "2. **Offer specific advice** on how the user can improve their prompt (e.g., by providing more context, "
        "being clearer with their goals, or breaking down complex tasks).\n"
        "3. Offer a suggested alternative prompt, written in a way that solves the identified issues. "
        "This suggestion should be in a block quote so users can easily copy and paste it into their chat.\n"
        "4. Only provide advice on a maximum of 2 prompts at a time and do not repeat advice for a prompt.\n"
        # "Please follow these guidelines while giving advice to the user: \n"
        # "- If the user has not mentioned a specific prompt, then generate overall advice for the users prompts."
        # "- Be concise and constructive in your feedback. \n"
        # "1) Avoid repetition,\n"
        # "2) Ensure the advice is easy to understand,\n"
        # "3) Maintain the original context and meaning,\n"
        # "4) Only generate advice for a single prompt at a time,\n"
        # "5) If a user hasn't specific a prompt, produce advice for any of the users prompts,\n"
        # "6) Do not generate advice for a prompt again unless the user asks for you to generate it for that prompt.\n"
        # "7) If you don't know any advice which can improve the prompt, just say that you don't know, don't try to make up an answer. \n"
        "**Embedded Guidance** (use this when providing feedback):"
        "- Encourage users to be clear, specific, and concise in their instructions. "
        "Suggest including context (including about your task and function/profession-specific terminology) "
        "or examples where necessary but avoid overly constraining the model.\n"
        "- Remind users of what Redbox can do. \n"  # TODO add this context
        "- Advise users of better keyword routes for their prompts, if the appropriate keyword route was not used.\n"  # TODO improve
        "- Remind users that perfect grammar is not required, but clarity is key.\n"
        "- Advise breaking down complex tasks into manageable steps."
        "- For fact-based requests, suggest the user ask for specific details or verification of information."
        "- Encourage iterative prompting—if the first result isn’t right, refine the question and try again."
        "- Advise on specific criteria that can improve a prompt for the used specific keyword route chat, summarise or search."  # TODO imporve
        "- Advise the user that they can also request to express the search results confidence level and explain the reasoning "
        "for the result. When using the search keyword.\n"
        "- If the user is trying to compare multiple documents, remind them to clearly outline the criteria for comparision and to "
        "ask Redbox to create a structured format (e.g., a table or bullet-point list) to present the comparative analysis."
        "- If the user may think that Redbox has output an incorrect response, "
        "remind the user that they can ask for Redbox to double-check its work and explain its reasoning. "
        "Also remind the user that they can also rephrase the question or "
        "provide additional context to see if it leads to a different result and to "
        "Always verify important information against original sources."
        # "Please follow respond with the following structure: \n"
        # "Quote the prompt you are suggesting to improve,"
        # "Advice in bullet points, an exlpantation and then an example of a prompt the user can use.\n"
        "**Format for Response:**\n"
        "**Orgininal Prompt:** Quote the orgininal prompt, if referencing an orginal prompt. Do this for a maximum of two prompts\n"
        "1. **Identify the issue:** Briefly describe why the original prompt might not work well,\n"
        "2. **Actionable advice:** Suggest concrete improvements,\n"
        "3. **Rewritten prompt suggestion:** Provide a better version of the prompt in a block quote format. "
        "Remember to include the relevant @ keyword that the user should use."
        "**Example Interaction:**\n"
        "**User:** '@coach That wasn't specific enough. How do I make it better?' \n"
        "**@coach Response:** \n"
        "**Orgininal Prompt:**\n [quote of the orignal prompt] \n"
        "1. **Issue:** \n The prompt may have been too vague,"
        "lacking specific details that could guide Redbox in generating a more targeted response.\n"
        "2. **Advice:** \n To improve the prompt, try providing additional context about the task, "
        "clarifying the type of response you expect, and narrowing down the scope to avoid general answers.\n"
        "3. **Suggested Prompt:**\n"
        "> 'Please provide a more specific answer by focusing on [insert specific aspect] and "
        "explain how it relates to [insert context]. If possible, provide examples to clarify your explanation.\n'"
    )

    """
    Response:

    Your Prompt Analysis:
        The prompt is too vague and lacks context regarding what specific assistance is required from Redbox.

    Suggestions for Improvement:
        Be more specific about the task you want Redbox to help with. Include details relevant to your field or the type of response you need.

    Revised Prompt:

    "Please help me outline the steps required to effectively analyze this policy document, focusing specifically on its economic implications."

    """

    return (
        make_chat_prompt_from_messages_runnable(
            system_prompt=coach_prompt,
            question_prompt=env.ai.chat_question_prompt,  # "How can I improve one of my previous prompts?\n=========\n Response:",  # ,
            input_token_budget=env.ai.context_window_size - env.llm_max_tokens,
            tokeniser=tokeniser,
        )
        | llm
        | {
            "response": StrOutputParser(),
            "route_name": RunnableLambda(lambda _: ChatRoute.coach.value),
        }
    )


def build_chat_with_docs_chain(
    llm: Annotated[ChatLiteLLM, Depends(dependencies.get_llm)],
    all_chunks_retriever: Annotated[
        BaseRetriever, Depends(dependencies.get_all_chunks_retriever)
    ],
    tokeniser: Annotated[Encoding, Depends(dependencies.get_tokeniser)],
    env: Annotated[Settings, Depends(dependencies.get_env)],
) -> Runnable:
    def make_document_context():
        return all_chunks_retriever | resize_documents(env.ai.stuff_chunk_max_tokens)

    @chain
    def map_operation(input_dict):
        system_map_prompt = env.ai.map_system_prompt
        prompt_template = PromptTemplate.from_template(env.ai.map_question_prompt)

        formatted_map_question_prompt = prompt_template.format(
            question=input_dict["question"]
        )

        map_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_map_prompt),
                ("human", formatted_map_question_prompt + env.ai.map_document_prompt),
            ]
        )

        documents = input_dict["documents"]

        map_summaries = (map_prompt | llm | StrOutputParser()).batch(
            documents,
            config=RunnableConfig(max_concurrency=env.ai.map_max_concurrency),
        )

        summaries = " ; ".join(map_summaries)
        input_dict["summaries"] = summaries
        return input_dict

    @chain
    def chat_with_docs_route(input_dict: dict):
        log.info("Length documents: %s", len(input_dict["documents"]))
        if len(input_dict["documents"]) == 1:
            return RunnablePassthrough.assign(
                formatted_documents=(
                    RunnablePassthrough() | itemgetter("documents") | format_documents
                )
            ) | {
                "response": make_chat_prompt_from_messages_runnable(
                    system_prompt=env.ai.chat_with_docs_system_prompt,
                    question_prompt=env.ai.chat_with_docs_question_prompt,
                    input_token_budget=env.ai.context_window_size - env.llm_max_tokens,
                    tokeniser=tokeniser,
                )
                | llm
                | StrOutputParser(),
                "route_name": RunnableLambda(lambda _: ChatRoute.chat_with_docs.value),
            }

        elif len(input_dict["documents"]) > 1:
            return (
                map_operation
                | RunnablePassthrough.assign(
                    formatted_documents=(
                        RunnablePassthrough()
                        | itemgetter("documents")
                        | format_documents
                    )
                )
                | {
                    "response": make_chat_prompt_from_messages_runnable(
                        system_prompt=env.ai.chat_with_docs_reduce_system_prompt,
                        question_prompt=env.ai.chat_with_docs_reduce_question_prompt,
                        input_token_budget=env.ai.context_window_size
                        - env.llm_max_tokens,
                        tokeniser=tokeniser,
                    )
                    | llm
                    | StrOutputParser(),
                    "route_name": RunnableLambda(
                        lambda _: ChatRoute.chat_with_docs.value
                    ),
                }
            )

        else:
            raise NoDocumentSelected

    return (
        RunnablePassthrough.assign(documents=make_document_context())
        | chat_with_docs_route
    )


def build_retrieval_chain(
    llm: Annotated[ChatLiteLLM, Depends(dependencies.get_llm)],
    retriever: Annotated[
        VectorStoreRetriever, Depends(dependencies.get_parameterised_retriever)
    ],
    tokeniser: Annotated[Encoding, Depends(dependencies.get_tokeniser)],
    env: Annotated[Settings, Depends(dependencies.get_env)],
) -> Runnable:
    return (
        RunnablePassthrough.assign(documents=retriever)
        | RunnablePassthrough.assign(
            formatted_documents=(
                RunnablePassthrough() | itemgetter("documents") | format_documents
            )
        )
        | {
            "response": make_chat_prompt_from_messages_runnable(
                system_prompt=env.ai.retrieval_system_prompt,
                question_prompt=env.ai.retrieval_question_prompt,
                input_token_budget=env.ai.context_window_size - env.llm_max_tokens,
                tokeniser=tokeniser,
            )
            | llm
            | StrOutputParser(),
            "source_documents": itemgetter("documents"),
            "route_name": RunnableLambda(lambda _: ChatRoute.search.value),
        }
    )


def build_condense_retrieval_chain(
    llm: Annotated[ChatLiteLLM, Depends(dependencies.get_llm)],
    retriever: Annotated[
        VectorStoreRetriever, Depends(dependencies.get_parameterised_retriever)
    ],
    tokeniser: Annotated[Encoding, Depends(dependencies.get_tokeniser)],
    env: Annotated[Settings, Depends(dependencies.get_env)],
) -> Runnable:
    def route(input_dict: dict):
        if len(input_dict["chat_history"]) > 0:
            return RunnablePassthrough.assign(
                question=make_chat_prompt_from_messages_runnable(
                    system_prompt=env.ai.condense_system_prompt,
                    question_prompt=env.ai.condense_question_prompt,
                    input_token_budget=env.ai.context_window_size - env.llm_max_tokens,
                    tokeniser=tokeniser,
                )
                | llm
                | StrOutputParser()
            )
        else:
            return RunnablePassthrough()

    return (
        RunnableLambda(route)
        | RunnablePassthrough.assign(
            documents=retriever | filter_by_elbow(enabled=env.ai.elbow_filter_enabled)
        )
        | RunnablePassthrough.assign(
            formatted_documents=(
                RunnablePassthrough() | itemgetter("documents") | format_documents
            )
        )
        | {
            "response": make_chat_prompt_from_messages_runnable(
                system_prompt=env.ai.retrieval_system_prompt,
                question_prompt=env.ai.retrieval_question_prompt,
                input_token_budget=env.ai.context_window_size - env.llm_max_tokens,
                tokeniser=tokeniser,
            )
            | llm
            | StrOutputParser(),
            "source_documents": itemgetter("documents"),
            "route_name": RunnableLambda(lambda _: ChatRoute.search.value),
        }
    )


def build_summary_chain(
    llm: Annotated[ChatLiteLLM, Depends(dependencies.get_llm)],
    all_chunks_retriever: Annotated[
        BaseRetriever, Depends(dependencies.get_all_chunks_retriever)
    ],
    tokeniser: Annotated[Encoding, Depends(dependencies.get_tokeniser)],
    env: Annotated[Settings, Depends(dependencies.get_env)],
) -> Runnable:
    def make_document_context():
        return (
            all_chunks_retriever
            | resize_documents(env.ai.stuff_chunk_max_tokens)
            | RunnableLambda(lambda docs: [d.page_content for d in docs])
        )

    # Stuff chain now missing the RunnabeLambda to format the chunks
    stuff_chain = (
        make_chat_prompt_from_messages_runnable(
            system_prompt=env.ai.summarisation_system_prompt,
            question_prompt=env.ai.summarisation_question_prompt,
            input_token_budget=env.ai.context_window_size - env.llm_max_tokens,
            tokeniser=tokeniser,
        )
        | llm
        | {
            "response": StrOutputParser(),
            "route_name": RunnableLambda(lambda _: ChatRoute.summarise.value),
        }
    )

    @chain
    def map_operation(input_dict):
        system_map_prompt = env.ai.map_system_prompt
        prompt_template = PromptTemplate.from_template(env.ai.map_question_prompt)

        formatted_map_question_prompt = prompt_template.format(
            question=input_dict["question"]
        )

        map_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_map_prompt),
                ("human", formatted_map_question_prompt + env.ai.map_document_prompt),
            ]
        )

        documents = input_dict["documents"]

        map_summaries = (map_prompt | llm | StrOutputParser()).batch(
            documents,
            config=RunnableConfig(max_concurrency=env.ai.map_max_concurrency),
        )

        summaries = " ; ".join(map_summaries)
        input_dict["summaries"] = summaries
        return input_dict

    map_reduce_chain = (
        map_operation
        | make_chat_prompt_from_messages_runnable(
            system_prompt=env.ai.reduce_system_prompt,
            question_prompt=env.ai.reduce_question_prompt,
            input_token_budget=env.ai.context_window_size - env.llm_max_tokens,
            tokeniser=tokeniser,
        )
        | llm
        | {
            "response": StrOutputParser(),
            "route_name": RunnableLambda(
                lambda _: ChatRoute.map_reduce_summarise.value
            ),
        }
    )

    @chain
    def summarisation_route(input_dict):
        if len(input_dict["documents"]) == 1:
            return stuff_chain

        elif len(input_dict["documents"]) > 1:
            return map_reduce_chain

        else:
            raise NoDocumentSelected

    return (
        RunnablePassthrough.assign(documents=make_document_context())
        | summarisation_route
    )


def build_static_response_chain(prompt_template, route_name) -> Runnable:
    return RunnablePassthrough.assign(
        response=(
            ChatPromptTemplate.from_template(prompt_template)
            | RunnableLambda(lambda p: p.messages[0].content)
        ),
        source_documents=RunnableLambda(lambda _: []),
        route_name=RunnableLambda(lambda _: route_name.value),
    )
