from graphrag.query.structured_search.global_search.community_context import GlobalCommunityContext
from graphrag.query.indexer_adapters import read_indexer_entities, read_indexer_reports
from graphrag.query.structured_search.global_search.search import GlobalSearch
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.typing import OpenaiApiType
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import tiktoken
import asyncio
import os

load_dotenv()

st.set_page_config(page_title="AI Search Assistant", layout="wide")
st.title("RAG based AI Assistant Using GraphRAG")

if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def initialize_search_engine():
    api_key = os.getenv("GRAPHRAG_API_KEY")
    llm_model = "gpt-4o"
    llm = ChatOpenAI(
        api_key=api_key,
        model=llm_model,
        api_type=OpenaiApiType.OpenAI,
        max_retries=20,
    )
    
    token_encoder = tiktoken.get_encoding("cl100k_base")

    INPUT_DIR = "output"
    COMMUNITY_REPORT_TABLE = "create_final_community_reports"
    ENTITY_TABLE = "create_final_nodes"
    ENTITY_EMBEDDING_TABLE = "create_final_entities"
    COMMUNITY_LEVEL = 0
    
    entity_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_TABLE}.parquet")
    report_df = pd.read_parquet(f"{INPUT_DIR}/{COMMUNITY_REPORT_TABLE}.parquet")
    entity_embedding_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_EMBEDDING_TABLE}.parquet")
    
    reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)
    entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)
    
    context_builder = GlobalCommunityContext(
        community_reports=reports,
        entities=entities,
        token_encoder=token_encoder,
    )
    
    context_builder_params = {
        "use_community_summary": False,
        "shuffle_data": True,
        "include_community_rank": True,
        "min_community_rank": 0,
        "community_rank_name": "rank",
        "include_community_weight": True,
        "community_weight_name": "occurrence weight",
        "normalize_community_weight": True,
        "max_tokens": 3_000,
        "context_name": "Reports",
    }
    
    map_llm_params = {
        "max_tokens": 1000,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    
    reduce_llm_params = {
        "max_tokens": 2000,
        "temperature": 0.0,
    }
    
    search_engine = GlobalSearch(
        llm=llm,
        context_builder=context_builder,
        token_encoder=token_encoder,
        max_data_tokens=12_000,
        map_llm_params=map_llm_params,
        reduce_llm_params=reduce_llm_params,
        allow_general_knowledge=False,
        json_mode=True,
        context_builder_params=context_builder_params,
        concurrent_coroutines=10,
        response_type="multiple-page report",
    )
    
    return search_engine

search_engine = initialize_search_engine()

async def get_search_results(query: str):
    result = await search_engine.asearch(query)
    return result

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        message_placeholder.text("Searching and analyzing data...")
        
        try:
            result = asyncio.run(get_search_results(prompt))
            response = result.response
            
            context_info = f"\n\nSources consulted: {len(result.context_data['reports'])} reports"
            full_response = f"{response}\n{context_info}"
            
            message_placeholder.markdown(full_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            message_placeholder.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})

with st.sidebar:
    st.title("About this Graphrag AI Search Assistant")
    st.markdown("""
    This AI Search Assistant helps you search and analyze data from the knowledge base using microsoft graphrag.
    
    - Ask questions in natural language
    - Get detailed responses based on available data
    - See sources consulted for each response
    """)