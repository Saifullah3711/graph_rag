from graphrag.query.structured_search.global_search.community_context import GlobalCommunityContext
from graphrag.query.structured_search.local_search.mixed_context import LocalSearchMixedContext
from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.structured_search.global_search.search import GlobalSearch
from graphrag.query.input.loaders.dfs import store_entity_semantic_embeddings
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.vector_stores.lancedb import LanceDBVectorStore
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.indexer_adapters import (
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
import streamlit as st
import pandas as pd
import tiktoken
import asyncio

st.set_page_config(page_title="GraphRAG Assistant", layout="wide")
st.title("GraphRAG Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def initialize_search_engines():
    api_key = st.secrets["general"]["GRAPHRAG_API_KEY"]
    llm_model = "gpt-4o"
    embedding_model = "text-embedding-3-small"
    llm = ChatOpenAI(api_key=api_key, model=llm_model, api_type=OpenaiApiType.OpenAI, max_retries=20)
    token_encoder = tiktoken.get_encoding("cl100k_base")
    text_embedder = OpenAIEmbedding(api_key=api_key, api_type=OpenaiApiType.OpenAI, model=embedding_model, max_retries=20)
    
    INPUT_DIR = "output"
    ENTITY_TABLE = "create_final_nodes"
    ENTITY_EMBEDDING_TABLE = "create_final_entities"
    RELATIONSHIP_TABLE = "create_final_relationships"
    COMMUNITY_REPORT_TABLE = "create_final_community_reports"
    TEXT_UNIT_TABLE = "create_final_text_units"
    LANCEDB_URI = f"{INPUT_DIR}/lancedb"
    COMMUNITY_LEVEL = 2
    
    entity_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_TABLE}.parquet")
    entity_embedding_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_EMBEDDING_TABLE}.parquet")
    relationship_df = pd.read_parquet(f"{INPUT_DIR}/{RELATIONSHIP_TABLE}.parquet")
    report_df = pd.read_parquet(f"{INPUT_DIR}/{COMMUNITY_REPORT_TABLE}.parquet")
    text_unit_df = pd.read_parquet(f"{INPUT_DIR}/{TEXT_UNIT_TABLE}.parquet")
   
    entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)
    relationships = read_indexer_relationships(relationship_df)
    reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)
    text_units = read_indexer_text_units(text_unit_df)
    
    description_embedding_store = LanceDBVectorStore(collection_name="entity_description_embeddings")
    description_embedding_store.connect(db_uri=LANCEDB_URI)
    store_entity_semantic_embeddings(entities=entities, vectorstore=description_embedding_store)
    
    local_context = LocalSearchMixedContext(
        community_reports=reports,
        text_units=text_units,
        entities=entities,
        relationships=relationships,
        entity_text_embeddings=description_embedding_store,
        embedding_vectorstore_key=EntityVectorStoreKey.ID,
        text_embedder=text_embedder,
        token_encoder=token_encoder,
    )
    
    local_context_params = {
        "text_unit_prop": 0.5,
        "community_prop": 0.1,
        "conversation_history_max_turns": 5,
        "conversation_history_user_turns_only": True,
        "top_k_mapped_entities": 10,
        "top_k_relationships": 10,
        "include_entity_rank": True,
        "include_relationship_weight": True,
        "include_community_rank": False,
        "return_candidate_context": False,
        "max_tokens": 12_000,
    }
    
    llm_params = {
        "max_tokens": 2_000,
        "temperature": 0.0,
    }
    
    local_search = LocalSearch(
        llm=llm,
        context_builder=local_context,
        token_encoder=token_encoder,
        llm_params=llm_params,
        context_builder_params=local_context_params,
        response_type="multiple paragraphs",
    )
    
    global_context = GlobalCommunityContext(
        community_reports=reports,
        entities=entities,
        token_encoder=token_encoder,
    )
    
    global_context_params = {
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
    
    global_search = GlobalSearch(
        llm=llm,
        context_builder=global_context,
        token_encoder=token_encoder,
        max_data_tokens=12_000,
        map_llm_params=map_llm_params,
        reduce_llm_params=reduce_llm_params,
        allow_general_knowledge=False,
        json_mode=True,
        context_builder_params=global_context_params,
        concurrent_coroutines=10,
        response_type="multiple-page report",
    )
    
    return local_search, global_search

local_search, global_search = initialize_search_engines()

async def get_search_results(query: str, search_type: str):
    if search_type == "Global Search":
        result = await global_search.asearch(query)
    else:
        result = await local_search.asearch(query)
    return result

with st.sidebar:
    st.title("GraphRAG Settings")
    search_type = st.radio(
        "Select Search Type",
        # ["Local Search","Global Search"],
        ["Local Search"],
        help="""
        Global Search: Best for holistic questions about the entire corpus
        Local Search: Best for specific entity-related questions
        """
    )
    
    st.divider()
    
    st.title("About")
    st.markdown("""
    This GraphRAG Assistant supports two types of search:
    
    **🌍 Global Search**
    - Analyzes entire document corpus
    - Best for general questions
    - Uses community summaries
    - Provides holistic understanding
    
    **🎯 Local Search**
    - Focuses on specific entities
    - Best for detailed questions
    - Uses entity relationships
    - Provides targeted insights
    """)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input(f"Ask a question using {search_type}..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        message_placeholder.text(f"Searching using {search_type}...")
        
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            result = loop.run_until_complete(get_search_results(prompt, search_type))
            response = result.response
            
            if search_type == "Global Search":
                context_info = f"\n\nSources consulted: {len(result.context_data['reports'])} reports"
            else:
                context_info = f"\n\nEntities analyzed: {len(result.context_data['entities'])} | Relationships: {len(result.context_data['relationships'])}"
            
            full_response = f"{response}\n{context_info}"
            
            message_placeholder.markdown(full_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            message_placeholder.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})
