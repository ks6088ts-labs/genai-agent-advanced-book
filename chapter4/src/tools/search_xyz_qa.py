from langchain.tools import tool
from openai import AzureOpenAI
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from src.configs import Settings
from src.custom_logger import setup_logger
from src.models import SearchOutput

# 検索結果の最大取得数
MAX_SEARCH_RESULTS = 3

logger = setup_logger(__name__)


class SearchQueryInput(BaseModel):
    query: str = Field(description="検索クエリ")


@tool(args_schema=SearchQueryInput)
def search_xyz_qa(query: str) -> list[SearchOutput]:
    """
    XYZシステムの過去の質問回答ペアを検索する関数。
    """

    logger.info(f"Searching XYZ QA by query: {query}")

    qdrant_client = QdrantClient("http://localhost:6333")

    settings = Settings()
    openai_client = AzureOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
    )

    logger.info("Generating embedding vector from input query")
    query_vector = (
        openai_client.embeddings.create(input=query, model=settings.azure_openai_model_embedding).data[0].embedding
    )

    search_results = qdrant_client.query_points(
        collection_name="documents", query=query_vector, limit=MAX_SEARCH_RESULTS
    ).points

    logger.info(f"Search results: {len(search_results)} hits")
    outputs = []

    for point in search_results:
        outputs.append(SearchOutput.from_point(point))

    logger.info("Finished searching XYZ QA by query")

    return outputs
