from pathlib import Path
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.clickhouse import ClickHouseVectorStore
import clickhouse_connect
import os

# ========================= CONFIG =========================
DOCS_ROOT = Path("./clickhouse-docs")

OPENAI_EMBED_MODEL = "text-embedding-3-large"   # Best quality (or use "text-embedding-3-small" for cheaper)

CH_HOST = "localhost"
CH_PORT = 8123
CH_USER = "admin"
CH_PASSWORD = "hiffiofsuperlabs"                    # Fill if needed
CH_DATABASE = "rag_knowledge"
CH_TABLE = "clickhouse_docs_v1"     # Vector table name

# =========================================================

def should_include_file(file_path: Path) -> bool:
    str_path = str(file_path).lower()
    if not str_path.endswith(('.md', '.mdx')):
        return False

    exclude = ["_snippets", "_clients", "home_links", "i18n", "contribute", "static", 
               "plugins", "scripts", "sidebars", "docusaurus.config", "README.md", 
               ".github", "node_modules", "docs/getting-started/example-datasets"]
    if any(ex in str_path for ex in exclude):
        return False

    include_dirs = ["docs/sql-reference", "docs/concepts", "docs/best-practices", 
                    "docs/managing-data", "docs/guides", "docs/data-modeling", 
                    "docs/operations_", "docs/tips-and-tricks", "docs/troubleshooting", 
                    "docs/getting-started", "knowledgebase"]
    
    for d in include_dirs:
        if d in str_path:
            return True
    
    # Root important files
    if file_path.parent.name == "docs":
        if file_path.name in {"intro.md", "tutorial.md", "introduction-index.md", "deployment-modes.md"}:
            return True
    return False


def load_filtered_documents(docs_root: Path):
    print("🔍 Loading and filtering documents...")
    documents = []
    count = 0

    for file_path in sorted(docs_root.rglob("*")):
        if file_path.is_file() and should_include_file(file_path):
            try:
                content = file_path.read_text(encoding="utf-8")
                rel_path = file_path.relative_to(docs_root)
                
                metadata = {
                    "file_path": str(rel_path),
                    "source": "clickhouse-docs",
                    "title": file_path.stem.replace("-", " ").replace("_", " ").title(),
                    "url": f"https://clickhouse.com/docs/{rel_path.with_suffix('')}"
                }
                
                documents.append(Document(text=content, metadata=metadata))
                count += 1
                if count % 300 == 0:
                    print(f"   Loaded {count} documents...")
            except Exception as e:
                print(f"⚠️  Failed to read {file_path}: {e}")
    
    print(f"✅ Loaded {len(documents)} documents for ingestion.")
    return documents


# ========================= MAIN =========================
if __name__ == "__main__":
    if not DOCS_ROOT.exists():
        print("❌ clickhouse-docs folder not found!")
        print("   Run: git clone https://github.com/ClickHouse/clickhouse-docs.git")
        exit(1)

    # 1. Load filtered docs
    documents = load_filtered_documents(DOCS_ROOT)

    # 2. Setup ClickHouse connection
    print("🔌 Connecting to ClickHouse...")
    client = clickhouse_connect.get_client(
        host=CH_HOST, port=CH_PORT, 
        username=CH_USER, password=CH_PASSWORD
    )
    client.command(f"CREATE DATABASE IF NOT EXISTS {CH_DATABASE}")

    vector_store = ClickHouseVectorStore(
        clickhouse_client=client,
        table=CH_TABLE,
        database=CH_DATABASE,
        dimension=3072 if "large" in OPENAI_EMBED_MODEL else 1536,
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 3. Embed and store
    print("🚀 Starting embedding and ingestion into ClickHouse...")
    embed_model = OpenAIEmbedding(model=OPENAI_EMBED_MODEL)

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )

    print("\n🎉 SUCCESS! ClickHouse RAG Knowledge Base is ready!")
    print(f"   Database : {CH_DATABASE}")
    print(f"   Table    : {CH_TABLE}")
    print(f"   Documents: {len(documents)}")
    print(f"   Embedding: {OPENAI_EMBED_MODEL}")