from pathlib import Path
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.clickhouse import ClickHouseVectorStore
from ch_client import get_ch_client

# ========================= CONFIG =========================
DOCS_ROOT = Path("./Chart.js")   # Path to cloned Chart.js repo

OPENAI_EMBED_MODEL = "text-embedding-3-large"

CH_DATABASE = "rag_knowledge"
CH_TABLE = "chartjs_docs_v1"      # Separate table for Chart.js

# =========================================================

def should_include_file(file_path: Path) -> bool:
    str_path = str(file_path).lower()
    if not str_path.endswith(('.md', '.mdx')):
        return False

    exclude = ["node_modules", "test", "dist", "coverage", "scripts", "types", 
               "CHANGELOG", "CONTRIBUTING", "LICENSE", ".github", "README"]
    if any(ex in str_path for ex in exclude):
        return False

    include_dirs = [
        "docs/charts", "docs/configuration", "docs/general", "docs/axes", 
        "docs/developers", "docs/samples", "docs/getting-started", "docs/api"
    ]
    
    for d in include_dirs:
        if d in str_path:
            return True
    return False


def load_filtered_documents(docs_root: Path):
    print("🔍 Loading and filtering Chart.js documents...")
    documents = []
    count = 0

    for file_path in sorted(docs_root.rglob("*")):
        if file_path.is_file() and should_include_file(file_path):
            try:
                content = file_path.read_text(encoding="utf-8")
                rel_path = file_path.relative_to(docs_root)
                
                metadata = {
                    "file_path": str(rel_path),
                    "source": "chartjs-docs",
                    "title": file_path.stem.replace("-", " ").replace("_", " ").title(),
                    "url": f"https://www.chartjs.org/docs/latest/{rel_path.with_suffix('')}"
                }
                
                documents.append(Document(text=content, metadata=metadata))
                count += 1
                if count % 100 == 0:
                    print(f"   Loaded {count} documents...")
            except Exception as e:
                print(f"⚠️ Failed to read {file_path}: {e}")
    
    print(f"✅ Loaded {len(documents)} Chart.js documents.")
    return documents


# ========================= MAIN =========================
if __name__ == "__main__":
    if not DOCS_ROOT.exists():
        print("❌ Chart.js folder not found!")
        print("   Please run: git clone https://github.com/chartjs/Chart.js.git")
        exit(1)

    # 1. Load documents
    documents = load_filtered_documents(DOCS_ROOT)

    # 2. Setup ClickHouse
    print("🔌 Connecting to ClickHouse...")
    client = get_ch_client()
    client.command(f"CREATE DATABASE IF NOT EXISTS {CH_DATABASE}")

    vector_store = ClickHouseVectorStore(
        clickhouse_client=client,
        table=CH_TABLE,
        database=CH_DATABASE,
        dimension=3072 if "large" in OPENAI_EMBED_MODEL else 1536,
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 3. Embed + Store
    print("🚀 Starting embedding and ingestion...")
    embed_model = OpenAIEmbedding(model=OPENAI_EMBED_MODEL)

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )

    print("\n🎉 SUCCESS! Chart.js Knowledge Base is ready!")
    print(f"   Database : {CH_DATABASE}")
    print(f"   Table    : {CH_TABLE}")
    print(f"   Documents: {len(documents)}")
    print(f"   Embedding: {OPENAI_EMBED_MODEL}")