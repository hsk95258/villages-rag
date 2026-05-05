import os
import pandas as pd
from dotenv import load_dotenv
import psycopg2
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

DB_URL = os.getenv("DATABASE_URL")

def fetch_village_data():
    print("Connecting to NeonDB...")
    conn = psycopg2.connect(DB_URL)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT 
            v.name as village_name,
            sd.name as subdistrict_name,
            d.name as district_name,
            s.name as state_name
        FROM "Village" v
        JOIN "SubDistrict" sd ON v."subDistrictId" = sd.id
        JOIN "District" d ON sd."districtId" = d.id
        JOIN "State" s ON d."stateId" = s.id
        LIMIT 200000
    """)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    print(f"Fetched {len(rows)} records")
    return rows

def build_documents(rows):
    print("Building documents...")
    # Group by district for meaningful chunks
    district_map = {}
    for village, subdistrict, district, state in rows:
        key = f"{district}||{state}"
        if key not in district_map:
            district_map[key] = {
                "district": district,
                "state": state,
                "subdistricts": {},
            }
        if subdistrict not in district_map[key]["subdistricts"]:
            district_map[key]["subdistricts"][subdistrict] = []
        district_map[key]["subdistricts"][subdistrict].append(village)

    docs = []
    for key, data in district_map.items():
        district = data["district"]
        state = data["state"]
        subdistricts = data["subdistricts"]
        total_villages = sum(len(v) for v in subdistricts.values())

        content = f"State: {state}\n"
        content += f"District: {district}\n"
        content += f"Total villages: {total_villages}\n"
        content += f"Sub-districts ({len(subdistricts)}): {', '.join(subdistricts.keys())}\n"
        for sd, villages in subdistricts.items():
            content += f"\nSub-district {sd} has {len(villages)} villages including: {', '.join(villages[:10])}"
            if len(villages) > 10:
                content += f" and {len(villages)-10} more"

        docs.append(Document(
            page_content=content,
            metadata={"district": district, "state": state}
        ))

    print(f"Built {len(docs)} district-level documents")
    return docs

def ingest():
    rows = fetch_village_data()
    docs = build_documents(rows)

    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
    )

    print("Creating ChromaDB vector store...")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    print(f"Done. {len(docs)} documents embedded and stored.")

if __name__ == "__main__":
    ingest()