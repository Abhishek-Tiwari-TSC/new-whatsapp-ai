
import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

ef = ONNXMiniLM_L6_V2()
client = chromadb.PersistentClient(path="./whatsapp_template_db")

for coll_name in ["utility_templates", "marketing_templates"]:
    print(f"\nMigrating {coll_name}...")
    
    # Get old collection (no embedding function — just raw fetch)
    old = client.get_collection(coll_name)
    data = old.get(include=["documents", "metadatas"])
    
    ids = data["ids"]
    docs = data["documents"]
    metas = data["metadatas"]
    
    print(f"  Found {len(ids)} records")
    
    # Delete old and recreate with ONNX
    client.delete_collection(coll_name)
    new = client.create_collection(coll_name, embedding_function=ef)
    
    # Re-add in batches of 100
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        new.add(
            ids=ids[i:i+batch_size],
            documents=docs[i:i+batch_size],
            metadatas=metas[i:i+batch_size] if metas else None
        )
        print(f"  Inserted {min(i+batch_size, len(ids))}/{len(ids)}")
    
        print(f"  ✅ {coll_name} migrated — {new.count()} records")
    
        print("\n✅ Migration complete! Now run py app.py")

