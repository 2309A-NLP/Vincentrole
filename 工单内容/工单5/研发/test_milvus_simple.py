"""测试Milvus连接 - 使用CollectionSchema"""
from pymilvus import MilvusClient, CollectionSchema, DataType, FieldSchema

# 测试连接
uri = "http://127.0.0.1:19530"
client = MilvusClient(uri=uri)
print(f"✅ Milvus连接成功: {uri}")

# 列出collections
collections = client.list_collections()
print(f"现有collections: {collections}")

# 测试创建和删除collection
test_name = "test_collection"
if client.has_collection(test_name):
    client.drop_collection(test_name)
    print(f"已删除旧测试collection: {test_name}")

# 手动定义schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=512),
    FieldSchema(name="chunk_idx", dtype=DataType.INT64),
    FieldSchema(name="page", dtype=DataType.INT64),
    FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=256),
]
schema = CollectionSchema(fields=fields, description="Test collection")

client.create_collection(
    collection_name=test_name,
    schema=schema,
)
print(f"✅ 创建collection成功: {test_name}")

# 插入测试数据（不提供id字段，因为auto_id=True）
data = [
    {"vector": [0.1] * 512, "chunk_idx": 0, "page": 1, "source_file": "test.pdf"},
    {"vector": [0.2] * 512, "chunk_idx": 1, "page": 2, "source_file": "test.pdf"},
]
client.insert(collection_name=test_name, data=data)
print(f"✅ 插入数据成功: {len(data)}条")

# 创建索引
index_params = client.prepare_index_params()
index_params.add_index(
    field_name="vector",
    metric_type="COSINE",
    index_type="IVF_FLAT",
    params={"nlist": 128}
)
client.create_index(collection_name=test_name, index_params=index_params)
print(f"✅ 创建索引成功")

# 加载collection（搜索前必须加载）
client.load_collection(collection_name=test_name)
print(f"✅ 加载collection成功")

# 搜索测试
results = client.search(
    collection_name=test_name,
    data=[[0.15] * 512],
    limit=2,
    output_fields=["chunk_idx", "page"],
)
print(f"✅ 搜索成功: 找到{len(results[0])}条结果")
for hit in results[0]:
    print(f"  - chunk_idx={hit['entity']['chunk_idx']}, score={hit['distance']:.4f}")

# 清理
client.drop_collection(test_name)
print(f"✅ 测试完成，已清理")
