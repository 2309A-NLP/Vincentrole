"""测试RAGSystem能否正常初始化"""
import config
from qa_engine.orchestrator import RAGSystem

llm_config = {
    '提供商': config.LLM_CONFIG.get('提供商', 'deepseek'),
    '模型': config.LLM_CONFIG.get('模型', 'deepseek-v4-flash'),
    'API地址': config.LLM_CONFIG.get('API地址', 'https://api.deepseek.com/v1'),
    'API密钥': config.LLM_CONFIG.get('API密钥', ''),
}

print('Creating RAGSystem...')
try:
    system = RAGSystem(llm_config=llm_config)
    print('SUCCESS: rag_system created')
    print('has embedding_model:', hasattr(system, 'embedding_model'))
    print('has retriever:', hasattr(system, 'retriever'))
except Exception as e:
    import traceback
    print(f'FAILED: {e}')
    traceback.print_exc()
