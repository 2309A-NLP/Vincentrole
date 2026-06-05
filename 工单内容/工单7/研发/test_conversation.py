"""测试多轮对话功能（不加载PDF）"""
from qa_engine.conversation import ConversationManager

def test_conversation():
    print("=" * 60)
    print("测试多轮对话管理器")
    print("=" * 60)
    
    cm = ConversationManager(max_history=3)
    
    # 模拟第1轮对话
    print("\n--- 第1轮 ---")
    q1 = "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？"
    a1 = "根据招股说明书，兴图新科来自军用领域的收入占比超过90%。"
    print(f"用户: {q1}")
    print(f"助手: {a1}")
    cm.add_turn(q1, a1)
    print(f"当前公司: {cm.current_company}")
    
    # 测试指代消解 - "他"（模拟回答中包含实体）
    print("\n--- 第2轮 ---")
    q2 = "他参与的哪个工程荣获了国家科技进步一等奖？"
    # 模拟一个包含具体实体的回答
    a2_mock = "张伟先生参与的XX工程荣获了国家科技进步一等奖。"
    cm.add_turn("上一个问题", a2_mock)  # 先添加一个包含实体的历史
    resolved, info = cm.resolve_references(q2)
    print(f"用户: {q2}")
    print(f"消解: {resolved}")
    print(f"当前实体: {cm.last_entity}")
    
    # 测试指代消解 - "这个公司"
    print("\n--- 第3轮 ---")
    q3 = "这个公司的法定代表人是谁？"
    resolved, info = cm.resolve_references(q3)
    print(f"用户: {q3}")
    print(f"消解: {resolved}")
    a3 = "法定代表人是张三。"
    print(f"助手: {a3}")
    cm.add_turn(q3, a3)
    
    # 测试切换公司
    print("\n--- 第4轮 ---")
    q4 = "那武汉力源信息技术股份有限公司呢？"
    resolved, info = cm.resolve_references(q4)
    print(f"用户: {q4}")
    print(f"消解: {resolved}")
    a4 = "力源信息的法定代表人是李四。"
    print(f"助手: {a4}")
    cm.add_turn(q4, a4)
    print(f"当前公司: {cm.current_company}")
    
    # 显示历史上下文
    print("\n--- 历史上下文 ---")
    print(cm.get_history_context())
    
    print("\n✅ 测试完成")

if __name__ == "__main__":
    test_conversation()
