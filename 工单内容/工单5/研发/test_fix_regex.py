import re

# 测试修复后的正则
pattern = r'[张李王刘陈杨黄赵周吴徐孙马朱胡郭何林罗高梁郑谢宋唐许邓冯韩曹曾彭萧蔡潘田董袁于余叶蒋杜苏魏程吕丁沈任姚卢傅钟姜崔谭廖范汪陆金石戴贾韦夏邱方侯邹熊孟秦白江阎薛尹段雷龙史陶贺顾毛郝龚邵万钱严赖覃康洪][\u4e00-\u9fa5]{1,3}'
text = '程先生担任董事长兼总经理'
matches = re.findall(pattern, text)
print(f'匹配结果: {matches}')
print('正则OK - 无报错')

# 测试完整模块导入
import sys
sys.path.insert(0, '.')
from qa_engine.conversation import ConversationManager
cm = ConversationManager()
print('ConversationManager 导入成功')

# 测试指代消解
cm.current_company = '武汉兴图新科电子股份有限公司'
cm.last_entity = '程先生'
resolved, info = cm.resolve_references('这个公司的法定代表人是谁？')
print(f'消解: {resolved}')
print(f'信息: {info}')
