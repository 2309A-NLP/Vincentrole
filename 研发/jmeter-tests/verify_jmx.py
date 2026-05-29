import xml.etree.ElementTree as ET

tree = ET.parse('roleplay-test-plan.jmx')
root = tree.getroot()
print('Root tag:', root.tag)
print('TestPlan:', root.find('.//TestPlan').get('testname'))

thread_groups = root.findall('.//ThreadGroup')
print(f'Thread groups: {len(thread_groups)}')
for tg in thread_groups:
    name = tg.get('testname')
    threads = tg.find('.//stringProp[@name="ThreadGroup.num_threads"]')
    loops = tg.find('.//stringProp[@name="LoopController.loops"]')
    print(f'  - {name}: threads={threads.text}, loops={loops.text}')

samples = root.findall('.//HTTPSamplerProxy')
print(f'HTTP Samplers: {len(samples)}')
for s in samples:
    path = s.find('.//stringProp[@name="HTTPSampler.path"]')
    method = s.find('.//stringProp[@name="HTTPSampler.method"]')
    name = s.get('testname')
    print(f'  - {name}: {method.text} {path.text}')
