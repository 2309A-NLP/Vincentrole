#!/usr/bin/env python3
"""生成完整版 JMeter 测试计划 JMX 文件"""

import xml.dom.minidom as md
import xml.etree.ElementTree as ET

HOST = "localhost"
PORT = "8080"
PROTO = "http"

root = md.Document()
jmeterTestPlan = root.createElement("jmeterTestPlan")
jmeterTestPlan.setAttribute("version", "1.2")
jmeterTestPlan.setAttribute("properties", "5.0")
jmeterTestPlan.setAttribute("jmeter", "5.6.3")
root.appendChild(jmeterTestPlan)

def make_str(name, value):
    el = root.createElement("stringProp")
    el.setAttribute("name", name)
    el.appendChild(root.createTextNode(str(value)))
    return el

def make_bool(name, value):
    el = root.createElement("boolProp")
    el.setAttribute("name", name)
    el.appendChild(root.createTextNode("true" if value else "false"))
    return el

def make_int(name, value):
    el = root.createElement("intProp")
    el.setAttribute("name", name)
    el.appendChild(root.createTextNode(str(value)))
    return el

def make_elem(name, etype, gui, tclass, tname, enabled="true"):
    el = root.createElement("elementProp")
    el.setAttribute("name", name)
    el.setAttribute("elementType", etype)
    el.setAttribute("guiclass", gui)
    el.setAttribute("testclass", tclass)
    el.setAttribute("testname", tname)
    el.setAttribute("enabled", enabled)
    return el

def make_coll(name):
    el = root.createElement("collectionProp")
    el.setAttribute("name", name)
    return el

def make_header(name, value):
    el = root.createElement("elementProp")
    el.setAttribute("name", "")
    el.setAttribute("elementType", "Header")
    el.appendChild(make_str("Header.name", name))
    el.appendChild(make_str("Header.value", value))
    return el

def make_header_manager(headers, name="HTTP信息头管理器"):
    el = root.createElement("HeaderManager")
    el.setAttribute("guiclass", "HeaderPanel")
    el.setAttribute("testclass", "HeaderManager")
    el.setAttribute("testname", name)
    el.setAttribute("enabled", "true")
    coll = make_coll("HeaderManager.headers")
    for h_name, h_val in headers:
        coll.appendChild(make_header(h_name, h_val))
    el.appendChild(coll)
    return el

def make_json_extractor(ref_names, json_paths, match_nums, defaults, name="JSON提取器"):
    el = root.createElement("JSONPostProcessor")
    el.setAttribute("guiclass", "JSONPostProcessorGui")
    el.setAttribute("testclass", "JSONPostProcessor")
    el.setAttribute("testname", name)
    el.setAttribute("enabled", "true")
    for k, v in [("JSONPostProcessor.referenceNames", ref_names),
                 ("JSONPostProcessor.jsonPathExprs", json_paths),
                 ("JSONPostProcessor.match_numbers", match_nums),
                 ("JSONPostProcessor.defaultValues", defaults)]:
        el.appendChild(make_str(k, v))
    return el

def make_assertion(pattern):
    el = root.createElement("ResponseAssertion")
    el.setAttribute("guiclass", "AssertionGui")
    el.setAttribute("testclass", "ResponseAssertion")
    el.setAttribute("testname", "响应断言")
    el.setAttribute("enabled", "true")
    el.appendChild(make_bool("Assertion.assume_success", False))
    el.appendChild(make_int("Assertion.test_field", 1))
    el.appendChild(make_int("Assertion.test_type", 2))  # 2=contains
    el.appendChild(make_bool("Assertion.force_error", False))
    coll = make_coll("Asserion.test_strings")
    val = root.createElement("stringProp")
    val.setAttribute("name", "-1891868937")
    val.appendChild(root.createTextNode(pattern))
    coll.appendChild(val)
    el.appendChild(coll)
    return el

def make_http_sampler(name, path, method, post_body="", enabled="true"):
    """HTTP采样器，直接带 domain/port/protocol"""
    el = root.createElement("HTTPSamplerProxy")
    el.setAttribute("guiclass", "HttpTestSampleGui")
    el.setAttribute("testclass", "HTTPSamplerProxy")
    el.setAttribute("testname", name)
    el.setAttribute("enabled", enabled)

    args_el = make_elem("HTTPsampler.Arguments", "Arguments",
                         "HTTPArgumentsPanel", "Arguments", "用户定义的变量")
    coll = make_coll("Arguments.arguments")
    if post_body:
        arg = root.createElement("elementProp")
        arg.setAttribute("name", "")
        arg.setAttribute("elementType", "HTTPArgument")
        arg.appendChild(make_bool("HTTPArgument.always_encode", False))
        arg.appendChild(make_str("Argument.value", post_body))
        arg.appendChild(make_str("Argument.metadata", "="))
        arg.appendChild(make_bool("HTTPArgument.use_equals", False))
        arg.appendChild(make_str("Argument.name", ""))
        coll.appendChild(arg)
    args_el.appendChild(coll)
    el.appendChild(args_el)

    el.appendChild(make_str("HTTPSampler.domain", HOST))
    el.appendChild(make_str("HTTPSampler.port", PORT))
    el.appendChild(make_str("HTTPSampler.protocol", PROTO))
    el.appendChild(make_str("HTTPSampler.contentEncoding", "UTF-8"))
    el.appendChild(make_str("HTTPSampler.path", path))
    el.appendChild(make_str("HTTPSampler.method", method))
    if post_body:
        el.appendChild(make_bool("HTTPSampler.postBodyRaw", True))
    el.appendChild(make_bool("HTTPSampler.follow_redirects", True))
    el.appendChild(make_bool("HTTPSampler.auto_redirects", False))
    el.appendChild(make_bool("HTTPSampler.use_keepalive", True))
    el.appendChild(make_bool("HTTPSampler.DO_MULTIPART_POST", False))
    el.appendChild(make_str("HTTPSampler.embedded_url_re", ""))
    el.appendChild(make_str("HTTPSampler.connect_timeout", "5000"))
    el.appendChild(make_str("HTTPSampler.response_timeout", "60000"))
    return el

def make_thread_group(name, num_threads, ramp_time, loops, enabled="true"):
    tg = root.createElement("ThreadGroup")
    tg.setAttribute("guiclass", "ThreadGroupGui")
    tg.setAttribute("testclass", "ThreadGroup")
    tg.setAttribute("testname", name)
    tg.setAttribute("enabled", enabled)
    tg.appendChild(make_str("ThreadGroup.on_sample_error", "continue"))
    ctrl = make_elem("ThreadGroup.main_controller", "LoopController",
                      "LoopControlPanel", "LoopController", "循环控制器")
    ctrl.appendChild(make_bool("LoopController.continue_forever", False))
    ctrl.appendChild(make_str("LoopController.loops", str(loops)))
    tg.appendChild(ctrl)
    tg.appendChild(make_str("ThreadGroup.num_threads", str(num_threads)))
    tg.appendChild(make_str("ThreadGroup.ramp_time", str(ramp_time)))
    tg.appendChild(make_bool("ThreadGroup.scheduler", False))
    tg.appendChild(make_str("ThreadGroup.duration", ""))
    tg.appendChild(make_str("ThreadGroup.delay", ""))
    tg.appendChild(make_bool("ThreadGroup.same_user_on_next_iteration", True))
    return tg

# ==================== 组装 ====================
top_hash = root.createElement("hashTree")
jmeterTestPlan.appendChild(top_hash)

test_plan = root.createElement("TestPlan")
test_plan.setAttribute("guiclass", "TestPlanGui")
test_plan.setAttribute("testclass", "TestPlan")
test_plan.setAttribute("testname", "角色扮演系统综合性能测试")
test_plan.setAttribute("enabled", "true")
test_plan.appendChild(make_str("TestPlan.comments",
    "角色扮演系统 JMeter 综合性能测试\n覆盖：健康检查、注册、登录、RAG对话、角色管理、管理后台"))
test_plan.appendChild(make_bool("TestPlan.functional_mode", False))
test_plan.appendChild(make_bool("TestPlan.tearDown_on_shutdown", True))
test_plan.appendChild(make_bool("TestPlan.serialize_threadgroups", True))  # 串行执行线程组

# User Defined Variables
udv = make_elem("TestPlan.user_defined_variables", "Arguments",
                 "ArgumentsPanel", "Arguments", "用户定义的变量")
coll = make_coll("Arguments.arguments")
for k, v in [
    ("TEST_PASSWORD", "JmeterTest123"),
    ("TEST_EMAIL", "jmeter_test@example.com"),
]:
    prop = root.createElement("elementProp")
    prop.setAttribute("name", k)
    prop.setAttribute("elementType", "Argument")
    for n, val in [("Argument.name", k), ("Argument.value", v), ("Argument.metadata", "=")]:
        prop.appendChild(make_str(n, val))
    prop.appendChild(make_bool("Argument.always_encode", False))
    prop.appendChild(make_bool("Argument.use_equals", True))
    coll.appendChild(prop)
udv.appendChild(coll)
test_plan.appendChild(udv)

top_hash.appendChild(test_plan)
plan_hash = root.createElement("hashTree")
top_hash.appendChild(plan_hash)

# ==================== HTTP默认值（全局，Test Plan 级别） ====================
defaults = root.createElement("ConfigTestElement")
defaults.setAttribute("guiclass", "HttpDefaultsGui")
defaults.setAttribute("testclass", "ConfigTestElement")
defaults.setAttribute("testname", "HTTP请求默认值")
defaults.setAttribute("enabled", "true")

a1 = make_elem("HTTPsampler.Arguments", "Arguments",
                "HTTPArgumentsPanel", "Arguments", "用户定义的变量")
a1.appendChild(make_coll("Arguments.arguments"))
defaults.appendChild(a1)
defaults.appendChild(make_str("HTTPSampler.domain", HOST))
defaults.appendChild(make_str("HTTPSampler.port", PORT))
defaults.appendChild(make_str("HTTPSampler.protocol", PROTO))
defaults.appendChild(make_str("HTTPSampler.contentEncoding", "UTF-8"))
defaults.appendChild(make_str("HTTPSampler.path", ""))
defaults.appendChild(make_str("HTTPSampler.concurrentPool", "6"))
defaults.appendChild(make_str("HTTPSampler.connect_timeout", "5000"))
defaults.appendChild(make_str("HTTPSampler.response_timeout", "60000"))

plan_hash.appendChild(defaults)
plan_hash.appendChild(root.createElement("hashTree"))

# ==================== 线程组1：健康检查（10线程，验证基础可用性） ====================
tg1 = make_thread_group("1-健康检查+基础接口", 10, 5, 1)
plan_hash.appendChild(tg1)
h1 = root.createElement("hashTree")
plan_hash.appendChild(h1)

# 使用 __threadNum 确保同线程统一用户名
UID = "__threadNum"

samplers = [
    ("Health", "GET", "/health", "", "true"),
    ("注册", "POST", "/api/register",
     '{"username":"jmeter_user_${'+UID+'}", "email":"user${'+UID+'}_${TEST_EMAIL}", "password":"${TEST_PASSWORD}"}', "true"),
    ("登录", "POST", "/api/login",
     '{"username":"jmeter_user_${'+UID+'}", "password":"${TEST_PASSWORD}"}', "true"),
    ("Bootstrap", "GET", "/api/bootstrap", "", "true"),
    ("更新角色偏好", "POST", "/api/preferences/active-role",
     '{"role_id":"friend_anime"}', "true"),
    ("清理角色历史", "DELETE", "/api/history/doctor_tcm", "", "true"),
    ("聊天(RAG)", "POST", "/api/chat",
     '{"user_id":"jmeter_user_${'+UID+'}", "char_id":"doctor_tcm", "query":"你好，请问你是健康顾问吗？"}', "true"),
]

for i, (label, method, path, body, enabled) in enumerate(samplers):
    sampler = make_http_sampler(f"{i+1}-{label}", path, method, body, enabled)
    h1.appendChild(sampler)
    sh = root.createElement("hashTree")
    h1.appendChild(sh)

    # 需要 Bearer token 的接口（非注册/登录/健康检查）
    needs_auth = "/api/login" not in path and "/api/register" not in path and path != "/health"
    needs_json = path != "/health"  # health check 不需要 Content-Type
    
    if needs_json:
        headers = [("Content-Type", "application/json")]
        if needs_auth:
            headers.append(("Authorization", "Bearer ${access_token}"))
        sh.appendChild(make_header_manager(headers))
        sh.appendChild(root.createElement("hashTree"))
    
    # 登录接口后提取 token
    if path == "/api/login":
        sh.appendChild(make_json_extractor(
            "access_token", "$.access_token", "1", "TOKEN_NOT_FOUND"))
        sh.appendChild(root.createElement("hashTree"))

# ==================== 线程组2：压力测试（健康检查500次） ====================
tg2 = make_thread_group("2-压力测试-健康检查(50线程×10循环=500请求)", 50, 10, 10)
plan_hash.appendChild(tg2)
h2 = root.createElement("hashTree")
plan_hash.appendChild(h2)

h2.appendChild(make_http_sampler("健康检查(压力)", "/health", "GET"))
h2.appendChild(root.createElement("hashTree"))

# ==================== 写入文件 ====================
xml_str = root.toprettyxml(indent="  ", encoding="UTF-8")
output_path = "/Users/suwente/Desktop/角色扮演系统-/jmeter-tests/roleplay-test-plan.jmx"
with open(output_path, "wb") as f:
    f.write(xml_str)
print(f"已生成: {output_path}  ({len(xml_str)} 字节)")

# Verify
tree = ET.parse(output_path)
root2 = tree.getroot()
tgs = root2.findall('.//ThreadGroup')
sams = root2.findall('.//HTTPSamplerProxy')
print(f"验证: {len(tgs)} 个线程组, {len(sams)} 个HTTP采样器, 平台={PROTO}://{HOST}:{PORT}")
for tg in tgs:
    name = tg.get('testname')
    threads = tg.find('.//stringProp[@name="ThreadGroup.num_threads"]')
    en = tg.get('enabled', 'true')
    print(f"  [{en}] {name} - {threads.text if threads is not None else '?'} 线程")
for s in sams:
    dom = s.find('.//stringProp[@name="HTTPSampler.domain"]')
    pt = s.find('.//stringProp[@name="HTTPSampler.port"]')
    path = s.find('.//stringProp[@name="HTTPSampler.path"]')
    print(f"    {s.get('testname')}: {dom.text if dom is not None else '?'}:{pt.text if pt is not None else '?'}{path.text if path is not None else '?'}")
