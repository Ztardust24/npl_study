import json
import pandas as pd
import re
import os

class DialogueSystem:
    def __init__(self):
        self.all_node_info = {}  # 存储所有场景节点信息
        self.slot_info = {}     # 存储槽位模板信息
        # 重听功能的触发关键词
        self.repeat_keywords = {"重听", "再说一遍", "重复", "再说一次"}
        self.load()
    
    def load(self):
        """加载所有场景和槽位模板"""
        self.load_scenario("scenario-买衣服.json")
        self.load_scenario("scenario-看电影.json")
        self.load_slot_templet("slot_fitting_templet.xlsx")

    def load_scenario(self, file):
        """加载单个场景脚本"""
        try:
            with open(file, 'r', encoding='utf-8') as f:
                scenario = json.load(f)
            
            scenario_name = os.path.basename(file).split('.')[0]
            for node in scenario:
                node_key = f"{scenario_name}_{node['id']}"
                self.all_node_info[node_key] = node
                # 处理子节点，添加场景前缀
                if "childnode" in node:
                    self.all_node_info[node_key]['childnode'] = [
                        f"{scenario_name}_{x}" for x in node['childnode']
                    ]
        except Exception as e:
            print(f"加载场景文件 {file} 出错: {e}")

    def load_slot_templet(self, file):
        """加载槽位模板Excel文件"""
        try:
            self.slot_templet = pd.read_excel(file)
            # 逐行读取，构建槽位信息字典
            for i in range(len(self.slot_templet)):
                slot = self.slot_templet.iloc[i]['slot']
                query = self.slot_templet.iloc[i]['query']
                values = self.slot_templet.iloc[i]['values']
                
                if pd.notna(slot) and slot not in self.slot_info:
                    self.slot_info[slot] = {
                        'query': query if pd.notna(query) else "",
                        'values': values if pd.notna(values) else ""
                    }
        except Exception as e:
            print(f"加载槽位模板 {file} 出错: {e}")

    def nlu(self, memory):
        """自然语言理解：意图识别和槽位填充"""
        memory = self.intent_judge(memory)
        memory = self.slot_filling(memory)
        return memory

    def intent_judge(self, memory):
        """意图识别：匹配当前可访问的节点"""
        query = memory['query']
        max_score = -1
        hit_node = None
        
        # 处理空节点列表情况
        if not memory["available_nodes"]:
            memory["hit_node"] = None
            memory["intent_score"] = 0
            return memory
            
        for node in memory["available_nodes"]:
            if node not in self.all_node_info:
                continue  # 跳过不存在的节点
            score = self.calculate_node_score(query, node)
            if score > max_score:
                max_score = score
                hit_node = node
        
        memory["hit_node"] = hit_node
        memory["intent_score"] = max_score
        return memory

    def calculate_node_score(self, query, node):
        """计算节点意图匹配分数"""
        node_info = self.all_node_info.get(node, {})
        intent_list = node_info.get('intent', [])
        max_score = -1
        
        for sentence in intent_list:
            score = self.calculate_sentence_score(query, sentence)
            if score > max_score:
                max_score = score
        return max_score
    
    def calculate_sentence_score(self, query, sentence):
        """计算两个句子的相似度（Jaccard系数）"""
        if not query or not sentence:
            return 0  # 空字符串相似度为0
        
        query_words = set(query)
        sentence_words = set(sentence)
        intersection = query_words.intersection(sentence_words)
        union = query_words.union(sentence_words)
        
        # 避免除以零
        return len(intersection) / len(union) if union else 0

    def slot_filling(self, memory):
        """槽位填充：从用户输入中提取槽位信息"""
        hit_node = memory.get("hit_node")
        if not hit_node or hit_node not in self.all_node_info:
            return memory
            
        query_text = memory.get('query', '')
        node_info = self.all_node_info[hit_node]
        
        for slot in node_info.get('slot', []):
            if slot in memory or slot not in self.slot_info:
                continue  # 已填充或无此槽位模板，跳过
                
            slot_values = self.slot_info[slot]["values"]
            if slot_values and re.search(slot_values, query_text):
                memory[slot] = re.search(slot_values, query_text).group()
        
        return memory

    def dst(self, memory):
        """对话状态跟踪：检查槽位是否完整"""
        hit_node = memory.get("hit_node")
        if not hit_node or hit_node not in self.all_node_info:
            memory["require_slot"] = None
            return memory
            
        node_info = self.all_node_info[hit_node]
        required_slots = node_info.get('slot', [])
        
        for slot in required_slots:
            if slot not in memory:
                memory["require_slot"] = slot
                return memory
                
        memory["require_slot"] = None
        return memory

    def dpo(self, memory):
        """对话策略优化：决定下一步动作"""
        if memory.get("require_slot") is None:
            # 槽位完整，准备回复并更新可用节点
            memory["policy"] = "reply"
            hit_node = memory.get("hit_node")
            if hit_node and hit_node in self.all_node_info:
                memory["available_nodes"] = self.all_node_info[hit_node].get("childnode", [])
        else:
            # 需要更多槽位信息，停留在当前节点
            memory["policy"] = "request"
            current_node = memory.get("hit_node")
            memory["available_nodes"] = [current_node] if current_node else []
            
        return memory

    def nlg(self, memory):
        """自然语言生成：生成回复内容"""
        if memory.get("policy") == "reply":
            hit_node = memory.get("hit_node")
            if hit_node and hit_node in self.all_node_info:
                node_info = self.all_node_info[hit_node]
                memory["response"] = self.fill_in_slot(node_info.get("response", ""), memory)
            else:
                memory["response"] = "抱歉，我不太明白您的意思。"
        else:
            # 请求补充槽位信息
            slot = memory.get("require_slot")
            if slot and slot in self.slot_info:
                memory["response"] = self.slot_info[slot]["query"]
            else:
                memory["response"] = "抱歉，我需要更多信息才能继续。"
                
        return memory

    def fill_in_slot(self, template, memory):
        """将槽位值填充到回复模板中"""
        if not template:
            return ""
            
        filled_template = template
        node = memory.get("hit_node")
        if node and node in self.all_node_info:
            node_info = self.all_node_info[node]
            for slot in node_info.get("slot", []):
                if slot in memory:
                    filled_template = filled_template.replace(slot, str(memory[slot]))
        
        return filled_template

    def check_repeat_request(self, query):
        """检查用户是否请求重听上一条回复"""
        query_clean = query.strip()
        return query_clean in self.repeat_keywords

    def run(self, query, memory):
        """处理用户输入，返回对话状态和响应"""
        # 检查是否需要重听
        if self.check_repeat_request(query):
            # 如果有上一条回复，则返回它
            if "last_response" in memory:
                memory["response"] = memory["last_response"]
            else:
                memory["response"] = "之前没有对话内容可以重听哦~"
            return memory
        
        # 正常处理流程
        memory["query"] = query.strip()  # 去除首尾空格
        memory = self.nlu(memory)
        memory = self.dst(memory)
        memory = self.dpo(memory)
        memory = self.nlg(memory)
        
        # 保存当前回复作为下一次可能的重听内容
        memory["last_response"] = memory["response"]
        return memory


if __name__ == '__main__':
    # 测试用例
    ds = DialogueSystem()
    print("槽位信息:", ds.slot_info)
    
    # 初始状态：可用节点为两个场景的起始节点，包含重听功能所需的last_response字段
    memory = {
        "available_nodes": ["scenario-买衣服_node1", "scenario-看电影_node1"],
        "last_response": ""
    }
    
    print("\n开始对话（输入'退出'结束，输入'重听'可重复上一条回复）：")
    while True:
        query = input("用户: ")
        if query == "退出":
            break
        if not query:
            print("系统: 请输入内容")
            continue
            
        memory = ds.run(query, memory)
        print("系统:", memory['response'])
        print("-" * 50)
