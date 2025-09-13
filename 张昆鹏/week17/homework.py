# -*- coding: utf-8 -*-
import json
import pandas as pd
import re
import os

class DialogueSystem:
    def __init__(self):
        self.load()
        # 重听（rehear）匹配正则，匹配用户表示没听清或要求重复的常见表达
        patterns = [
            r"我(没|不)听清",
            r"再说(一遍)?",
            r"重复",
            r"你能再说(一遍)?吗",
            r"听不清",
            r"能不能再说",
            r"说(一遍|一下)"
        ]
        self.rehear_regex = re.compile("|".join(patterns), re.I)

    def load(self):
        self.all_node_info = {}
        self.load_scenario(r"N:\八斗\八斗精品班\第十七周 对话系统\week17 对话系统\scenario\scenario-买衣服.json")
        # self.load_scenario("scenario-看电影.json")
        self.load_slot_templet(r"N:\八斗\八斗精品班\第十七周 对话系统\week17 对话系统\scenario\slot_fitting_templet.xlsx")

    def load_scenario(self, file):
        with open(file, 'r', encoding='utf-8') as f:
            scenario = json.load(f)
        scenario_name = os.path.basename(file).split('.')[0]
        for node in scenario:
            self.all_node_info[scenario_name + "_" + node['id']] = node
            if "childnode" in node:
                self.all_node_info[scenario_name + "_" + node['id']]['childnode'] = [scenario_name + "_" + x for x in node['childnode']]

    def load_slot_templet(self, file):
        self.slot_templet = pd.read_excel(file)
        # 三列：slot, query, values
        self.slot_info = {}
        for i in range(len(self.slot_templet)):
            slot = self.slot_templet.iloc[i]['slot']
            query = self.slot_templet.iloc[i]['query']
            values = self.slot_templet.iloc[i]['values']
            if slot not in self.slot_info:
                self.slot_info[slot] = {}
            self.slot_info[slot]['query'] = query
            self.slot_info[slot]['values'] = values

    def nlu(self, memory):
        memory = self.intent_judge(memory)
        memory = self.slot_filling(memory)
        return memory

    def intent_judge(self, memory):
        query = memory['query']
        max_score = -1
        hit_node = None
        for node in memory.get("available_nodes", []):
            score = self.calucate_node_score(query, node)
            if score > max_score:
                max_score = score
                hit_node = node
        memory["hit_node"] = hit_node
        memory["intent_score"] = max_score
        return memory

    def calucate_node_score(self, query, node):
        node_info = self.all_node_info[node]
        intent = node_info['intent']
        max_score = -1
        for sentence in intent:
            score = self.calucate_sentence_score(query, sentence)
            if score > max_score:
                max_score = score
        return max_score

    def calucate_sentence_score(self, query, sentence):
        # 简单的字符级 Jaccard 相似度
        query_words = set(query)
        sentence_words = set(sentence)
        intersection = query_words.intersection(sentence_words)
        union = query_words.union(sentence_words)
        if not union:
            return 0.0
        return len(intersection) / len(union)

    def slot_filling(self, memory):
        # 槽位填充（修复：使用 memory 中的 query）
        query = memory.get('query', '')
        hit_node = memory.get("hit_node")
        if not hit_node:
            return memory
        node_info = self.all_node_info[hit_node]
        for slot in node_info.get('slot', []):
            if slot not in memory:
                slot_values = str(self.slot_info.get(slot, {}).get("values", ""))
                if slot_values and re.search(slot_values, str(query)):
                    memory[slot] = re.search(slot_values, str(query)).group()
        return memory

    def dst(self, memory):
        hit_node = memory.get("hit_node")
        if not hit_node:
            memory["require_slot"] = None
            return memory
        node_info = self.all_node_info[hit_node]
        slot = node_info.get('slot', [])
        for s in slot:
            if s not in memory:
                memory["require_slot"] = s
                return memory
        memory["require_slot"] = None
        return memory

    def dpo(self, memory):
        if memory.get("require_slot") is None:
            memory["policy"] = "reply"
            hit_node = memory.get("hit_node")
            node_info = self.all_node_info.get(hit_node, {})
            memory["available_nodes"] = node_info.get("childnode", [])
        else:
            memory["policy"] = "request"
            memory["available_nodes"] = [memory.get("hit_node")]
        return memory

    def nlg(self, memory):
        if memory.get("policy") == "reply":
            hit_node = memory.get("hit_node")
            node_info = self.all_node_info.get(hit_node, {})
            memory["response"] = self.fill_in_slot(node_info.get("response", ""), memory)
        else:
            slot = memory.get("require_slot")
            memory["response"] = self.slot_info.get(slot, {}).get("query", "我没听懂，你能再说一遍吗？")
        # 保存最后一次回复，供重听使用
        memory["last_response"] = memory.get("response")
        return memory

    def fill_in_slot(self, template, memory):
        node = memory.get("hit_node")
        node_info = self.all_node_info.get(node, {})
        for slot in node_info.get("slot", []):
            if slot in memory:
                template = template.replace(slot, str(memory[slot]))
        return template

    def is_rehear(self, query):
        return bool(self.rehear_regex.search(str(query)))

    def run(self, query, memory):
        '''
        query: 用户输入
        memory: 用户状态
        '''
        memory["query"] = query
        # 若用户请求重听，则直接重复上一次的 response（不推进状态机）
        if self.is_rehear(query):
            prev = memory.get("last_response")
            if prev is not None:
                memory["response"] = prev
                return memory
            # 如果没有上一条回复，则继续正常流程

        memory = self.nlu(memory)
        memory = self.dst(memory)
        memory = self.dpo(memory)
        memory = self.nlg(memory)
        return memory


if __name__ == '__main__':
    ds = DialogueSystem()
    print("slot info keys:\n", list(ds.slot_info.keys()))

    # memory = {"available_nodes": ["scenario-买衣服_node1", "scenario-看电影_node1"]}
    memory = {"available_nodes": ["scenario-买衣服_node1"]}
    while True:
        query = input("请输入：")
        memory = ds.run(query, memory)
        print(memory)
        print()
        response = memory.get('response', '')
        print(response)
        print("===========")
