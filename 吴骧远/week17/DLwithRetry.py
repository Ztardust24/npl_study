import json
import os.path
import re
import pandas as pd

class DialogSystem:
    def __init__(self):
        self.load()

    def load(self):
        self.all_node_info = {}
        self.load_scenario("scenario-买衣服.json")
        self.load_slot_template("slot_fitting_templet.xlsx")

    def load_scenario(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            scenario = json.load(f)
        scenario_name = os.path.basename(filename).split(".")[0]
        for node in scenario:
            self.all_node_info[scenario_name + ":" + node["id"]] = node
            if "childnode" in node:
                self.all_node_info[scenario_name + ":" + node["id"]]["childnode"] = [scenario_name + ":" + child for child in node["childnode"]]

    def load_slot_template(self, filename):
        self.slot_template = pd.read_excel(filename)
        # 分三列 ,逐行读取 slot作为key，然后query,values作为 value
        self.slot_info = {}
        for i in range(len(self.slot_template)):
            slot = self.slot_template.iloc[i, 0]
            query = self.slot_template.iloc[i, 1]
            values = self.slot_template.iloc[i, 2]
            self.slot_info[slot] = {"query": query, "values": values}
            print(slot, query, values)
    def nlu(self, memory):
        memory = self.intent_judge(memory)
        memory = self.slot_filling(memory)
        return memory

    def intent_judge(self, memory):
        query = memory["query"]
        max_score = -1
        hit_node = None
        for node in memory["available_nodes"]:
            score = self.calucate_node_score(query, node)
            if score > max_score:
                max_score = score
                hit_node = node
        memory["hit_node"] = hit_node
        memory["intent_score"] = max_score
        return memory

    def calucate_node_score(self, query, node):
        # 节点意图打分，算和intent相似度
        node_info = self.all_node_info[node]
        intent = node_info['intent']
        max_score = -1
        for sentence in intent:
            score = self.calculate_sentence_score(query, sentence)
            if score > max_score:
                max_score = score
        return max_score

    def calculate_sentence_score(self, query, sentence):
        #使用jaccard相似度
        query_set = set(query.split(" "))
        sentence_set = set(sentence.split(" "))
        return len(query_set & sentence_set) / len(query_set | sentence_set)

    def slot_filling(self, memory):
        hit_node = memory["hit_node"]
        node_info = self.all_node_info[hit_node]
        query = memory["query"]

        for slot in node_info.get('slot', []):
            if slot not in memory:
                slot_values = self.slot_info[slot]["values"]
                if re.search(slot_values, query):
                    memory[slot] = re.search(slot_values, query).group()
        return memory
    def dst(self, memory):
        hit_node = memory["hit_node"]
        node_info = self.all_node_info[hit_node]
        slot = node_info.get('slot', [])

        current_asking_slot = memory.get("require_slot")
        if current_asking_slot and current_asking_slot not in memory:
            query = memory["query"]
            slot_values = self.slot_info[current_asking_slot]["values"]
            if not re.search(slot_values, query):
                # 用户输入不匹配，标记需要重听
                memory["need_retry"] = True
                return memory

        # 清除重听标记
        memory["need_retry"] = False

        for s in slot:
            if s not in memory:
                memory["require_slot"] = s
                return memory
        memory["require_slot"] = None
        return memory

    def dpo(self, memory):
        # 检查是否需要重听
        if memory.get("need_retry", False):
            memory["policy"] = "retry"
            memory["available_nodes"] = [memory["hit_node"]]  # 停留在当前节点
            return memory

        if memory["require_slot"] is None:
            # 没有需要填充的槽位
            memory["policy"] = "reply"
            # self.take_action(memory)
            hit_node = memory["hit_node"]
            node_info = self.all_node_info[hit_node]
            memory["available_nodes"] = node_info.get("childnode", [])
        else:
            # 有欠缺的槽位
            memory["policy"] = "request"
            memory["available_nodes"] = [memory["hit_node"]]  # 停留在当前节点
        return memory

    def nlg(self, memory):
        # 根据policy执行反问或回答
        if memory["policy"] == "reply":
            hit_node = memory["hit_node"]
            node_info = self.all_node_info[hit_node]
            memory["response"] = self.fill_in_slot(node_info["response"], memory)

        elif memory["policy"] == "retry":
            slot = memory["require_slot"]
            slot_values = self.slot_info[slot]["values"]
            memory["response"] = f"{self.slot_info[slot]['query']}。请从以下选项中做出选择：{slot_values}。"

        else:
            # policy == "request"
            slot = memory["require_slot"]
            memory["response"] = self.slot_info[slot]["query"]
        return memory

    def fill_in_slot(self, sentence, memory):
        node = memory["hit_node"]
        node_info = self.all_node_info[node]
        for slot in node_info.get("slot", []):
            sentence = sentence.replace(slot, memory[slot])
        return sentence
    def run(self, query, memory):

        memory["query"] = query
        memory = self.nlu(memory)
        memory = self.dst(memory)
        memory = self.dpo(memory)
        memory = self.nlg(memory)

        return memory


if __name__ == "__main__":
    ds = DialogSystem()
    print(ds.all_node_info)
    print(ds.slot_info)
    memory = {"available_nodes": ["scenario-买衣服:node1"]}  # 记录用户状态
    while True:
        query = input("用户：")
        memory = ds.run(query, memory)
        print(memory)
        response = memory["response"]
        print("机器人:", response)
