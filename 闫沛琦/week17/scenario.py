
import re
import os
import json
import pandas


class DialougeSystem:

    def __init__(self):
        self.all_node_info = {}
        self.slot_info = {}
        self.load()
        return

    def load(self):
        # TODO: 加载数据
        self.load_scenario("scenario-买衣服.json")
        self.load_template("slot_fitting_templet.xlsx")
        return


    # 加载场景文件
    def load_scenario(self, file_name):
        scenario_name = file_name.replace(".json", "")
        # 打开文件，以只读模式打开，编码格式为utf-8
        with open(file_name, "r", encoding="utf-8") as f:
            # 将文件内容加载为json格式，并赋值给self.scenario
            for node_info in json.load(f):
                node_id = node_info["id"]
                node_id = scenario_name + "_" + node_id
                if "childnode" in node_info:
                    node_info["childnode"] = [scenario_name + "_" + child_node for child_node in node_info["childnode"]]
                self.all_node_info[node_id] = node_info
        return
    
    def load_template(self, file_name):
        # TODO: 加载模板文件
        self.slot_template = pandas.read_excel(file_name, sheet_name="Sheet1")
        for index, row in self.slot_template.iterrows():
            self.slot_info[row["slot"]] = [row["query"], row["values"]]
        return

    def run(self, user_query, memory):
        if len(memory) == 0:
            memory["available_nodes"] = ["scenario-买衣服_node1"]
        memory["sue_query"] = user_query

        memory = self.nlu(memory)
        if not memory["other_sentence"]:
            memory = self.dst(memory)
            memory = self.dpo(memory)
            memory = self.nlg(memory)
        else:
            memory["response"] = "抱歉，我不明白您的意思，请重新输入。"
        return memory
    
    def nlu(self, memory):
        # TODO: 实现自然语言理解
        memory = self.intent_recognition(memory)

        if not memory["other_sentence"]:
            memory = self.slot_filling(memory)
        return memory
    
    def intent_recognition(self, memory):
        # TODO: 实现意图识别
        hit_score = 0
        hit_node_id = None
        memory["other_sentence"] = False
        for node_id in memory["available_nodes"]:
            node_info = self.all_node_info[node_id]
            score = self.calc_intent_score(node_info, memory)
            if score >= hit_score:
                hit_score = score
                hit_node_id = node_id
        memory["hit_node_id"] = hit_node_id
        memory["intent_score"] = hit_score

        if hit_score < 0.5 and ("missing_slot" not in memory or memory["missing_slot"] is None):
            memory["other_sentence"] = True

        return memory
    
    def calc_intent_score(self, node_info, memory):
        # TODO: 计算意图得分
        user_query = memory["sue_query"]
        intent_list = node_info["intent"]
        all_scores = []
        for intent in intent_list:
            score = self.sentence_similarity(intent, user_query)
            all_scores.append(score)
        return max(all_scores)
    
    def sentence_similarity(self, sentence1, sentence2):
        # TODO: 计算句子相似度
        score = len(set(sentence1) & set(sentence2)) / len(set(sentence1) | set(sentence2))
        return score
    
    def slot_filling(self, memory):
        # TODO: 实现槽位填充
        # 获取用户查询
        user_query = memory["sue_query"]
        # 获取匹配到的节点ID
        hit_node_id = memory["hit_node_id"]
        # 获取匹配到的节点的槽位列表
        slot_list = self.all_node_info[hit_node_id].get("slot", [])
        # 遍历槽位列表
        for slot in slot_list:
            # 获取槽位的候选词
            _, candidates = self.slot_info[slot]
            # 在用户查询中搜索候选词
            search_result = re.search(candidates, user_query)
            # 如果搜索结果不为空
            if search_result:
                memory[slot] = search_result.group()
        return memory

    def dst(self, memory):
        # TODO: 实现对话状态跟踪
        for slot in self.all_node_info[memory["hit_node_id"]].get("slot", []):
            if slot not in memory or memory[slot] is None:
                memory["missing_slot"] = slot
                return memory
        memory["missing_slot"] = None
        return memory

    def dpo(self, memory):
        # TODO: 实现对话策略优化
        if memory["missing_slot"] is not None:
            memory["policy"] = "ask"
            memory["available_nodes"] = [memory["hit_node_id"]]
        else:
            memory["policy"] = "answer"
            memory["available_nodes"] = self.all_node_info[memory["hit_node_id"]].get("childnode", [])
        return memory

    def nlg(self, memory):
        # TODO: 实现自然语言生成
        if memory["policy"] == "ask":
            slot = memory["missing_slot"]
            ask_sentence, _ = self.slot_info[slot]
            memory["response"] = ask_sentence
        elif memory["policy"] == "answer":
            response = self.all_node_info[memory["hit_node_id"]].get("response", "")
            response = self.replace_slot(response, memory)
            memory["response"] = response
        return memory
    
    def replace_slot(self, response, memory):
        slots = self.all_node_info[memory["hit_node_id"]].get("slot", [])
        for slot in slots:
            response = re.sub(slot, memory[slot], response)
        return response
    


if __name__ == "__main__":
    dialouge_system = DialougeSystem()

    print(dialouge_system.all_node_info)
    print(dialouge_system.slot_info)

    # memory = dialouge_system.run("你好，我想买一件短袖衣服", {})
    # print(memory)

    memory = {}
    while True:
        user_query = input("输入：")
        memory = dialouge_system.run(user_query, memory)
        print(memory["response"])
        print(memory)