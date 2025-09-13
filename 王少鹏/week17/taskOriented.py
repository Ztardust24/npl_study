# -*- coding: utf-8 -*-
"""
Task-oriented Multi-turn Dialogue System
Scenario: Buying a Car
"""

import json
import pandas as pd
import re
import os


class DialogueSystem:
    def __init__(self, scenario_file="scenario-买车.json", slot_file="slot_fitting_templet.xlsx"):
        self.node_info = {}
        self.slot_templates = {}
        self.load_scenario(scenario_file)
        self.load_slots(slot_file)

    # ===================== Data Loading =====================
    def load_scenario(self, file_path):
        """Load scenario JSON into memory"""
        with open(file_path, "r", encoding="utf-8") as f:
            scenario = json.load(f)
        scenario_name = os.path.basename(file_path).split('.')[0]

        for node in scenario:
            node_id = f"{scenario_name}_{node['id']}"
            self.node_info[node_id] = node
            if "childnode" in node:
                self.node_info[node_id]["childnode"] = [
                    f"{scenario_name}_{cid}" for cid in node["childnode"]
                ]

    def load_slots(self, file_path):
        """Load slot templates from Excel"""
        slot_table = pd.read_excel(file_path)
        for _, row in slot_table.iterrows():
            slot = row["slot"]
            self.slot_templates[slot] = {
                "query": str(row["query"]),
                "values": str(row["values"])
            }

    # ===================== NLU =====================
    def nlu(self, state):
        state = self.match_intent(state)
        state = self.fill_slots(state)
        return state

    def match_intent(self, state):
        """Intent recognition: select best matching node"""
        query = state["query"]
        max_score, best_node = -1, None
        for node_id in state["available_nodes"]:
            score = self.calc_node_score(query, node_id)
            if score > max_score:
                max_score, best_node = score, node_id
        state["hit_node"] = best_node
        state["intent_score"] = max_score
        return state

    def calc_node_score(self, query, node_id):
        """Calculate similarity between query and node intents"""
        node = self.node_info[node_id]
        return max(self.calc_text_similarity(query, intent) for intent in node["intent"])

    def calc_text_similarity(self, text1, text2):
        """Simple similarity: Jaccard based on word tokens"""
        words1 = set(re.findall(r"\w+", text1))
        words2 = set(re.findall(r"\w+", text2))
        return len(words1 & words2) / len(words2) if words2 else 0

    def fill_slots(self, state):
        """Slot filling based on regex match"""
        query = state["query"]
        node = self.node_info[state["hit_node"]]
        for slot in node.get("slot", []):
            if slot not in state:
                values_pattern = self.slot_templates[slot]["values"]
                match = re.search(values_pattern, query)
                if match:
                    state[slot] = match.group()
        return state

    # ===================== DST =====================
    def track_state(self, state):
        """Dialogue state tracking: check required slots"""
        node = self.node_info[state["hit_node"]]
        for slot in node.get("slot", []):
            if slot not in state:
                state["require_slot"] = slot
                return state
        state["require_slot"] = None
        return state

    # ===================== Policy =====================
    def decide_policy(self, state):
        """Dialogue policy: decide reply or request slot"""
        if state["require_slot"] is None:
            state["policy"] = "reply"
            state["available_nodes"] = self.node_info[state["hit_node"]].get("childnode", [])
        else:
            state["policy"] = "request"
            state["available_nodes"] = [state["hit_node"]]
        return state

    # ===================== NLG =====================
    def generate_response(self, state):
        """Natural language generation"""
        if state["policy"] == "reply":
            node = self.node_info[state["hit_node"]]
            state["response"] = self.render_response(node["response"], state)
        else:
            if self.is_repeat_request(state["query"]):
                prev = state.get("response", "")
                state["response"] = "好的，现在重新说一遍：" + prev
            else:
                slot = state["require_slot"]
                state["response"] = self.slot_templates[slot]["query"]
        return state

    def render_response(self, template, state):
        """Replace {slot} placeholders with actual values"""
        for slot in re.findall(r"\{(.*?)\}", template):
            if slot in state:
                template = template.replace("{" + slot + "}", state[slot])
        return template

    def is_repeat_request(self, text):
        if not text:
            return False
        repeat_pattern = r"(没听清|听不清|再说(一遍|一次)?|重复|重听|重说|重讲|请再说|能再说一遍|repeat|你说啥|啥|什么|没听懂|听不懂|听不清楚)"
        return re.search(repeat_pattern, text, re.IGNORECASE) is not None

    # ===================== Run =====================
    def run(self, query, state):
        """Main pipeline"""
        state["query"] = query
        state = self.nlu(state)
        state = self.track_state(state)
        state = self.decide_policy(state)
        state = self.generate_response(state)
        return state


if __name__ == "__main__":
    ds = DialogueSystem()
    print("🚗 Dialogue System Ready (Buy Car Scenario)")
    state = {"available_nodes": ["scenario-买车_node1"]}

    while True:
        user_input = input("用户：")
        state = ds.run(user_input, state)
        print("系统：", state["response"])
        print("===")
