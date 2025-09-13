from transformers import BertTokenizer, BertForQuestionAnswering, pipeline
import torch
import re
import random


class CompleteDialogueSystem:
    def __init__(self):
        # 初始化问答模型
        self.qa_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        self.qa_model = BertForQuestionAnswering.from_pretrained("bert-base-cased")

        # 初始化对话模型（用于闲聊）
        self.chatbot = pipeline(
            "text-generation",
            model="microsoft/DialoGPT-medium",
            tokenizer="microsoft/DialoGPT-medium"
        )

        self.conversation_history = []
        self.current_context = ""
        self.last_qa_answer = ""

    def set_context(self, context):
        """设置对话上下文"""
        self.current_context = context
        print(f"已设置上下文：{context[:100]}...")

    def extract_answer(self, question):
        """从上下文中提取答案"""
        try:
            inputs = self.qa_tokenizer(question, self.current_context, return_tensors="pt", truncation=True,
                                       max_length=512)

            with torch.no_grad():
                outputs = self.qa_model(**inputs)

            start_scores = outputs.start_logits
            end_scores = outputs.end_logits

            start_idx = torch.argmax(start_scores)
            end_idx = torch.argmax(end_scores)

            answer_tokens = inputs["input_ids"][0][start_idx:end_idx + 1]
            answer = self.qa_tokenizer.decode(answer_tokens, skip_special_tokens=True)

            return answer.strip()
        except:
            return None

    def chat_response(self, user_input):
        """生成闲聊回复"""
        # 构建对话历史
        chat_history = ""
        for msg in self.conversation_history[-4:]:  # 最近4条消息
            if msg['role'] == 'user':
                chat_history += f"用户：{msg['content']}\n"
            else:
                chat_history += f"助手：{msg['content']}\n"

        prompt = f"{chat_history}用户：{user_input}\n助手："

        try:
            response = self.chatbot(
                prompt,
                max_length=200,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )[0]['generated_text']

            # 提取助手的回复
            assistant_response = response.split("助手：")[-1].strip()
            return assistant_response
        except:
            # 如果模型失败，使用预设回复
            return self.get_fallback_response(user_input)

    def get_fallback_response(self, user_input):
        """预设回复"""
        greetings = ["你好！", "嗨！", "您好！", "很高兴和您聊天！"]
        questions = ["很有趣的问题！", "让我想想...", "这是个好问题！"]
        unknowns = ["我不太明白，能换个说法吗？", "抱歉，我不太确定您的意思", "能再说得详细些吗？"]

        user_input = user_input.lower()

        if any(word in user_input for word in ["你好", "嗨", "hello", "hi"]):
            return random.choice(greetings)
        elif any(word in user_input for word in ["吗？", "吗", "？", "?"]):
            return random.choice(questions)
        else:
            return random.choice(unknowns)

    def is_repeat_request(self, user_input):
        """检测重复请求"""
        repeat_patterns = [
            r'没听懂', r'再说一遍', r'重复', r'没听清',
            r'再说一次', r'什么', r'pardon', r'what',
            r'刚刚说什么', r'没听明白'
        ]

        user_input = user_input.lower()
        return any(re.search(pattern, user_input) for pattern in repeat_patterns)

    def is_qa_question(self, user_input):
        """检测是否是问答问题"""
        qa_keywords = ["谁", "什么", "哪里", "何时", "为什么", "怎么", "如何", "who", "what", "where", "when", "why",
                       "how"]
        user_input = user_input.lower()
        return any(keyword in user_input for keyword in qa_keywords) and "？" in user_input or "?" in user_input

    def process_message(self, user_input):
        """处理用户输入"""
        user_input = user_input.strip()

        # 保存用户消息
        self.conversation_history.append({"role": "user", "content": user_input})

        # 检查重复请求
        if self.is_repeat_request(user_input) and self.last_qa_answer:
            response = f"好的，我再说一遍：{self.last_qa_answer}"
            self.conversation_history.append({"role": "assistant", "content": response})
            return response

        # 检查是否是问答问题且有上下文
        if self.is_qa_question(user_input) and self.current_context:
            answer = self.extract_answer(user_input)
            if answer and len(answer) > 2:  # 确保答案有效
                self.last_qa_answer = answer
                response = f"根据上下文：{answer}"
            else:
                response = "抱歉，我在上下文中找不到这个问题的答案。"
        else:
            # 闲聊模式
            response = self.chat_response(user_input)

        # 保存助手回复
        self.conversation_history.append({"role": "assistant", "content": response})
        return response


# 增强版对话界面
def run_enhanced_chat():
    system = CompleteDialogueSystem()

    print("=" * 50)
    print("🤖 智能对话机器人已启动！")
    print("📝 功能：")
    print("  - 问答：基于上下文回答谁、什么、哪里等问题")
    print("  - 闲聊：普通对话交流")
    print("  - 重听：说'没听懂'可以重复上一个答案")
    print("  - 输入'设置上下文'来设置问答背景")
    print("  - 输入'退出'结束对话")
    print("=" * 50)

    # 初始上下文设置
    initial_context = """
    Jim Henson是一位美国木偶师、动画师、漫画家、演员、发明家和电影制片人，因创作《布偶秀》而享誉全球。
    他于1936年出生，1990年去世。亨森创造了著名的角色如青蛙克米特、猪小姐和大鸟。
    他的作品对儿童电视节目产生了深远影响，并获得了多个艾美奖。
    """
    system.set_context(initial_context)

    while True:
        try:
            user_input = input("\n👤 您：").strip()

            if user_input.lower() in ['退出', 'exit', 'quit', 'bye']:
                print("🤖 机器人：再见！很高兴和您聊天！")
                break

            elif user_input.lower() in ['设置上下文', 'set context']:
                new_context = input("请输入新的上下文：")
                system.set_context(new_context)
                print("🤖 机器人：上下文已更新！")
                continue

            elif not user_input:
                print("🤖 机器人：请说点什么吧~")
                continue

            # 处理用户输入
            response = system.process_message(user_input)
            print(f"🤖 机器人：{response}")

        except KeyboardInterrupt:
            print("\n🤖 机器人：感谢使用，再见！")
            break
        except Exception as e:
            print(f"🤖 机器人：出错了，请重新输入 ({str(e)})")


# 快速测试
def quick_test():
    system = CompleteDialogueSystem()

    # 设置测试上下文
    context = """
    清华大学是中国著名的综合性大学，位于北京市。成立于1911年，是中国最顶尖的高等学府之一。
    清华大学在工程、计算机科学、经济管理等领域享有盛誉。校园占地面积约400公顷，风景优美。
    """
    system.set_context(context)

    test_cases = [
        "你好！",
        "清华大学在哪里？",
        "没听懂，再说一遍",
        "清华大学成立于哪一年？",
        "今天天气真好",
        "清华大学以什么专业闻名？",
        "再见"
    ]

    for test in test_cases:
        print(f"\n👤 测试输入：{test}")
        response = system.process_message(test)
        print(f"🤖 回复：{response}")


if __name__ == "__main__":
    # 运行完整对话系统
    # run_enhanced_chat()

    # 或者运行快速测试
    quick_test()
