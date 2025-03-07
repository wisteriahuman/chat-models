import json
import random


input_templates = [
    "こんにちは",
    "おはようございます",
    "こんばんは",
    "元気ですか？",
    "今日の天気は？",
    "あなたの名前は？",
    "調子はどう？",
    "何をしていますか？",
    "今どこにいますか？",
    "最近どうですか？"
]

output_templates = [
    "こんにちは！何かお手伝いできることはありますか？",
    "おはようございます！良い一日をお過ごしください。",
    "こんばんは！どんなご用件でしょうか？",
    "私は元気です。あなたはいかがですか？",
    "申し訳ありませんが、リアルタイムの情報にはアクセスできません。ご了承ください。",
    "私はAIアシスタントです。お気軽にご質問ください。",
    "順調です。何かご質問はありますか？",
    "今はあなたのお話を聞いています。",
    "どこにいるかは秘密ですが、いつもオンラインです。",
    "最近は色々な質問にお答えしています。"
]

conversation_pairs = []
num_pairs = 100

for _ in range(num_pairs):
    inp = random.choice(input_templates)
    out = random.choice(output_templates)
    conversation_pairs.append({
        "input": inp,
        "output": out
    })


with open("data/conversations.json", "w", encoding="utf-8") as f:
    json.dump(conversation_pairs, f, ensure_ascii=False, indent=2)

print("会話データの生成が完了しました。")
