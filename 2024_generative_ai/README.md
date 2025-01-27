# Introduction to Generative AI 2024 Spring

## 1、2/23 課程內容說明

### 第 0 講: 課程說明

预习:
1. [80 分钟快速了解大语言模型](https://youtu.be/wG8-IUtqu-s?si=zW_iZQi4ctTLT_6A)

2. [NVIDIA 深度学习培训中心(DLI)](https://www.nvidia.cn/training/)

3. [能够使用工具的 AI](https://youtu.be/ZID220t_MpI?feature=shared)


### 第 1 講: 生成式 AI 是什麼?

---

## 2、3/1 提示工程 & AI 代理人

### 第 2 講: 今日的生成式人工智慧厲害在哪裡? 從「工具」變為「工具人」

### 第 3 講: 訓練不了人工智慧嗎? 你可以訓練你自己

#### Prompt Engineering

##### 1. 神奇咒語!

1. 叫模型思考 Chain of Thought (CoT)
   在 Prompt 中写出思考或者操作的步骤, 让 ChatGPT 参考该步骤

   InstructGPT (text-davinci-002)

2. 請模型解釋一下自己的答案

3. 對模型情緒勒索

4. No need to be polite with LLM so there is no need to add phrases like "please", "if you don't mind", "thank you", "I would like to", etc., 对 LLM 不需要礼貌, 所以不需要添加"请"、"如果你不介意"、"谢谢"、"我想"等短语.

5. Employ affirmative directives such as 'do', while steering clear of negative language like 'don't'. 使用肯定的指示, 如"做", 同时避开消极的语言, 如"不".

6. Add "I'm going to tip $xxx for a better solution!" 加上"如果有更好的解决方案, 我要给你xxx美元小费！"

7. Incorporate the following phrases: "You will be penalized" 加入以下短语: "你会受到惩罚的."

8. Add to your prompt the following phrase "Ensure that your answer is unbiased and avoids relying on stereotypes." 在你的提示中加上下面这句话: "确保你的答案是公正的, 避免依赖于刻板印象."

9. Prompt 中的 **画** 字可能导致 ChatGPT 生成图片, 如果不是要求 ChatGPT 生成图片, 需要慎用.

10. 用 AI 來找神奇咒語
    用增強式學習 (Reinforcement Learning, RL)

##### 2. 提供額外資訊

1. 把前提講清楚

2. 提供生成式 AI 不清楚的資訊

3. 提供範例
   In-context learning

##### 3. 把任務分多步驟來解

1. 拆解任務

2. 語言模型檢查自己的錯誤

3. 為什麼同一個問題每次答案都不同?
   Self-Consistency 自洽性

4. 打一套組合拳 Tree of Thoughts (ToT)

5. Algorithm of Thoughts

6. Graph of Thoughts

> note:
> Constitutional AI
>

##### 4. 使用工具

1. 使用工具 -- 搜尋引擎
   Retrieval Augmented Generation (RAG) 检索增强生成（RAG）

2. 使用工具 -- 寫程式

3. 使用工具 -- 文字生圖 AI (DALL-E)

4. GPT Plug-in

> 語言模型是怎麼使用工具的呢?
>

##### 5. 語言模型彼此合作

1. 讓合適的模型做合適的事情
   FrugalGPT

2. 讓模型彼此討論
   Multi-agent Debate 多代理的辩论

3. 引入不同的角色
   MetaGPT、ChatDev

---

## 3、3/8 生成策略 & 從專才到通才

### 第 4 講: 訓練不了人工智慧? 你可以訓練你自己(中)

---

## 4、3/22 深度學習 & Transformer

### 第 5 講: 訓練不了人工智慧? 你可以訓練你自己 (下) -- 讓語言彼此合作, 把一個人活成一個團隊

### 第 6 講: 大型語言模型修練史 -- 第一階段: 自我學習, 累積實力

1. 第一階段: 自我學習, 累積實力

自督導式學習 (Self-supervised Learning)
預訓練 (Pre-train)

2. 第二階段: 名師指點, 發揮潛力

Instruction Fine-tuning 指令微调
督導式學習 (Supervised Learning)

Adapter e.g. LoRA 适配器，例如 LoRA

3. 第三階段: 參與實戰, 打磨技巧

Reinforcement Learning (RL) 增強式學習

Reinforcement Learning from Human Feedback (RLHF) 通过人类反馈进行强化学习（RLHF）

RLHF -> RLAIF

Foundation Model 基础模型

Alignment 对齐

---

## 5、3/29深度學習 & Transformer

### 第 7 講: 大型語言模型修練史 -- 第二階段: 名師指點, 發揮潛力 (兼談對 ChatGPT 做逆向工程與 LLaMA 時代的開始)

---

## 6、4/12 評估生成式AI & 道德議題

### 第 8 講: 大型語言模型修練史 -- 第三階段: 參與實戰, 打磨技巧 (Reinforcement Learning from Human Feedback, RLHF)

### 第 9 講: 以大型語言模型打造的 AI Agent

Typically in AI, an agent refers to an artificial entity capable of perceiving its surroundings using sensors, making decisions, and then taking actions in response using actuators 通常在 AI 中, agent 是指能够使用传感器感知周围环境, 做出决策, 然后使用执行器采取响应行动的人工实体

The concept of agents originated in Philosophy, with roots tracing back to thinkers like Aristotle and Hume. It describes entities possessing desires, beliefs, intentions, and the ability to take actions. 代理人的概念起源于哲学, 其根源可以追溯到亚里士多德和休谟等思想家. 它描述了拥有欲望、信仰、意图和采取行动能力的实体.

This idea transitioned into computer science, intending to enable computers to understand users’ interests and autonomously perform actions on their behalf 这一想法转变为计算机科学, 旨在使计算机能够理解用户的兴趣, 并自主地代表他们执行操作

"agent" is an entity with the capacity to act. "代理人"是指有行为能力的实体.

甚麼是 agent
規劃 (修改規劃)
記憶 (詮釋記憶)

---

## 7、5/03 Transformer & 生成式 AI 的可解釋性

### 第 10 講: 今日的語言模型是如何做文字接龍的 -- 淺談 Transformer

模型演進:
N-gram -> Feed-forward Network -> Recurrent Neural Network (RNN) -> Transformer

### 第 11 講: 大型語言模型在「想」什麼呢?  -- 淺談大型語言模型的可解釋性

1. 找出影響輸出的關鍵輸入
2. 找出影響輸出的關鍵訓練資料
3. 分析 Embedding 中存有甚麼樣的資訊

---

## 8、5/10 評估生成式AI & 道德議題

### 第 12 講: 淺談檢定大型語言模型能力的各種方式

1. Massive Multitask Language Understanding (MMLU)
2. Chatbot Arena https://chat.lmsys.org/
3. 也許可以用強大的語言模型來評估? MT-Bench
4. glue
5. super glue
6. flan
7. crossfit
8. big-bench
9. natural instructions
10. 閱讀長文的能力 大海撈針 (Needle in a Haystack)
11. 語言模型會不會為達目的不擇手段? MACHIAVELLI Benchmark
12. 機器有沒有心智理論 (Theory of Mind)

### 第 13 講: 淺談大型語言模型相關的安全性議題 (上)

1. 大型語言模型還是會講錯話怎麼辦?

   事實查核 Factscore、FacTool

   有害詞彙檢測

2. 大型語言模型會不會自帶偏見?

3. 這句話是不是大型語言模型講的?

4. 大型語言模型也會被詐騙 Prompt Hacking

   Jailbreaking & Prompt Injection

---

## 9、5/17 生成策略

### 第 14 講: 淺談大型語言模型相關的安全性議題 (下) -- 欺騙大型語言模型

### 第 15 講: 為什麼語言模型用文字接龍, 圖片生成不用像素接龍呢? -- 淺談生成式人工智慧的生成策略

### 第 16 講: 可以加速所有語言模型生成速度的神奇外掛 -- Speculative Decoding

---

## 10、5/31 影像的生成式AI

### 第 17 講: 有關影像的生成式AI (上) -- AI 如何產生圖片和影片 (Sora 背後可能用的原理)

### 第 18 講: 有關影像的生成式AI (下) -- 快速導讀經典影像生成方法 (VAE, Flow, Diffusion, GAN) 以及與生成的影片互動

---

## 11、Extra lesson

### GPT-4o 背後可能的語音技術猜測

---
