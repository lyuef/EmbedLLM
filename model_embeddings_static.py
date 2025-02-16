import torch
from transformers import AutoModel, AutoTokenizer

model_path = 'Alibaba-NLP/gte-large-en-v1.5'
# model_path = '/nas-alinlp/linzhang.zx/models/Alibaba-NLP/gte-large-en-v1_5'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
model.eval().cuda()

model_order_csv = """
0,0,Qwen__Qwen1.5-7B-Chat
1,1,ConvexAI__Luminex-34B-v0.1
2,2,lmsys__vicuna-13b-v1.5
3,3,deepseek-ai__deepseek-math-7b-instruct
4,4,TigerResearch__tigerbot-13b-base
5,5,ConvexAI__Luminex-34B-v0.2
6,6,berkeley-nest__Starling-LM-7B-alpha
7,7,EleutherAI__llemma_7b
8,8,CultriX__NeuralTrix-bf16
9,9,SciPhi__SciPhi-Mistral-7B-32k
10,10,TheBloke__tulu-30B-fp16
11,11,lmsys__vicuna-33b-v1.3
12,12,scb10x__typhoon-7b
13,13,mlabonne__AlphaMonarch-7B
14,14,mistralai__Mistral-7B-Instruct-v0.1
15,15,01-ai__Yi-34B-Chat
16,16,meta-llama__Llama-2-13b-chat-hf
17,17,eren23__ogno-monarch-jaskier-merge-7b-OH-PREF-DPO
18,18,ibivibiv__alpaca-dragon-72b-v1
19,19,golaxy__gowizardlm
20,20,codellama__CodeLlama-34b-Instruct-hf
21,21,OpenBuddy__openbuddy-codellama2-34b-v11.1-bf16
22,22,deepseek-ai__deepseek-coder-1.3b-base
23,23,Neko-Institute-of-Science__pygmalion-7b
24,24,cognitivecomputations__yayi2-30b-llama
25,25,meta-llama__LlamaGuard-7b
26,26,NousResearch__Nous-Hermes-13b
27,27,tiiuae__falcon-40b-instruct
28,28,meta-llama__Llama-2-7b-chat-hf
29,29,mosaicml__mpt-7b-chat
30,30,Qwen__Qwen1.5-32B-Chat
31,31,NousResearch__Nous-Hermes-2-Yi-34B
32,32,deepseek-ai__deepseek-coder-6.7b-instruct
33,33,google__gemma-7b-it
34,34,EleutherAI__llemma_34b
35,35,zhengr__MixTAO-7Bx2-MoE-v8.1
36,36,yam-peleg__Experiment26-7B
37,37,meta-llama__Meta-Llama-3-8B
38,38,mosaicml__mpt-30b-instruct
39,39,fblgit__UNA-SimpleSmaug-34b-v1beta
40,40,FelixChao__vicuna-7B-physics
41,41,TheBloke__koala-13B-HF
42,42,meta-llama__Meta-Llama-3-70B
43,43,Plaban81__Moe-4x7b-math-reason-code
44,44,meta-math__MetaMath-Mistral-7B
45,45,BioMistral__BioMistral-7B
46,46,FelixChao__Scorpio-7B
47,47,SciPhi__SciPhi-Self-RAG-Mistral-7B-32k
48,48,microsoft__phi-2
49,49,CausalLM__34b-beta
50,50,meta-llama__Meta-Llama-3-70B-Instruct
51,51,meta-math__MetaMath-Llemma-7B
52,52,lmsys__vicuna-7b-v1.5-16k
53,53,cloudyu__Mixtral_11Bx2_MoE_19B
54,54,Qwen__Qwen1.5-4B-Chat
55,55,FelixChao__vicuna-7B-chemical
56,56,HuggingFaceH4__zephyr-7b-beta
57,57,OpenAssistant__oasst-sft-4-pythia-12b-epoch-3.5
58,58,BioMistral__BioMistral-7B-DARE
59,59,Biomimicry-AI__ANIMA-Nectar-v2
60,60,microsoft__phi-1_5
61,61,meta-llama__Meta-Llama-Guard-2-8B
62,62,rishiraj__CatPPT-base
63,63,kyujinpy__Sakura-SOLRCA-Math-Instruct-DPO-v1
64,64,meta-llama__Meta-Llama-3-8B-Instruct
65,65,google__gemma-2b-it
66,66,upstage__SOLAR-10.7B-Instruct-v1.0
67,67,CorticalStack__pastiche-crown-clown-7b-dare-dpo
68,68,01-ai__Yi-6B
69,69,codefuse-ai__CodeFuse-DeepSeek-33B
70,70,abhishek__zephyr-beta-math
71,71,bardsai__jaskier-7b-dpo-v5.6
72,72,allenai__tulu-2-dpo-70b
73,73,Harshvir__Llama-2-7B-physics
74,74,lmsys__vicuna-13b-v1.5-16k
75,75,shleeeee__mistral-ko-tech-science-v1
76,76,JaeyeonKang__CCK_Asura_v1
77,77,codellama__CodeLlama-7b-hf
78,78,Nexusflow__Starling-LM-7B-beta
79,79,microsoft__Orca-2-13b
80,80,Neko-Institute-of-Science__metharme-7b
81,81,bigcode__octocoder
82,82,PharMolix__BioMedGPT-LM-7B
83,83,SUSTech__SUS-Chat-34B
84,84,kevin009__llamaRAGdrama
85,85,meta-llama__Llama-2-70b-chat-hf
86,86,TheBloke__CodeLlama-70B-Instruct-AWQ
87,87,openchat__openchat_3.5
88,88,dfurman__HermesBagel-34B-v0.1
89,89,project-baize__baize-v2-13b
90,90,augmxnt__shisa-base-7b-v1
91,91,lmsys__vicuna-7b-v1.5
92,92,Intel__neural-chat-7b-v3-3
93,93,AdaptLLM__medicine-LLM-13B
94,94,openchat__openchat-3.5-0106
95,95,deepseek-ai__deepseek-llm-67b-chat
96,96,FelixChao__llama2-13b-math1.2
97,97,MaziyarPanahi__WizardLM-Math-70B-v0.1
98,98,01-ai__Yi-6B-200K
99,99,WizardLM__WizardLM-70B-V1.0
100,100,bigscience__bloom-7b1
101,101,sail__Sailor-7B
102,102,codellama__CodeLlama-13b-Instruct-hf
103,103,Writer__palmyra-med-20b
104,104,Qwen__Qwen1.5-0.5B-Chat
105,105,databricks__dolly-v2-12b
106,106,nomic-ai__gpt4all-13b-snoozy
107,107,stabilityai__stablelm-tuned-alpha-7b
108,108,AdaptLLM__medicine-chat
109,109,AdaptLLM__medicine-LLM
110,110,EleutherAI__pythia-12b
111,111,Q-bert__Optimus-7B
"""

input_texts = [l.split(',')[-1] for l in model_order_csv.split('\n') if l]

# Tokenize the input texts
batch_dict = tokenizer(input_texts, max_length=8192, padding=True, truncation=True, return_tensors='pt')

with torch.inference_mode():
    outputs = model(**batch_dict.to('cuda'))
    embeddings = outputs.last_hidden_state[:, 0]
 
# (Optionally) normalize embeddings
embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
torch.save(embeddings, 'data/model_embeddings_static.pth')
print('saved')
