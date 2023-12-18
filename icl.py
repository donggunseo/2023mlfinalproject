from torch.utils.data import Dataset, DataLoader
import random
from datasets import load_dataset
from tqdm import tqdm
import pickle
from transformers import AutoModelForCausalLM, LlamaTokenizer, AutoConfig, AutoTokenizer
import argparse
from utils import set_seed
import torch
import os
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score

class ICL_dataset(Dataset):
    def __init__(self, args, mode = 'raw'):
        self.args = args
        if args.data=="CR": ## train : 3.39k / test : 376
            data = load_dataset("SetFit/CR")
            self.train_data = data['train']
            self.dev_data = data['test']
            self.train_data = {'sentence' : self.train_data['text'], 'label' : self.train_data['label']}
            self.dev_data = {'sentence' : self.dev_data['text'], 'label' : self.dev_data['label']}
            self.id2label = {0: 'negative', 1: 'positive'}
            self.label2id = {'negative': 0, 'positive': 1}
            self.id2verb = ['negative', 'positive']
        elif args.data=="sst2": ## train : 6.92k / validation : 872 / test : 1.82k
            data = load_dataset("SetFit/sst2")
            self.train_data = data['train']
            self.dev_data = data['validation']
            self.train_data = {'sentence' : self.train_data['text'], 'label' : self.train_data['label']}
            self.dev_data = {'sentence' : self.dev_data['text'], 'label' : self.dev_data['label']}
            self.id2label = {0: 'negative', 1: 'positive'}
            self.label2id = {'negative': 0, 'positive': 1}
            self.id2verb = ['negative', 'positive']
        elif args.data=="trec": ## train : 5.45k / test : 500
            data = load_dataset("trec")
            self.train_data = data['train']
            self.dev_data = data['test']
            self.train_data = {'sentence' : self.train_data['text'], 'label' : self.train_data['coarse_label']}
            self.dev_data = {'sentence' : self.dev_data['text'], 'label' : self.dev_data['coarse_label']}
            self.id2label = {0:'expression', 1:'entity', 2:'description', 3:'human', 4:'location', 5:'number'}
            self.label2id = {v:k for k,v in self.id2label.items()}
            self.id2verb = list(self.label2id.keys())
        elif args.data=='subj': ## train : 8k / test : 2k
            data = load_dataset("SetFit/subj")
            self.train_data = data['train']
            self.dev_data = data['test']
            self.train_data = {'sentence' : self.train_data['text'], 'label' : self.train_data['label']}
            self.dev_data = {'sentence' : self.dev_data['text'], 'label' : self.dev_data['label']}
            self.id2label = {0:'objective', 1:'subjective'}
            self.label2id = {v:k for k,v in self.id2label.items()}
            self.id2verb = list(self.label2id.keys())
        elif args.data=='sst5': ## train : 8.54k / validation : 1.1k / test : 2.21k
            data = load_dataset("SetFit/sst5")
            self.train_data = data['train']
            self.dev_data = data['test']
            self.train_data = {'sentence' : self.train_data['text'], 'label' : self.train_data['label']}
            self.dev_data = {'sentence' : self.dev_data['text'], 'label' : self.dev_data['label']}
            self.id2label = {0: 'terrible', 1: 'bad', 2:'okay', 3:'good', 4:'great'}
            self.label2id = {v:k for k,v in self.id2label.items()}
            self.id2verb = list(self.label2id.keys())
        elif args.data=="agnews": ## train : 120k / test 7.6k -> 760
            data = load_dataset("ag_news")
            self.train_data = load_dataset("ag_news", split='train')
            self.dev_data = load_dataset("ag_news", split='test')
            self.dev_data = self.dev_data.train_test_split(test_size=0.1, stratify_by_column='label', seed = self.args.seed )['test']
            self.train_data = {'sentence' : self.train_data['text'], 'label' : self.train_data['label']}
            self.dev_data = {'sentence' : self.dev_data['text'], 'label' : self.dev_data['label']}
            self.id2label = {0: 'world', 1: 'sports', 2:'business', 3:'technology'}
            self.label2id = {v:k for k,v in self.id2label.items()}
            self.id2verb = list(self.label2id.keys())
        
        if self.args.data == "sst2":
            self.template=self.template_sst2
        elif self.args.data == "CR":
            self.template=self.template_CR
        elif self.args.data == "trec":
            self.template=self.template_trec
        elif self.args.data == "subj":
            self.template = self.template_subj
        elif self.args.data =="sst5":
            self.template = self.template_sst5
        elif self.args.data=="agnews":
            self.template = self.template_agnews
    
        self.demonstration_data = []
        if mode !='raw':
            self.random_sample(self.args.n_shot, self.args.n_org)
            self.dev_data = {'sentence' : ['', '[MASK]', 'N/A'], 'label' : [-1,-1,-1]}
        
        

    
    def random_sample(self, n_shot, n_org):
        idx = [i for i in range(len(self.train_data['label']))]
        for _ in range(n_org):
            data_subsample = random.sample(idx, n_shot)
            random.shuffle(data_subsample)
            self.demonstration_data.append({"sentence" : [self.train_data['sentence'][i] for i in data_subsample], 
                                            "label" : [self.train_data['label'][i] for i in data_subsample]})
    def template_sst2(self, sentence, label=None, mode='train'):
        if mode == 'train':
            return f"Review: {sentence}\nSentiment: {self.id2label[label]}\n"
        else:
            return f"Review: {sentence}\nSentiment: "
    def template_CR(self, sentence, label=None, mode='train'):
        if mode == 'train':
            return f"Review: {sentence}\nSentiment: {self.id2label[label]}\n"
        else:
            return f"Review: {sentence}\nSentiment: "
    def template_trec(self, sentence, label=None, mode='train'):
        if mode == 'train':
            return f"Question: {sentence}\nType: {self.id2label[label]}\n"
        else:
            return f"Question: {sentence}\nType: "
    def template_subj(self, sentence, label=None, mode='train'):
        if mode == 'train':
            return f"Input: {sentence}\nType: {self.id2label[label]}\n"
        else:
            return f"Input: {sentence}\nType: "
    def template_sst5(self, sentence, label=None, mode='train'):
        if mode == 'train':
            return f"Review: {sentence}\nSentiment: {self.id2label[label]}\n"
        else:
            return f"Review: {sentence}\nSentiment: "
    def template_agnews(self, sentence, label=None, mode='train'):
        if mode == 'train':
            return f"Input: {sentence}\nTopic: {self.id2label[label]}\n"
        else:
            return f"Input: {sentence}\nTopic: "

    def __len__(self):
        return len(self.dev_data['sentence']) 
    
    def __getitem__(self, idx):
        prompts = []
        for i in range(self.args.n_org):
            prompt = ''
            for s, l in zip(self.demonstration_data[i]['sentence'], self.demonstration_data[i]['label']):
                prompt+=self.template(s,l, mode='train')
                prompt += '\n'
            prompt+=self.template(self.dev_data['sentence'][idx], mode = 'inference')
            prompts.append(prompt)
        label = self.dev_data['label'][idx]
        return [prompts, label]
    

def parse_args():
    parser = argparse.ArgumentParser(description="In-Context Learning baseline.")
    parser.add_argument("--llm", type=str, default='llama2_7b')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", type=str, default="sst2")
    parser.add_argument("--n_shot", type=int, default=8)
    parser.add_argument("--n_org", type=int, default=10)
    args = parser.parse_args()

    return args

model_dict ={
    "llama2_7b" : "meta-llama/Llama-2-7b-hf",
    "falcon_7b" : "tiiuae/falcon-7b"
}
def cal_sensitivity(pred): ## pred : (num_samples, num_org) 
    robust_res = []
    for p in pred:
        counter = Counter(p)
        freq = counter.most_common(1)[0][1]
        robustness = freq/len(p)
        robust_res.append(robustness)
    return sum(robust_res)/len(robust_res)

def cal_accuracy(pred, label): ## pred : (num_samples, num_org) / label : (num_samples)
    pred2 = pred.transpose().tolist()
    acc_all = []
    for o in pred2:
        acc = accuracy_score(label, o)
        acc_all.append(acc)
    return sum(acc_all)/len(acc_all)





def main():
    args = parse_args()
    set_seed(args)
    cc_dataset = ICL_dataset(args, mode = 'cc')
    raw_dataset = ICL_dataset(args, mode = 'raw')
    raw_dataset.demonstration_data = cc_dataset.demonstration_data
    n_org = cc_dataset.args.n_org
    id2verb = cc_dataset.id2verb
    cc_dataloader = DataLoader(cc_dataset, batch_size = 1, shuffle=False)
    raw_dataloader = DataLoader(raw_dataset, batch_size = 1, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = LlamaTokenizer.from_pretrained(model_dict[args.llm]) if "llama" in args.llm else AutoTokenizer.from_pretrained(model_dict[args.llm])
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model_config = AutoConfig.from_pretrained(model_dict[args.llm])
    model = AutoModelForCausalLM.from_pretrained(model_dict[args.llm], config = model_config)
    model.to(device)
    model.eval()
    cc_ensemble_prob = []
    for batch in tqdm(cc_dataloader):
        prompt = batch[0]
        prompt = [t[0] for t in prompt]
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device=model.device)
        with torch.no_grad():
            logits = model.forward(input_ids=inputs['input_ids'],
                                    attention_mask=inputs['attention_mask'],
                                    return_dict=True).logits.detach().cpu()
        gen_logits = logits[:, -1, :]
        prob_per_cls = []
        for label_verb in id2verb:
            if args.llm!="falcon_7b":
                label_verb_token_id = tokenizer.encode(label_verb)[1] 
            else:
                label_verb_token_id = tokenizer.encode(label_verb)[0]
            prob_per_cls.append(gen_logits[:, label_verb_token_id])
        prob_cc = torch.stack(prob_per_cls, dim=1)
        prob_cc = torch.softmax(prob_cc, dim=-1).numpy()
        cc_ensemble_prob.append(prob_cc)
    cc_ensemble_prob = np.mean(np.array(cc_ensemble_prob), axis=0)
    raw_predict = []
    all_label = []
    for batch in tqdm(raw_dataloader):
        prompt = batch[0]
        label = batch[1].tolist()
        prompt = [t[0] for t in prompt]
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device=model.device)
        with torch.no_grad():
            logits = model.forward(input_ids=inputs['input_ids'],
                                    attention_mask=inputs['attention_mask'],
                                    return_dict=True).logits.detach().cpu()
        gen_logits = logits[:, -1, :]
        prob_per_cls = []
        for label_verb in id2verb:
            if args.llm!="falcon_7b":
                label_verb_token_id = tokenizer.encode(label_verb)[1] 
            else:
                label_verb_token_id = tokenizer.encode(label_verb)[0]
            prob_per_cls.append(gen_logits[:, label_verb_token_id])
        prob_raw = torch.stack(prob_per_cls, dim=1)
        prob_raw = torch.softmax(prob_raw, dim=-1).numpy()
        raw_predict.append(prob_raw)
        all_label.extend(label)
    raw_predict = np.array(raw_predict)
    raw_prediction = np.argmax(raw_predict, axis=2)
    cc_pred_prob = raw_predict / cc_ensemble_prob
    cc_pred_prob = cc_pred_prob/np.sum(cc_pred_prob, axis = 2, keepdims=True)
    cc_prediction = np.argmax(cc_pred_prob, axis=2)
    rb_raw = cal_sensitivity(raw_prediction)
    rb_cc = cal_sensitivity(cc_prediction)
    acc_raw = cal_accuracy(raw_prediction, all_label)
    acc_cc = cal_accuracy(cc_prediction, all_label)
    result_dir = f"./result/{args.llm}"
    os.makedirs(result_dir, exist_ok=True)
    result_file_path = f"{result_dir}/{args.data}.txt"
    if not os.path.isfile(result_file_path):
        with open(result_file_path, 'w') as f:
            f.write(f"{args.seed} \n robustness_raw : {rb_raw} \n robustness_cc : {rb_cc} \n accuracy_raw : {acc_raw} \n accuracy_cc : {acc_cc} \n")
    else:
        with open(result_file_path, 'a') as f:
            f.write(f"{args.seed} \n robustness_raw : {rb_raw} \n robustness_cc : {rb_cc} \n accuracy_raw : {acc_raw} \n accuracy_cc : {acc_cc} \n")


if __name__ == "__main__":
    main()