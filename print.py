from scipy.stats import pearsonr
dataset = ['agnews', 'CR', 'sst2', 'sst5', 'subj', 'trec']
models = ['llama2_7b', 'falcon_7b']
for model in models:
    print(f"{model} results")
    for ds in dataset:
        raw_rb = []
        raw_acc=[]
        cc_rb =[]
        cc_acc = []
        with open(f'./result/{model}/{ds}.txt', 'r') as f:
            lines = f.readlines()
        for i in range(len(lines)):
            if i%5==0:
                continue
            else:
                t = float(lines[i].strip().split(":")[1])
                t = round(t,3)
                if i%5==1:
                    raw_rb.append(t)
                elif i%5==2:
                    cc_rb.append(t)
                elif i%5==3:
                    raw_acc.append(t)
                elif i%5==4:
                    cc_acc.append(t)
        print(f'{ds} raw_rb : {sum(raw_rb)/len(raw_rb)} / cc_rb : {sum(cc_rb)/len(cc_rb)} / raw_acc : {sum(raw_acc)/len(raw_acc)} / cc_acc : {sum(cc_acc)/len(cc_acc)}')
        print(f'{ds} pearson correlation coefficient \nraw : {round(pearsonr(raw_rb, raw_acc)[0],3)} / cc : {round(pearsonr(cc_rb, cc_acc)[0],3)}')
        print("________________________")