'''
COMMAND LINE EXAMPLE:
python3 cv_stratified.py ../N2C2 1,6 3,8,12
'''

################
### packages ###
################

# for reading files
import pandas as pd 
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from pathlib import Path
import sys
import re
import argparse
import textwrap
import pickle

# for cleaning discharge summaries
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import warnings
from sklearn.utils import shuffle

# for labeling functions
from snorkel.labeling import labeling_function
from snorkel.labeling.lf.nlp import nlp_labeling_function
from nltk.tokenize import RegexpTokenizer
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LFAnalysis

# for label models
from snorkel.labeling import LabelModel
from snorkel.labeling import MajorityLabelVoter
from sklearn import metrics

##############
### labels ###
##############
MATCHING =  1
NOT_MATCHING = 0
ABSTAIN = -1

##############
### models ###
##############
sys.path.insert(1, '../models')
from lf_models import lf_model_svm_ade_only_linear
from lf_models import lf_model_svm_ade_only_polynomial
from lf_models import lf_model_svm_ade_only_sigmoid
from lf_models import lf_model_svm_ade_only_rbf
from lf_models import lf_model_svm_ade_only_gridsearch
from lf_models import lf_model_svm_ade_drug_linear
from lf_models import lf_model_svm_ade_drug_polynomial
from lf_models import lf_model_svm_ade_drug_sigmoid
from lf_models import lf_model_svm_ade_drug_rbf
from lf_models import lf_model_svm_ade_drug_gridsearch
from lf_models import lf_model_svm_reason_drug_linear
from lf_models import lf_model_svm_reason_drug_polynomial
from lf_models import lf_model_svm_reason_drug_sigmoid
from lf_models import lf_model_svm_reason_drug_rbf
from lf_models import lf_model_svm_reason_drug_gridsearch
from lf_models import lf_model_multi_label_one_vs_rest
from lf_models import lf_model_multi_label_binary_relevance
from lf_models import lf_model_multi_label_classifier_chains
from lf_models import lf_model_multi_label_label_powerset
from lf_models import lf_model_multi_label_adapted_algorithm
from lf_models import lf_model_flair


#############
### rules ###
#############
sys.path.insert(1, '../rules')
from rule1 import lf_ade_drug_single
from rule2 import lf_ade_drug_pair
from rule3 import lf_ade_drug_pair_lem
from rule4 import lf_ade_drug_pair_lem_keyword_triggers
from rule5 import lf_sider2_triggers
from rule6 import lf_sider2_triggers_25words
from rule7 import lf_semmeddb_triggers
from rule8 import lf_keyword_triggers
from rule9 import lf_paper_triggers
from rule10 import lf_paper_triggers_200char
from rule11 import lf_paper_triggers_200char_negate
from rule12 import lf_paper_triggers_25words

#############################
### create folder for log ###
#############################

if Path("../outputs/log_cv.csv").exists() :
    log_df = pd.read_csv("../outputs/log_cv.csv")
else :
    log_df = pd.DataFrame({'run': [], 'models': [], 'rules': [], 'train_0': [], 'train_1': [], 'test_0': [], 'test_1': [],
    'lm_accuracy': [], 'lm_precision': [], 'lm_recall': [], 'lm_ROCAUC': [], 'lm_f1': [], 'lm_CM_TN': [], 'lm_CM_FP': [], 'lm_CM_FN': [], 'lm_CM_TP': [],
    'mlv_accuracy': [], 'mlv_precision': [], 'mlv_recall': [], 'mlv_ROCAUC': [], 'mlv_f1': [], 'mlv_CM_TN': [], 'mlv_CM_FP': [], 'mlv_CM_FN': [], 'mlv_CM_TP': []})

########################
### models and rules ###
########################

models_dict = {
    "1": lf_model_svm_ade_only_linear,
    "2": lf_model_svm_ade_only_polynomial,
    "3": lf_model_svm_ade_only_sigmoid,
    "4": lf_model_svm_ade_only_rbf,
    "5": lf_model_svm_ade_only_gridsearch,
    "6": lf_model_svm_ade_drug_linear,
    "7": lf_model_svm_ade_drug_polynomial,
    "8": lf_model_svm_ade_drug_sigmoid,
    "9": lf_model_svm_ade_drug_rbf,
    "10": lf_model_svm_ade_drug_gridsearch,
    "11": lf_model_svm_reason_drug_linear,
    "12": lf_model_svm_reason_drug_polynomial,
    "13": lf_model_svm_reason_drug_sigmoid,
    "14": lf_model_svm_reason_drug_rbf,
    "15": lf_model_svm_reason_drug_gridsearch,
    "16": lf_model_multi_label_one_vs_rest,
    "17": lf_model_multi_label_binary_relevance,
    "18": lf_model_multi_label_classifier_chains,
    "19": lf_model_multi_label_label_powerset,
    "20": lf_model_multi_label_adapted_algorithm,
    "21": lf_model_flair
}

models_dict_desc = {
    "0": "No Models Selected",
    "1": "ADE-Only Prediction using Linear SVM",
    "2": "ADE-Only Prediction using Polynomial SVM",
    "3": "ADE-Only Prediction using Sigmoid SVM",
    "4": "ADE-Only Prediction using RBF SVM",
    "5": "ADE-Only Prediction using GridSearch SVM",
    "6": "ADE-Drug Prediction using Linear SVM",
    "7": "ADE-Drug Prediction using Polynomial SVM",
    "8": "ADE-Drug Prediction using Sigmoid SVM",
    "9": "ADE-Drug Prediction using RBF SVM",
    "10": "ADE-Drug Prediction using GridSearch SVM",
    "11": "Reason-Drug Prediction using Linear SVM",
    "12": "Reason-Drug Prediction using Polynomial SVM",
    "13": "Reason-Drug Prediction using Sigmoid SVM",
    "14": "Reason-Drug Prediction using RBF SVM",
    "15": "Reason-Drug Prediction using GridSearch SVM",
    "16": "Multi-Label Prediction using One vs Rest",
    "17": "Multi-Label Prediction using Binary Relevance",
    "18": "Multi-Label Prediction using Classifier Chains",
    "19": "Multi-Label Prediction using Label Powerset",
    "20": "Multi-Label Prediction using Adapted Algorithm",
    "21": "Flair"
}

rules_dict = {
    "1": lf_ade_drug_single,
    "2": lf_ade_drug_pair,
    "3": lf_ade_drug_pair_lem,
    "4": lf_ade_drug_pair_lem_keyword_triggers,
    "5": lf_sider2_triggers, 
    "6": lf_sider2_triggers_25words,
    "7": lf_semmeddb_triggers,
    "8": lf_keyword_triggers,
    "9": lf_paper_triggers,
    "10": lf_paper_triggers_200char,
    "11": lf_paper_triggers_200char_negate,
    "12": lf_paper_triggers_25words
}

rules_dict_desc = {
    "0": "No Rules Selected",
    "1": "lf_ade_drug_single - any keywords in ade_drug_single found in discharge summary",
    "2": "lf_ade_drug_pair - any pair of keywords in ade_drug_pair found in discharge summary",
    "3": "lf_ade_drug_pair_lem - any pair of lemmatised keywords in ade_drug_pair found in discharge summary",
    "4": "lf_ade_drug_pair_lem_keyword_triggers - any pair of lemmatised keywords in ade_drug_pair and any trigger word in keyword_triggers found in discharge summary",
    "5": "lf_sider2_triggers - any pair of trigger words in sider2_triggers found in discharge summary",
    "6": "lf_sider2_triggers_25words - any pair of trigger words in sider2_triggers within 25 words of each other found in discharge summary",
    "7": "lf_semmeddb_triggers - any pair of trigger words in semmeddb_triggers found in discharge summary",
    "8": "lf_keyword_triggers - any trigger word in keyword_triggers found in discharge summary",
    "9": "lf_paper_triggers - any trigger word in paper_triggers found in discharge summary",
    "10": "lf_paper_triggers_200char - any trigger word in paper_triggers within 200 characters of any keyword in ade_drug_single found in discharge summary",
    "11": "lf_paper_triggers_200char_negate - any trigger word in paper_triggers within 200 characters of any keyword in negate found in discharge summary",
    "12": "lf_paper_triggers_25words - any trigger word in paper_triggers within 25 words of any keyword in ade_drug_single found in discharge summary"
}

#################
### argparser ###
#################

### create the parser ###
parser = argparse.ArgumentParser(description='', formatter_class=argparse.RawTextHelpFormatter)

### add the arguments ###
# data
parser.add_argument('data_path', type=str,
    help='''
        data path of folder with:
        - train_txt: folder with discharge summaries for train dataset
        - train_ann: folder with annotated files (derived from discharge summaries) for train dataset
        - test_txt: folder with discharge summaries for test dataset
        - test_ann: folder with annotated files (derived from discharge summaries) for test dataset
        ''')
# model
parser.add_argument('models', type=str, 
    help='''    
        FORMAT: use , to separate models (eg. 1,6)
        0. No Models Selected
        1. ADE-Only Prediction using Linear SVM
        2. ADE-Only Prediction using Polynomial SVM
        3. ADE-Only Prediction using Sigmoid SVM
        4. ADE-Only Prediction using RBF SVM
        5. ADE-Only Prediction using GridSearch SVM
        6. ADE-Drug Prediction using Linear SVM
        7. ADE-Drug Prediction using Polynomial SVM
        8. ADE-Drug Prediction using Sigmoid SVM
        9. ADE-Drug Prediction using RBF SVM
        10. ADE-Drug Prediction using GridSearch SVM
        11. Reason-Drug Prediction using Linear SVM
        12. Reason-Drug Prediction using Polynomial SVM
        13. Reason-Drug Prediction using Sigmoid SVM
        14. Reason-Drug Prediction using RBF SVM
        15. Reason-Drug Prediction using GridSearch SVM
        16. Multi-Label Prediction using One vs Rest
        17. Multi-Label Prediction using Binary Relevance
        18. Multi-Label Prediction using Classifier Chains
        19. Multi-Label Prediction using Label Powerset
        20. Multi-Label Prediction using Adapted Algorithm
        21. Flair
        ''')
# rules
parser.add_argument('rules', type=str, 
    help='''
        FORMAT: use , to separate rules (eg. 3,8,12)
        0: No Rules Selected
        1: lf_ade_drug_single - any keywords in ade_drug_single found in discharge summary 
        2: lf_ade_drug_pair - any pair of keywords in ade_drug_pair found in discharge summary 
        3: lf_ade_drug_pair_lem - any pair of lemmatised keywords in ade_drug_pair found in discharge summary 
        4: lf_ade_drug_pair_lem_keyword_triggers - any pair of lemmatised keywords in ade_drug_pair and any trigger word in keyword_triggers found in discharge summary
        5: lf_sider2_triggers - any pair of trigger words in sider2_triggers found in discharge summary
        6: lf_sider2_triggers_25words - any pair of trigger words in sider2_triggers within 25 words of each other found in discharge summary
        7: lf_semmeddb_triggers - any pair of trigger words in semmeddb_triggers found in discharge summary
        8: lf_keyword_triggers - any trigger word in keyword_triggers found in discharge summary
        9: lf_paper_triggers - any trigger word in paper_triggers found in discharge summary
        10: lf_paper_triggers_200char - any trigger word in paper_triggers within 200 characters of any keyword in ade_drug_single found in discharge summary
        11: lf_paper_triggers_200char_negate - any trigger word in paper_triggers within 200 characters of any keyword in negate found in discharge summary
        12: lf_paper_triggers_25words - any trigger word in paper_triggers within 25 words of any keyword in ade_drug_single found in discharge summary
        ''')

### execute the parse_args() method ###
args = parser.parse_args()

data_path = args.data_path
input_models = args.models
input_rules = args.rules

lfs= []

# get models #
print("models selected:")
models_list = input_models.split(",")
for i in range(0, len(models_list)) :
    print(models_list[i], ":", models_dict_desc[models_list[i]])
    lfs.append(models_dict[models_list[i]])

print("\n")
print("rules selected:")
# get rules #
rules_list = input_rules.split(",")
for i in range(0, len(rules_list)) :
    print(rules_list[i], ":", rules_dict_desc[rules_list[i]])
    if rules_list[i] != "0" :
        lfs.append(rules_dict[rules_list[i]])

###########################
### discharge summaries ###
###########################

# dataset 
patient_list = []
summary = []
for file in os.listdir(data_path + "/all_txt"):
    if file != ".DS_Store" :
        with open(join(data_path + "/all_txt", file), 'r') as document_summary_file:
            data = " ".join(line.strip() for line in document_summary_file)
            patient_list.append(file[:-4])
            summary.append(data)

# get train discharge summaries labels
patient_label = []  
labelled = []
for file in os.listdir(data_path + "/all_ann") :
    with open(join(data_path + "/all_ann", file), 'rb') as document_anno_file:
        lines = document_anno_file.readlines()
        patient = file[:-4]
        boolean = False
        for line in lines :
            if b"ADE-Drug" in line:
                boolean = True
        # label which has ADE-Drug
        if boolean == True :
            labelled.append(1)
        else :
            labelled.append(0)
        patient_label.append(patient)

# attach labels to discharge summaries datasets
summary_df = pd.DataFrame({"patient": patient_list, "summary": summary})
label_df = pd.DataFrame({"patient": patient_label, "label": labelled})
data_df = summary_df.merge(label_df, on=['patient'], how='left')

# clean discharge summaries
if not sys.warnoptions:
    warnings.simplefilter("ignore")

def cleanHtml(sentence):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', str(sentence))
    return cleantext

def cleanPunc(sentence): #function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")
    return cleaned

def keepAlpha(sentence):
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent

# remove stopwords
stop_words = set(stopwords.words('english'))
stop_words.update(['zero','one','two','three','four','five','six','seven','eight','nine','ten','may','also','across','among','beside','however','yet','within'])
re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)
def removeStopWords(sentence):
    global re_stop_words
    return re_stop_words.sub(" ", sentence)
stemmer = SnowballStemmer("english")

def stemming(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence

# iterate through data_df.data
data_df_cleaned = data_df.copy()
for i in range(0, len(data_df_cleaned.summary)) :
    data_df_cleaned.summary[i] = data_df_cleaned.summary[i].lower()
    data_df_cleaned.summary[i] = cleanHtml(data_df_cleaned.summary[i])
    data_df_cleaned.summary[i] = cleanPunc(data_df_cleaned.summary[i])
    data_df_cleaned.summary[i] = keepAlpha(data_df_cleaned.summary[i])
    data_df_cleaned.summary[i] = removeStopWords(data_df_cleaned.summary[i])
    # data_df_cleaned.summary[i] = stemming(data_df_cleaned.summary[i])
# rename summary columns
data_df_cleaned = data_df_cleaned.rename(columns = {'summary': 'cleaned_summary'})
# left join the cleaned summary with the original
data_df= pd.merge(data_df, data_df_cleaned, on=['patient', 'label'])
# split by label
data_0 = data_df.loc[data_df['label']==0]
data_0 = data_0.reset_index(drop=True)
data_1 = data_df.loc[data_df['label']==1]
data_1 = data_1.reset_index(drop=True)

# len data_0 = 113
# len data_1 = 392

print("\n")
print("dataset cleaned")

#############################
###  Stratified 5 Fold CV ###
#############################
# shuffle dataframe
data_0 = shuffle(data_0)
data_0 = data_0.reset_index(drop=True)
data_1 = shuffle(data_1)
data_1 = data_1.reset_index(drop=True)

# split into 5 groups
g1_0 = data_0[0:22]
g2_0 = data_0[22:44]
g3_0 = data_0[44:66]
g4_0 = data_0[66:88]
g5_0 = data_0[88:]
g1_1 = data_1[0:78]
g2_1 = data_1[78:156]
g3_1 = data_1[156:234]
g4_1 = data_1[234:312]
g5_1 = data_1[312:]

# rejoin
g1 = pd.concat([g1_0, g1_1])
g2 = pd.concat([g2_0, g2_1])
g3 = pd.concat([g3_0, g3_1])
g4 = pd.concat([g4_0, g4_1])
g5 = pd.concat([g5_0, g5_1])

###################
###  LF applier ###
###################
applier = PandasLFApplier(lfs=lfs) 

def cv(train_list, testdata, i, log_df) :
    traindata = pd.concat(train_list)

    # count labels
    train_0 = len(traindata.loc[traindata['label']==0])
    train_1 = len(traindata.loc[traindata['label']==1])
    test_0 = len(testdata.loc[testdata['label']==0])
    test_1 = len(testdata.loc[testdata['label']==1])

    L_train = applier.apply(df=traindata)
    L_test = applier.apply(df=testdata)

    # label model
    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L_train=L_train, n_epochs=500,
                    lr=0.001, log_freq=100, seed=42)
    # weights of labeling functions used
    label_model_weights = np.around(label_model.get_weights(), 2)  
    # prediction
    normal_labels = label_model.predict(L_test)
    # metrics
    lm_metric_accuracy = metrics.accuracy_score(testdata.label, normal_labels)
    lm_metric_precision = metrics.precision_score(testdata.label, normal_labels)
    lm_metric_recall = metrics.recall_score(testdata.label, normal_labels)
    lm_metric_roc_auc = metrics.roc_auc_score(testdata.label, normal_labels)
    lm_f1 = metrics.f1_score(testdata.label, normal_labels)
    lm_metric_cm = metrics.confusion_matrix(testdata.label, normal_labels)
    # print
    print("accuracy:", lm_metric_accuracy)
    print("precision:", lm_metric_precision)
    print("recall:", lm_metric_recall)
    print("ROC AUC:", lm_metric_roc_auc)
    print("f1 score:", lm_f1)
    print("confusion matrix:")
    print(lm_metric_cm)

    # majority label voter
    majority_model = MajorityLabelVoter(cardinality=2)
    # prediction
    majority_labels = majority_model.predict(L_test)
    # metrics
    mlv_metric_accuracy = metrics.accuracy_score(testdata.label, majority_labels)
    mlv_metric_precision = metrics.precision_score(testdata.label, majority_labels)
    mlv_metric_recall = metrics.recall_score(testdata.label, majority_labels)
    mlv_metric_roc_auc = metrics.roc_auc_score(testdata.label, majority_labels)
    mlv_f1 = metrics.f1_score(testdata.label, majority_labels)
    mlv_metric_cm = metrics.confusion_matrix(testdata.label, majority_labels)
    # print
    print("accuracy:", mlv_metric_accuracy)
    print("precision:", mlv_metric_precision)
    print("recall:", mlv_metric_recall)
    print("ROC AUC:", mlv_metric_roc_auc)
    print("f1 score:", mlv_f1)
    print("confusion matrix:")
    print(mlv_metric_cm)

    ### update log.csv ###
    new_log = {'run': i, 'models': input_models, 'rules': input_rules, 'train_0': train_0, 'train_1': train_1, 'test_0': test_0, 'test_1': test_1,
    'lm_accuracy': lm_metric_accuracy, 'lm_precision': lm_metric_precision, 'lm_recall': lm_metric_recall, 'lm_ROCAUC': lm_metric_roc_auc, 'lm_f1': lm_f1, 'lm_CM_TN': lm_metric_cm[0][0], 'lm_CM_FP': lm_metric_cm[0][1], 'lm_CM_FN': lm_metric_cm[1][0], 'lm_CM_TP': lm_metric_cm[1][1],
    'mlv_accuracy': mlv_metric_accuracy, 'mlv_precision': mlv_metric_precision, 'mlv_recall': mlv_metric_recall, 'mlv_ROCAUC': mlv_metric_roc_auc, 'mlv_f1': mlv_f1, 'mlv_CM_TN': mlv_metric_cm[0][0], 'mlv_CM_FP': mlv_metric_cm[0][1], 'mlv_CM_FN': mlv_metric_cm[1][0], 'mlv_CM_TP': mlv_metric_cm[1][1]}
    log_df = log_df.append(new_log, ignore_index=True)
    return log_df

log_df = cv([g2, g3, g4, g5], g1, 1, log_df)
log_df = cv([g1, g2, g3, g4], g2, 2, log_df)
log_df = cv([g1, g2, g4, g5], g3, 3, log_df)
log_df = cv([g1, g2, g3, g5], g4, 4, log_df)
log_df = cv([g1, g2, g3, g4], g5, 5, log_df)

log_df.to_csv("../outputs/log_cv.csv", index=False)
