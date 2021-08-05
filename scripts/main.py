'''
COMMAND LINE EXAMPLE:
python3 main.py ../N2C2 1,6 3,8,12
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
from datetime import datetime

# for cleaning discharge summaries
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import warnings

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

if Path("../outputs/log.csv").exists() :
    log_df = pd.read_csv("../outputs/log.csv")
else :
    log_df = pd.DataFrame({'folder_name': [], 'models': [], 'rules': [], 
    'lm_accuracy': [], 'lm_precision': [], 'lm_recall': [], 'lm_ROCAUC': [], 'lm_f1': [], 'lm_CM_TN': [], 'lm_CM_FP': [], 'lm_CM_FN': [], 'lm_CM_TP': [],
    'mlv_accuracy': [], 'mlv_precision': [], 'mlv_recall': [], 'mlv_ROCAUC': [], 'mlv_f1': [], 'mlv_CM_TN': [], 'mlv_CM_FP': [], 'mlv_CM_FN': [], 'mlv_CM_TP': []})

################################
### create folder for output ###
################################
try :
    run_folder = "run01"
    run_folder_path = "../outputs/" + run_folder + "/"
    os.mkdir(run_folder_path)
except :
    dir_list = [i[0] for i in os.walk("../outputs/")]
    dir_list.sort(reverse=True)
    run_number = int(dir_list[0][-2:]) + 1
    if run_number < 10: 
        run_number = '0' + str(run_number)
    run_folder = "run" + str(run_number)
    run_folder_path = "../outputs/" + run_folder + "/"
    os.mkdir(run_folder_path)

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

### save ###
info = pd.DataFrame({'models': [input_models], 'rules': [input_rules]})
info.to_csv(run_folder_path + 'info.csv', index=False)

### print ###
print("\n")
print("output folder path:")
print(run_folder_path)
print("\n")
lfs= []

# get models #
print("models selected:")
models_list = input_models.split(",")
for i in range(0, len(models_list)) :
    print(models_list[i], ":", models_dict_desc[models_list[i]])
    if models_list[i] != "0" :
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

train_bool = False
test_bool = False

# train dataset 
train_txt_folder = data_path + "/train_txt"
if Path(train_txt_folder).exists() :
    train_bool = True
    train_patient = []
    train_summary = []
    for file in os.listdir(train_txt_folder):
        with open(join(train_txt_folder, file), 'r') as document_summary_file:
            data = " ".join(line.strip() for line in document_summary_file)
            train_patient.append(file[:-4])
            train_summary.append(data)

    # get train discharge summaries labels
    train_patient_label = []  
    train_labelled = []
    train_ADE = []
    for file in os.listdir(data_path + "/train_ann") :
        with open(join(data_path + "/train_ann", file), 'rb') as document_anno_file:
            lines = document_anno_file.readlines()
            patient = file[:-4]
            boolean = False
            for line in lines :
                if b"ADE-Drug" in line:
                    boolean = True
            # label which has ADE-Drug
            if boolean == True :
                train_labelled.append(1)
            else :
                train_labelled.append(0)
            train_patient_label.append(patient)
else :
    train_bool = False

# test dataset 
test_txt_folder = data_path + "/test_txt"
if Path(test_txt_folder).exists():
    test_bool = True
    test_patient = []
    test_summary = []
    for file in os.listdir(test_txt_folder):
        with open(join(test_txt_folder, file), 'r') as document_summary_file:
            data = " ".join(line.strip() for line in document_summary_file)
            test_patient.append(file[:-4])
            test_summary.append(data)
    # get test discharge summaries labels
    test_ann_folder = data_path + "/test_ann"
    test_patient_label = []
    test_labelled = []
    for file in os.listdir(test_ann_folder) :
        with open(join(test_ann_folder, file), 'rb') as document_anno_file:
            lines = document_anno_file.readlines()
            patient = file[:-4]
            boolean = False
            for line in lines :
                if b"ADE-Drug" in line:
                    boolean = True
            # label which has ADE-Drug
            if boolean == True :
                test_labelled.append(1)
            else :
                test_labelled.append(0)
            test_patient_label.append(patient)
else :
    test_bool = False 

# attach labels to discharge summaries datasets
if train_bool :
    summary_traindf = pd.DataFrame({"patient": train_patient, "summary": train_summary})
    label_traindf = pd.DataFrame({"patient": train_patient_label, "label": train_labelled})
    trainData = summary_traindf.merge(label_traindf, on=['patient'], how='left')

if test_bool :
    summary_testdf = pd.DataFrame({"patient": test_patient, "summary": test_summary})
    label_testdf = pd.DataFrame({"patient": test_patient_label, "label": test_labelled})
    testData = summary_testdf.merge(label_testdf, on=['patient'], how='left')

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

# train
if train_bool :
    # iterate through trainData.data
    trainData_cleaned = trainData.copy()
    for i in range(0, len(trainData_cleaned.summary)) :
        trainData_cleaned.summary[i] = trainData_cleaned.summary[i].lower()
        trainData_cleaned.summary[i] = cleanHtml(trainData_cleaned.summary[i])
        trainData_cleaned.summary[i] = cleanPunc(trainData_cleaned.summary[i])
        trainData_cleaned.summary[i] = keepAlpha(trainData_cleaned.summary[i])
        trainData_cleaned.summary[i] = removeStopWords(trainData_cleaned.summary[i])
        # trainData_cleaned.summary[i] = stemming(trainData_cleaned.summary[i])
    # rename summary columns in trainData and testData
    trainData_cleaned = trainData_cleaned.rename(columns = {'summary': 'cleaned_summary'})
    # left join the cleaned summary with the original trainData and testData
    trainData = pd.merge(trainData, trainData_cleaned, on=['patient', 'label'])
    print("\n")
    print("train dataset cleaned")
else :
    print("\n")
    print("no train dataset found")

# iterate through testData.data
if test_bool :
    testData_cleaned = testData.copy()
    for i in range(0, len(testData_cleaned.summary)) :
        testData_cleaned.summary[i] = testData_cleaned.summary[i].lower()
        testData_cleaned.summary[i] = cleanHtml(testData_cleaned.summary[i])
        testData_cleaned.summary[i] = cleanPunc(testData_cleaned.summary[i])
        testData_cleaned.summary[i] = keepAlpha(testData_cleaned.summary[i])
        testData_cleaned.summary[i] = removeStopWords(testData_cleaned.summary[i])
        # testData_cleaned.summary[i] = stemming(testData_cleaned.summary[i])
    # rename summary columns in trainData and testData
    testData_cleaned = testData_cleaned.rename(columns = {'summary': 'cleaned_summary'})
    # left join the cleaned summary with the original trainData and testData
    testData = pd.merge(testData, testData_cleaned, on=['patient', 'label'])
    print("test dataset cleaned")
else :
    print("no test dataset found")

###################
###  LF applier ###
###################

applier = PandasLFApplier(lfs=lfs) 
if train_bool :
    L_train = applier.apply(df=trainData)
    print("\n")
    print("Labeling Function Analysis on train dataset")
    print(f"{LFAnalysis(L_train, lfs).lf_summary()}")
if test_bool :
    L_test = applier.apply(df=testData)

#####################
###  Label Models ###
#####################

if train_bool :
    ### Label Model ###
    print("\n")
    print("###################")
    print("### Label Model ###")
    print("###################")
    # define model
    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L_train=L_train, n_epochs=500,
                    lr=0.001, log_freq=100, seed=42)
    # weights of labeling functions used
    label_model_weights = np.around(label_model.get_weights(), 2)  
    # prediction
    if test_bool :
        normal_labels = label_model.predict(L_test)
        # metrics
        lm_metric_accuracy = metrics.accuracy_score(testData.label, normal_labels)
        lm_metric_precision = metrics.precision_score(testData.label, normal_labels)
        lm_metric_recall = metrics.recall_score(testData.label, normal_labels)
        lm_metric_roc_auc = metrics.roc_auc_score(testData.label, normal_labels)
        lm_f1 = metrics.f1_score(testData.label, normal_labels)
        lm_metric_cm = metrics.confusion_matrix(testData.label, normal_labels)
        # print
        print("accuracy:", lm_metric_accuracy)
        print("precision:", lm_metric_precision)
        print("recall:", lm_metric_recall)
        print("ROC AUC:", lm_metric_roc_auc)
        print("f1 score:", lm_f1)
        print("confusion matrix:")
        print(lm_metric_cm)
    # save model as pickle file
    filename_lm = '../outputs/' + run_folder + '/label_model.pkl'
    with open(filename_lm, 'wb') as file:  
        pickle.dump(label_model, file)
    print("label model saved as ../outputs/" + run_folder + "/label_model.pkl")

    ### Majority Label Voter ###
    print("\n")
    print("############################")
    print("### Majority Label Voter ###")
    print("############################")
    # define model
    majority_model = MajorityLabelVoter(cardinality=2)
    # prediction
    if test_bool :
        majority_labels = majority_model.predict(L_test)
        # metrics
        mlv_metric_accuracy = metrics.accuracy_score(testData.label, majority_labels)
        mlv_metric_precision = metrics.precision_score(testData.label, majority_labels)
        mlv_metric_recall = metrics.recall_score(testData.label, majority_labels)
        mlv_metric_roc_auc = metrics.roc_auc_score(testData.label, majority_labels)
        mlv_f1 = metrics.f1_score(testData.label, majority_labels)
        mlv_metric_cm = metrics.confusion_matrix(testData.label, majority_labels)
        # print
        print("accuracy:", mlv_metric_accuracy)
        print("precision:", mlv_metric_precision)
        print("recall:", mlv_metric_recall)
        print("ROC AUC:", mlv_metric_roc_auc)
        print("f1 score:", mlv_f1)
        print("confusion matrix:")
        print(mlv_metric_cm)
    # save model as pickle file
    filename_mlv = '../outputs/' + run_folder + '/majority_label_voter.pkl'
    with open(filename_mlv, 'wb') as file:  
        pickle.dump(majority_model, file)
    print("majority label voter saved as ../outputs/" + run_folder + "/majority_label_voter.pkl")

    ### update log.csv ###
    new_log = {'folder_name': run_folder, 'models': input_models, 'rules': input_rules, 
    'lm_accuracy': lm_metric_accuracy, 'lm_precision': lm_metric_precision, 'lm_recall': lm_metric_recall, 'lm_ROCAUC': lm_metric_roc_auc, 'lm_f1': lm_f1, 'lm_CM_TN': lm_metric_cm[0][0], 'lm_CM_FP': lm_metric_cm[0][1], 'lm_CM_FN': lm_metric_cm[1][0], 'lm_CM_TP': lm_metric_cm[1][1],
    'mlv_accuracy': mlv_metric_accuracy, 'mlv_precision': mlv_metric_precision, 'mlv_recall': mlv_metric_recall, 'mlv_ROCAUC': mlv_metric_roc_auc, 'mlv_f1': mlv_f1, 'mlv_CM_TN': mlv_metric_cm[0][0], 'mlv_CM_FP': mlv_metric_cm[0][1], 'mlv_CM_FN': mlv_metric_cm[1][0], 'mlv_CM_TP': mlv_metric_cm[1][1]}
    log_df = log_df.append(new_log, ignore_index=True)
    log_df.to_csv("../outputs/log.csv", index=False)
    print("\n")
    print("log.csv updated")

    ### pick a label model ###
    print("\n")
    print("################")
    print("### Optional ###")
    print("################")
    optional = input("Would you like to use the Label Model / Majority Label Voter now?: Y/N ")

    if optional == 'Y' :
        model_selected = input("Enter 1 to use Label Model and 2 to use Majority Label Voter: ") 
        data_selected_path = input("Enter the full path to the folder with discharge summaries (.txt): ")
        
        # get new data
        new_data_patient = []
        new_data_summary = []
        for file in os.listdir(data_selected_path):
            if file != ".DS_Store" :
                with open(join(data_selected_path, file), 'r') as document_summary_file:
                    data = " ".join(line.strip() for line in document_summary_file)
                    new_data_patient.append(file[:-4])
                    new_data_summary.append(data)
        new_data_df = pd.DataFrame({"patient": new_data_patient, "summary": new_data_summary})

        # clean new_data_df discharge summaries
        new_data_df_cleaned = new_data_df.copy()
        new_data_df_cleaned['cleaned_summary'] = new_data_df_cleaned['summary']
        for i in range(0, len(new_data_df_cleaned)) :
            new_data_df_cleaned.cleaned_summary[i] = new_data_df_cleaned.cleaned_summary[i].lower()
            new_data_df_cleaned.cleaned_summary[i] = cleanHtml(new_data_df_cleaned.cleaned_summary[i])
            new_data_df_cleaned.cleaned_summary[i] = cleanPunc(new_data_df_cleaned.cleaned_summary[i])
            new_data_df_cleaned.cleaned_summary[i] = keepAlpha(new_data_df_cleaned.cleaned_summary[i])
            new_data_df_cleaned.cleaned_summary[i] = removeStopWords(new_data_df_cleaned.cleaned_summary[i])
            new_data_df_cleaned.cleaned_summary[i] = stemming(new_data_df_cleaned.cleaned_summary[i])
        
        print("\n")
        L_new_data = applier.apply(df=new_data_df_cleaned)
        print("new dataset cleaned")

        # label model
        if model_selected == "1" :
            # prediction
            new_normal_labels = label_model.predict(L_new_data)  
            all_normal_prob = label_model.predict_proba(L_new_data)
            new_normal_prob = []
            for i in range(0, len(new_normal_labels)) :
                if new_normal_labels[i] == 0 :
                    new_normal_prob.append(all_normal_prob[i][0])
                elif new_normal_labels[i] == 1 :
                    new_normal_prob.append(all_normal_prob[i][1])
            print("predicted labels generated")
            # add prediction to new_data_df    
            new_data_df['predicted_label'] = new_normal_labels
            new_data_df['predicted_probability'] = new_normal_prob
            # save as .csv   
            new_data_df.to_csv("../outputs/" + run_folder + "/" + input_models + "_" + input_rules + "_" + "label_model.csv", index=False)
            print("predicted labels saved as " + "../outputs/" + run_folder + "/" + input_models + "_" + input_rules + "_" + "label_model.csv")
            print("\n")
            print("###########")
            print("### END ###")
            print("###########")
            print("\n")
            
        # majority label voter
        elif model_selected == "2" :
            # prediction
            new_majority_labels = majority_model.predict(L_new_data)
            all_majority_prob = majority_model.predict_proba(L_new_data)
            new_majority_prob = []
            for i in range(0, len(new_majority_labels)) :
                if new_majority_labels[i] == 0 :
                    new_majority_prob.append(all_majority_prob[i][0])
                elif new_majority_labels[i] == 1 :
                    new_majority_prob.append(all_majority_prob[i][1])
            print("predicted labels generated")
            # add prediction and prediction probability to new_data_df
            new_data_df['predicted_label'] = new_majority_labels
            new_data_df['predicted_probability'] = new_majority_prob
            # save as .csv
            new_data_df.to_csv("../outputs/" + run_folder + "/" + input_models + "_" + input_rules + "_" + "majority_label_voter.csv", index=False)
            print("predicted labels saved as " + "../outputs/" + run_folder + "/" + input_models + "_" + input_rules + "_" + "majority_label_voter.csv")
            print("\n")
            print("###########")
            print("### END ###")
            print("###########")
            print("\n")

    else :
        print("\n")
        print("###########")
        print("### END ###")
        print("###########")
        print("\n")
else :
    print("\n")
    print("###########")
    print("### END ###")
    print("###########")
    print("\n")







