import random


CSV_DATASETS = [
    {
        "name": "TREC_06",
        "path": "datasets/TREC_06.csv",
        "subject_col": "subject",
        "body_col": "body",
        "label_col": "label",
        "sender_col": "sender",
        "receiver_col": "receiver",
        "date_col" : "date",
        "pos_labels": [1],
        "neg_labels": [0]
    },
    {
        "name": "SpamAssasin",
        "path": "datasets/SpamAssasin.csv",
        "subject_col": "subject",
        "body_col": "body",
        "label_col": "label",
        "sender_col": "sender",
        "receiver_col": "receiver",
        "date_col" : "date",
        "pos_labels": [1],
        "neg_labels": [0]
    },
    {
        "name": "Nigerian_Fraud",
        "path": "datasets/Nigerian_Fraud.csv",
        "subject_col": "subject",
        "body_col": "body",
        "label_col": "label",
        "sender_col": "sender",
        "receiver_col": "receiver",
        "date_col" : "date",
        "pos_labels": [1],
        "neg_labels": [0]
    },
    {
        "name": "TREC_07",
        "path": "datasets/TREC_07.csv",
        "subject_col": "subject",
        "body_col": "body",
        "label_col": "label",
        "sender_col": "sender",
        "receiver_col": "receiver",
        "date_col" : "date",
        "pos_labels": [1],
        "neg_labels": [0]
    },
]

EML_DATASETS = [
    {
        "name": "epvme_phish",
        "root_dir": "headers/1",
        "label_for_all": 1,
    },
    {
        "name": "epvme_phish",
        "root_dir": "headers/2",
        "label_for_all": 1,
    },
    {
        "name": "epvme_phish",
        "root_dir": "headers/3",
        "label_for_all": 1,
    },
    {
        "name": "epvme_phish",
        "root_dir": "headers/4",
        "label_for_all": 1,
    },
    {
        "name": "enron",
        "root_dir": "headers/enron_down_53k",
        "label_for_all": 0,
    },
    {
        "name": "epvme_phish",
        "root_dir": "headers/5",
        "label_for_all": 1,
    },
    {
        "name": "spamassassin",
        "root_dir": "headers/spamassassin/hard_ham",
        "label_for_all": 0,
    },
    {
        "name": "spamassassin",
        "root_dir": "headers/spamassassin/hard_ham_3",
        "label_for_all": 0,
    },
    {
        "name": "phising_pot",
        "root_dir": "headers/phising_pot/email",
        "label_for_all": 1,
    },
    {
        "name": "spamassassin",
        "root_dir": "headers/spamassassin/easy_ham",
        "label_for_all": 0,
    },
    {
        "name": "spamassassin",
        "root_dir": "headers/spamassassin/easy_ham_2",
        "label_for_all": 0,
    },
    {
        "name": "spamassassin",
        "root_dir": "headers/spamassassin/easy_ham_3",
        "label_for_all": 0,
    },
    {
        "name": "spamassassin",
        "root_dir": "headers/spamassassin/spam",
        "label_for_all": 1,
    },
    {
        "name": "spamassassin",
        "root_dir": "headers/spamassassin/spam_2",
        "label_for_all": 1,
    },
    {
        "name": "spamassassin",
        "root_dir": "headers/spamassassin/spam_3",
        "label_for_all": 1,
    },
    {
        "name": "spamassassin",
        "root_dir": "headers/spamassassin/spam_4",
        "label_for_all": 1,
    },
    {
        "name": "wooyun_xss",
        "root_dir": "headers/wooyun_xss",
        "label_for_all": 1,
    },
]

def get_csv_datasets(seed=None):
    lst = CSV_DATASETS.copy()
    if seed is not None:
        random.seed(seed)
    random.shuffle(lst)
    return lst

def get_eml_datasets(seed=None):
    lst = EML_DATASETS.copy()
    if seed is not None:
        random.seed(seed)
    random.shuffle(lst)
    return lst