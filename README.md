# Patient Level Predictor

Adverse drug events (ADEs) are unwanted and dangerous side effects to medical treatment which is why it is such a pressing issue to tackle in healthcare. Hospital discharge summaries are known to be full of information with mentions of ADEs. However, given the sheer number of patients, it is infeasible for medical experts to analyze every discharge summary to find these ADE mentions. This report describes the implementation of a pipeline that identifies ADEs in unstructured hospital discharge summaries using a mixture of supervised models and rules. In addition, semi-supervised learning methods were used to generate labels to train the supervised models rather than rely on the prohibitively slow and expensive manual labelling. Replicating the National NLP Clinical Challenges (N2C2) in 2018 with a subset of 505 discharge summaries were used from the Medical Information Mart for Intensive Care 3 (MIMIC3) clinical database, a Patient Level Predictor (PLP) was created to predict if a discharge summary has a mention of any ADEs. Our experiments have shown that the inclusion of rules mined improves the performance of PLP compared to just models alone and that the PLP can be used in improving the performance of ADE mining on a state-of-the- art model.


Google Drive Link for the following: https://drive.google.com/drive/folders/1nR_Y6Ht-v--aPGzwCj4cB65HM7xINgeQ?usp=sharing

- amie
- annotate
- clinicalnlp-ade-master
- data
- DRUM-master
- Jupyter Notebooks
- MIMIC
- MIMIC_data
- models
- N2C2
- User Manual
