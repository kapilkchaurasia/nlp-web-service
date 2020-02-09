
import os
import time
import pandas as pd
import codecs
import re
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class ClassifierTraining:
    """
    This class contains the code to train the model.
    """

    ## [EVALUATE] Prepare Metrics
    @staticmethod
    def get_metrics(y_test, y_predicted):
        # true positives / (true positives+false positives)
        precision = precision_score(y_test, y_predicted, pos_label=None,
                                average='weighted')
        # true positives / (true positives + false negatives)
        recall = recall_score(y_test, y_predicted, pos_label=None,
                          average='weighted')

        # harmonic mean of precision and recall
        f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')

        # true positives + true negatives/ total
        accuracy = accuracy_score(y_test, y_predicted)
        return accuracy, precision, recall, f1

    ## [PREPROCESS] Text Cleaning
    @staticmethod
    def standardize_text(df, text_field):
        # normalize by turning all letters into lowercase
        df[text_field] = df[text_field].str.lower()
        # get rid of URLS
        df[text_field] = df[text_field].apply(lambda elem: re.sub(r"http\S+", "", elem))
        return df

    @staticmethod
    def train_classifier():
        """
        This method loads the data, trains the model and saves the model in path "resources/model".
        """
        # Step 01: Set the data paths
        resources_path = "resources/"
        input_path = "data/input.data"
        output_directory = "model"
        model_output_file = "model.pkl"
        vc_output_file = "vc.pkl"

        # Step 02: Retrieve the dataset from some local fs source
        try:
            input_file = codecs.open(resources_path+input_path, "r",encoding='utf-8', errors='replace')
        except Exception as exc:
            raise Exception("Error occurred while loading the input datafile: " + str(exc))

        questions = pd.read_csv(input_file)
        questions.columns=['id', 'text', 'choose_one', 'class_label']
        questions.drop(['id' ], axis=1, inplace=True)
        print ('DONE - [ETL] Import Data')

        # Step 03:[PREPROCESS] Cleaning the dataset
        # Call the text cleaning function
        clean_questions = ClassifierTraining.standardize_text(questions, "text")
        print ('DONE - [PREPROCESS] Text Cleaning')

        ## Step 04: [PREPROCESS] Tokenize
        tokenizer = RegexpTokenizer(r'\w+')
        clean_questions["tokens"] = clean_questions["text"].apply(tokenizer.tokenize)

        ## Step 05: Creating a test and train dataset
        list_corpus = clean_questions["text"]
        list_labels = clean_questions["class_label"]

        X_train, X_test, y_train, y_test = train_test_split(list_corpus, list_labels, test_size=0.2, random_state=40)

        print("Training set: %d samples" % len(X_train))
        print("Test set: %d samples" % len(X_test))

        ## Step 06: [EMBEDDING] Tranform Tweets to BOW Embedding
        count_vectorizer = CountVectorizer(analyzer='word', token_pattern=r'\w+')
        bow = dict()
        bow["train"] = (count_vectorizer.fit_transform(X_train), y_train)
        bow["test"]  = (count_vectorizer.transform(X_test), y_test)
        print(bow["train"][0].shape)
        print(bow["test"][0].shape)

        ##  Step 07: [CLASSIFY] Initialize Logistic Regression and Fitting a model
        lr_classifier = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg',
                                   multi_class='multinomial', random_state=40)

        lr_classifier.fit(*bow["train"])
        print ('DONE - [CLASSIFY] Train Classifier on Embeddings')

        ## Step 08: Validating the model
        ## Predict on our Test Data so we can score our Model.
        y_predict = lr_classifier.predict(bow["test"][0])

        accuracy, precision, recall, f1 = ClassifierTraining.get_metrics(bow["test"][1], y_predict)
        print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

        ## Step 09: Persist/Update the serialized model to local FS
        model_directory = os.path.join(resources_path, output_directory)
        model_file_path = os.path.join(model_directory, model_output_file)
        vc_file_path = os.path.join(model_directory, vc_output_file)

        try:
            joblib.dump(lr_classifier, model_file_path)
            joblib.dump(count_vectorizer, vc_file_path)
        except Exception as exc:
            raise Exception("Error occurred while loading the classifier model: " + str(exc))

if __name__ == "__main__":
    tic = time.time()
    ClassifierTraining.train_classifier()
    toc = time.time()
    print("\nTotal time taken to train the classifier:", toc - tic, "seconds.")
