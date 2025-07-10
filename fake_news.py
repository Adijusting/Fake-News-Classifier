import pandas as pd
import numpy as np
import string
import re
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import PassiveAggressiveClassifier
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Multiple ML models to train and see which gives us best result
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

"""
Firstly we load both the datastes and concat them to randomize 
"""
# Loading the dataset
df = pd.read_csv('C:/Users/Aditya/OneDrive/Desktop/Python_projs/Fake_news_classifier/Fake.csv', usecols=['title','text'],low_memory=False)
print(df.head())
df['label'] = 0 #Assign label 0 for Fake news

# Load real news dataset
real_df = pd.read_csv('C:/Users/Aditya/OneDrive/Desktop/Python_projs/Fake_news_classifier/True.csv', usecols=['title','text'], low_memory=False)
print(real_df.head)
real_df['label'] = 1 #Assign label 1 to real news

# Combining both datasets
df = pd.concat([df, real_df], axis=0)
# df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Keep only useful columns
df = df[['title','text','label']]

# Shuffle the data to randomize the order
df = df.sample(frac=1).reset_index(drop=True)

"""
Now we normalize and clean the data so that ML model can process it effectively
"""

stop_words = set(stopwords.words('english'))
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^a-zA-Z]',' ', text)
    text=text.lower()
    text=text.split()
    text=[word for word in text if word not in stop_words]
    return ' '.join(text)

# Apply text cleaning on the dataset 'text' column
tqdm.pandas()
df['clean_text'] = df['text'].progress_apply(clean_text)


"""
Converting cleaned text data into numerical features that our ML model can understand
"""

# Initialize the vectorizer
vectorizer = TfidfVectorizer(stop_words = 'english', max_df=0.7)
x = vectorizer.fit_transform(df['clean_text'])
y = df['label']

x_train, x_test, y_train, y_test = train_test_split(x,y , test_size=0.3, random_state=42)

# Fit on training data transform both train and test sets


"""
Training ML model 
"""

# Inilialize the model
# classifier = PassiveAggressiveClassifier(max_iter=1000)

# # train
# classifier.fit(x_train_tfidf, y_train)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "PassiveAggresiveClassifier": PassiveAggressiveClassifier(max_iter=1000),
    "Multinomial Naive Bayes": MultinomialNB(),
    # "Random Forest": RandomForestClassifier(n_estimators=100),
    "Linear SVM": LinearSVC(),
    # "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
}

# Training each model
for name, model in models.items():
    print(f"\nTraining: {name}")
    model.fit(x_train, y_train)
    y_pred=model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy score of {name}: {accuracy*100:.4f}")

"""
Checking the accuracy of model and confusion matrix
"""

# Prediting the labels
y_pred = model.predict(x_test)

# Evaluating accuracy
accuracy = accuracy_score(y_test, y_pred)

# conf_mat = confusion_matrix(y_test, y_pred)

def predict_news(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    label = model.predict(vectorized)[0]
    if label==1:
        print("Real news")
    else:
        print("Fake news")