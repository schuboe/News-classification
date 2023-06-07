# News-classification
Code and datasets for text classification of news articles (language detection, topic identification, web scraping, Streamlit app)

**Text classification** refers to the categorization of text documents based on their content. In this project, I addressed two text classification problems: language detection and topic identification.

For **language detection**, I trained a multinomial Naive Bayes classifier using unigrams and bigrams of characters within words. The model can detect English, German, French, and Spanish. The folder "language_detection" contains the notebook (language_detection_NB.ipynb) and the dataset used for training (Language_Detection.csv).

**Topic identification** models were trained for the content of German and English news articles, distinguishing the following topic categories: Business, Entertainment & Arts, Media, Panorama, Politics, Science, Sport, and Tech.

Streamlit app

Web scraping
