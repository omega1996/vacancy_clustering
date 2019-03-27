from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 5), analyzer='word')


def fit_transform_words(bag):
    answer = tfidf_vectorizer.fit_transform(bag)
    return answer


print(fit_transform_words(['hello', 'again', 'my', 'friend']))