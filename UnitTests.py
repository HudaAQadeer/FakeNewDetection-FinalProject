import unittest
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure nltk resources are available
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

class TestPreprocessing(unittest.TestCase):
    
    def setUp(self):
        """ Set up test data before each test """
        self.text = "The quick brown fox jumped over the lazy dog."
        self.expected_tokens = ['The', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog', '.']
        self.expected_filtered_tokens = ['quick', 'brown', 'fox', 'jumped', 'lazy', 'dog', '.']
        self.expected_lemmatized = ['quick', 'brown', 'fox', 'jump', 'lazy', 'dog', '.']
        self.vectorizer = TfidfVectorizer()

    def test_tokenization(self):
        """ Tests if the sentence is tokenized """
        tokens = word_tokenize(self.text)
        self.assertEqual(tokens, self.expected_tokens)

    def test_stopword_removal(self):
        """ tests if stopwords are removed """
        tokens = word_tokenize(self.text)
        filtered_tokens = [word.lower() for word in tokens if word.lower() not in stopwords.words('english')]
        self.assertEqual(filtered_tokens, self.expected_filtered_tokens)


    def test_vectorization(self):
        """ tests if vectorization is being applied """
        corpus = ["The quick brown fox.", "Lazy dog jumped."]
        vectorized = self.vectorizer.fit_transform(corpus)
        self.assertEqual(vectorized.shape[0], len(corpus))  

if __name__ == '__main__':
    unittest.main()
