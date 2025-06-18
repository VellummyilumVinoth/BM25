import string
import nltk
from typing import List
from nltk import word_tokenize, SnowballStemmer
from nltk.corpus import stopwords


class BM25Tokenizer:
    def __init__(
            self,
            lower_case: bool,
            remove_punctuation: bool,
            remove_stopwords: bool,
            stem: bool,
            language: str,
    ):
        self.nltk_setup()
        self.lower_case = lower_case
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.stem = stem
        self.language = language
        self._stemmer = SnowballStemmer(language)
        self._stop_words = set(stopwords.words(language))
        self._punctuation = set(string.punctuation)

        if self.stem and not self.lower_case:
            raise ValueError(
                "Stemming applies lower case to tokens, so lower_case must be True if stem is True"
            )

    @staticmethod
    def nltk_setup() -> None:
        """Download required NLTK data if not already present"""
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")

        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords")

    def __call__(self, text: str) -> List[str]:
        """
        Tokenize and process text according to the configured options

        Args:
            text: Input text to tokenize

        Returns:
            List of processed tokens
        """
        # Tokenize the text
        tokens = word_tokenize(text, self.language)

        # Apply lowercase if enabled
        if self.lower_case:
            tokens = [word.lower() for word in tokens]

        # Remove punctuation if enabled
        if self.remove_punctuation:
            tokens = [word for word in tokens if word not in self._punctuation]

        # Remove stopwords if enabled
        if self.remove_stopwords:
            if self.lower_case:
                tokens = [word for word in tokens if word not in self._stop_words]
            else:
                tokens = [
                    word for word in tokens if word.lower() not in self._stop_words
                ]

        # Apply stemming if enabled
        if self.stem:
            tokens = [self._stemmer.stem(word) for word in tokens]

        return tokens