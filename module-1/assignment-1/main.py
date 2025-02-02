from textblob import TextBlob
from transformers import pipeline
from textblob.sentiments import NaiveBayesAnalyzer
import nltk

# Setup for NaiveBayesAnalyzer
nltk.download("movie_reviews")
nltk.download("punkt")
nltk.download("punkt_tab")


def sentiment_pattern_analyzer(text: str) -> float:
    return TextBlob(text).sentiment.polarity


def sentiment_naive_bayes_analyzer(text: str) -> str:
    classification, _, _ = TextBlob(text, analyzer=NaiveBayesAnalyzer()).sentiment
    return classification


def sentiment_llm_distilbert(text: str, sentiment_pipeline):
    return sentiment_pipeline(text)[0]


SAMPLE_TEXTS = [
    "I loved the movie. It was fantastic!",
    "The product is terrible. I would not recommend it.",
    "It’s horrible. I didn't like it.",
    "The view is not bad.",
]


def main():
    # Initialize model once (expensive)
    sentiment_model = pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    )

    print("\nSentiment Analysis Results:\n" + "=" * 40)

    for sentence in SAMPLE_TEXTS:
        pattern_polarity = sentiment_pattern_analyzer(sentence)
        naive_bayes_class = sentiment_naive_bayes_analyzer(sentence)
        llm_result = sentiment_llm_distilbert(sentence, sentiment_model)

        print(f"\nSentence: {sentence}")
        print(f"TextBlob Sentiment: {pattern_polarity:.2f}")
        print(f"Naive Bayes Classification: {naive_bayes_class}")
        print(f"Distilbert: {llm_result['label']} ({llm_result['score']:.4f})")

    print("\n" + "=" * 40)


if __name__ == "__main__":
    main()
