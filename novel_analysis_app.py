import jieba
import streamlit as st
from functions import (
    load_stop_words,
    basic_statistics,
    load_text,
    preprocess_text,
    word_frequency_analysis,
    generate_wordcloud,
    analyze_keyword_distribution,
)


def main():
    st.title("Novel Text Analysis")
    st.write(
        "Upload a .txt file for analysis of word frequency, word cloud, and keyword distribution."
    )

    uploaded_file = st.file_uploader("Choose a .txt file", type="txt")
    if uploaded_file is not None:
        # Read the content of the uploaded file as a string
        text = uploaded_file.read().decode(
            "utf-8"
        )  # Decode the file content to a string

        basic_statistics(text)
        words = jieba.lcut(text)
        stop_words = load_stop_words()
        words = preprocess_text(text, stop_words)
        words = [word for word in words if len(word) > 1 and word not in stop_words]

        # Word Frequency Analysis
        st.header("Word Frequency Analysis")
        word_freq_image = word_frequency_analysis(words)
        st.image(word_freq_image, caption="Word Frequency Analysis")

        # Word Cloud Generation
        st.header("Word Cloud")
        wordcloud_image = generate_wordcloud(words)
        st.image(wordcloud_image, caption="Word Cloud")

        # Keyword Distribution
        st.header("Keyword Frequency Distribution")
        keyword_dist_image = analyze_keyword_distribution(words, [])
        st.image(keyword_dist_image, caption="Keyword Frequency Distribution")


if __name__ == "__main__":
    main()
