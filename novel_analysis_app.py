import streamlit as st
from functions import (
    load_stop_words,
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
        # Process the uploaded file
        text = load_text(uploaded_file)
        stop_words = load_stop_words()
        words = preprocess_text(text, stop_words)

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
