import chardet
import string
import jieba
import streamlit as st
from functions import (
    load_stop_words,
    basic_statistics,
    preprocess_text,
    word_frequency_analysis,
    generate_wordcloud,
    analyze_keyword_distribution,
)


# Encoding detection function
def detect_encoding(file_content):
    raw_data = file_content[:10000]  # Use the first 10KB for detection
    result = chardet.detect(raw_data)
    return result.get("encoding", None)  # Safely get the encoding or return None


# Function to load the text file using the detected encoding
def load_text(file_content):
    encoding = detect_encoding(file_content)
    if not encoding:
        raise ValueError(
            "Unable to detect encoding. Please ensure the file is a valid text file."
        )
    return file_content.decode(encoding, errors="ignore")


# Process the user input to match the desired format
def process_keywords(keywords_input):
    try:
        # Replace Chinese commas with English commas for consistent splitting
        keywords_input = keywords_input.replace("，", ",")

        # Split the input into groups by English comma
        keyword_groups = keywords_input.split(",")

        # Prepare the list to store the final result
        final_keywords = []

        # For each keyword group, split by '+' to form subgroups and add to final list
        for group in keyword_groups:
            subgroups = group.split("+")
            # Clean any extra spaces and add to the final list
            final_keywords.append([keyword.strip() for keyword in subgroups])

        return final_keywords
    except Exception as e:
        raise ValueError(f"Error processing keywords: {e}")


def main():
    st.title("Novel Text Analysis")
    st.write(
        "Upload a .txt file for analysis of word frequency, word cloud, and keyword distribution."
    )

    uploaded_file = st.file_uploader("Choose a .txt file", type="txt")
    keywords_input = st.text_area(
        "Enter keywords for analysis (in the format: keyword1+keyword2, keyword3, keyword4+keyword5)"
    )

    if uploaded_file is not None:
        # Read the uploaded file content as bytes
        file_content = uploaded_file.read()

        # Load the text using detected encoding
        try:
            text = load_text(file_content)
        except ValueError as e:
            st.error(str(e))
            return

        st.write(f"File successfully decoded with detected encoding.")

        # Display basic statistics (if implemented)
        stats = basic_statistics(text)
        st.subheader("Basic Statistics")
        st.text(stats)

        # Load stop words
        stop_words = load_stop_words()

        # Preprocess the text (e.g., tokenize and remove stop words)
        words = preprocess_text(text, stop_words)

        # Filter words: keep only valid words longer than 1 character
        words = [word for word in words if len(word) > 1 and word not in stop_words]
        words = [word for word in words if word.isalpha()]

        if not words:
            st.error(
                "The text contains no valid words after preprocessing. Please check your input."
            )
            return

        # Process the user-provided keywords
        if keywords_input:
            try:
                keywords = process_keywords(keywords_input)
                st.write(f"Processed keywords: {keywords}")
            except ValueError as e:
                st.error(str(e))
                return
        else:
            keywords = []

        # Reprocess the words before keyword frequency analysis
        if keywords:
            st.header("Keyword Frequency Analysis")
            keyword_freq_image = analyze_keyword_distribution(words, keywords)
            st.image(keyword_freq_image, caption="Keyword Frequency Analysis")

        # Reprocess the words before word frequency analysis
        st.header("Word Frequency Analysis")
        word_freq_image = word_frequency_analysis(words)
        st.image(word_freq_image, caption="Word Frequency Analysis")

        # Reprocess the words before word cloud generation
        st.header("Word Cloud")
        wordcloud_image = generate_wordcloud(words)
        st.image(wordcloud_image, caption="Word Cloud")

        # Reprocess the words before keyword distribution analysis
        st.header("Keyword Frequency Distribution")
        keyword_dist_image = analyze_keyword_distribution(words, [])
        st.image(keyword_dist_image, caption="Keyword Frequency Distribution")


if __name__ == "__main__":
    main()
