# NLP Tokenization Notes

> A compact, interview-ready record of the tokenization exercises in this project.

## What I practised

The notebook [Tokenization Example.ipynb](src/Tokenization%20Example.ipynb) explores how **NLTK** turns raw text into smaller units that an NLP pipeline can process.

```text
Paragraph → sentences → words/tokens
```

| Technique | NLTK API | What it demonstrates |
| --- | --- | --- |
| Sentence tokenization | `sent_tokenize` | Splitting a paragraph into sentence-level documents. |
| Word tokenization | `word_tokenize` | Splitting text or individual sentences into word and punctuation tokens. |
| Regex-based word/punctuation tokenization | `wordpunct_tokenize` | Treating punctuation separately using a simpler tokenization approach. |
| Treebank tokenization | `TreebankWordTokenizer` | Penn Treebank-style rules; useful for seeing how tokenization rules affect punctuation. |

## Key takeaways

- **Tokenization is a preprocessing step:** it converts unstructured text into units a model or downstream NLP task can consume.
- **Sentence tokenization** is helpful when each sentence should be processed as a separate document, for example before sentence-level analysis or chunking.
- **Word tokenization** can be applied to an entire paragraph or to each sentence after sentence splitting.
- **Punctuation handling is tokenizer-dependent.** The notebook compares `wordpunct_tokenize` with `TreebankWordTokenizer`, illustrating that token boundaries are determined by each tokenizer's rules.
- NLTK resources such as `punkt_tab` are downloaded before tokenization so the sentence tokenizer has the required language data.

## Interview prompt & answer

**Q: Why does the choice of tokenizer matter?**

**A:** A tokenizer defines the boundaries of the tokens passed to later stages. Its treatment of punctuation, contractions, abbreviations, and special characters changes the input representation, so it can affect feature extraction and ultimately model behaviour. Choose a tokenizer that matches the language, data, and model being used.

## Run the notebook

Install the project's dependencies, then start Jupyter Lab:

```bash
poetry run jupyter lab \
  --ip=0.0.0.0 \
  --port=8888 \
  --no-browser \
  --ServerApp.token=''
```

Open `src/Tokenization Example.ipynb` and run the cells in order. The notebook downloads the required NLTK datasets on first use.

## Learning source

These exercises follow the Udemy lesson [Complete Generative AI Course with LangChain and Hugging Face](https://capgemini.udemy.com/course/complete-generative-ai-course-with-langchain-and-huggingface/learn/lecture/45162113?start=15#overview).
