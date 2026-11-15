# PromptForge — Final Reflection (Weeks 1–4)

The four PromptForge projects covered different parts of a document-question-answering system. The work started with calling an LLM, moved to semantic retrieval, combined retrieval with generation, and finished with a Streamlit application that accepts a PDF and answers questions using its contents.

## Week 1 — LLM Fundamentals and API Basics

I learned how to send messages to a Groq-hosted language model and preserve conversation context by keeping an ordered list of `system`, `user`, and `assistant` messages. Sending that history with each request allowed the chatbot to refer to earlier turns in the same session.

I compared three system prompts using the same question about Python loops. The Patient Coding Tutor gave a detailed and encouraging explanation, the Concise Study Assistant used a shorter structure, and the Creative Storyteller used an analogy and more expressive language. The comparison helped me see how instructions in a system message can change the tone, structure, and level of detail of a response.

In the temperature experiment, the same poem request produced different wording at temperatures `1.2` and `0.0`. Both responses were coherent, but this was one observed comparison rather than proof that a higher temperature always produces a better or more creative answer. I also learned that conversation memory depends on maintaining the messages in the correct order and with the correct roles.

## Week 2 — Embeddings and Semantic Search

I learned that an embedding represents text as a numeric vector. My semantic search project used `sentence-transformers/all-MiniLM-L6-v2` to create 384-dimensional embeddings for 40 student tech-support FAQ records. I cached the dataset vectors and embedded only the incoming query during a search.

The clearest example of semantic search outperforming the keyword baseline was the query, “Booting is painfully sluggish, and the desktop takes forever to appear.” Semantic search returned FAQ013, “Why does my computer start slowly?”, as its top result with a cosine similarity of `0.5732`. The keyword search returned no match because the query did not share any non-stopword terms with the FAQ text.

The comparison also showed that semantic search is not correct for every query. Some results were only partially relevant or unrelated, so I treated cosine similarity as a ranking signal rather than a guarantee that the returned record was correct. Caching the embeddings avoided repeating work for a dataset that did not change between queries.

## Week 3 — Retrieval-Augmented Generation

I combined semantic retrieval with Groq generation using a 4,091-word fictional student technology-support handbook. The document was split into overlapping chunks, embedded with the same Week 2 model, and stored in a persistent ChromaDB collection. For each question, the pipeline retrieved relevant passages and included them in the prompt sent to Groq.

I tested three chunk configurations: 100 words with 20 words of overlap, 250 words with 50 words of overlap, and 400 words with 80 words of overlap. They produced 51, 21, and 13 chunks respectively. All three captured runs generated answers for the five in-scope questions and returned the required abstention for the three out-of-scope questions. The 250/50 configuration gave the most consistently complete answers in the observed runs, although the experiment did not prove that chunk size alone caused every difference.

The project helped me understand grounding as answering from retrieved source evidence instead of relying only on the model's general knowledge. The similarity threshold filtered weak retrieval results, and the prompt required `I don't know based on the provided context.` when the evidence was insufficient. These measures reduce the chance of unsupported answers, but they do not guarantee that retrieval or generation will always be correct.

ChromaDB also made the retrieval implementation more structured than the Week 2 NumPy matrix. It stored the chunk text, IDs, embeddings, and metadata in a collection and handled nearest-neighbor retrieval. I still needed to inspect the returned passages because a nearest result is not automatically evidence for the answer.

## Week 4 — Chat With Your PDF

The final project brought the earlier components into a Streamlit application. A user uploads a PDF, `pypdf` extracts the text page by page, and the Week 3 chunking configuration creates 250-word chunks with a 50-word overlap. The chunks are embedded with `sentence-transformers/all-MiniLM-L6-v2` on CPU and added to a ChromaDB collection scoped to the current session and uploaded PDF.

When a question is submitted, the application embeds the question, retrieves up to three relevant chunks, and applies the minimum cosine-similarity threshold of `0.25`. The retrieved excerpts and recent conversation turns are sent to Groq using the `openai/gpt-oss-20b` model by default. The system prompt restricts factual answers to the retrieved PDF excerpts and asks the model to cite the source page numbers. When the available evidence is insufficient, the required response is `I don't know based on the provided context.`

The Streamlit interface uses `st.chat_message` and `st.chat_input` for the conversation and `st.session_state` to retain messages across app reruns. The retrieved passages can be inspected in the **PDF sources** expander, which displays the excerpt, PDF page label, and cosine-similarity score. The sidebar shows the model and retrieval configuration, and the application includes a processing indicator and a **New Chat / Upload New PDF** action.

### Local Test Results

I used `PromptForge_Week4_FinalProject.pdf` as the first test document. The app reported three text-containing pages and created five chunks.

I asked the following questions in the same chat:

| Test | Question | Observed result |
|---|---|---|
| 1 — Project objective | What is the main objective of the PromptForge Week 4 final project? | The answer described the goal of building and deploying the “Chat With Your PDF” RAG application and cited Page 1. |
| 2 — Processing workflow | Explain the complete PDF processing workflow, from upload to ChromaDB storage. | The answer covered upload, text extraction, chunking, embedding, and ChromaDB indexing, with references to the relevant PDF excerpts. |
| 3 — Chat interface | What features are required in the chat interface, and how should conversation history be maintained? | The answer identified `st.chat_message`, `st.chat_input`, `st.session_state`, retrieval, source transparency, loading indicators, and reset behavior. |
| 4 — Edge cases and deployment | What edge cases must the app handle, and what are the deployment requirements? | The answer identified the missing-PDF, empty-question, and no-extractable-text cases, along with Streamlit Community Cloud and secret-management requirements. |
| 5 — Unsupported information | What is the exact monthly stipend paid to PromptForge interns? | The assistant returned `I don't know based on the provided context.` |

The fifth question was useful because the uploaded brief did not specify an internship stipend. The assistant did not invent an amount and returned the configured abstention response. The other questions were answerable from the uploaded brief and the responses included source-page references.

The PDF utility tests were also run locally with `python -m unittest discover -s tests -v`. All four tests passed:

- `test_chunk_keeps_page_numbers_when_spanning_pages`
- `test_default_chunking_uses_250_words_and_50_overlap`
- `test_empty_pages_return_no_chunks`
- `test_invalid_overlap_is_rejected`

### Implementation Decisions and Observations

The Week 3 chunking settings and embedding model were retained so the final project remained a continuation of the earlier work. ChromaDB is used through an in-memory client, with a separate collection name for each app session and uploaded PDF. The embedding model is cached within the Streamlit process so the same CPU model does not need to be loaded separately for every session.

During local setup, the chat input remained disabled while the running app could not read `GROQ_API_KEY`. This made the dependency between the UI and the API configuration clear: the PDF can be extracted and indexed without a Groq key, but question answering is disabled until the key is available to Streamlit. After the API configuration was corrected, the chat requests completed.

The local evaluation confirms the PDF upload, indexing, question-answering, and abstention flow for the test brief. It does not yet establish behavior on a wide range of PDFs. The scanned/image-only PDF branch is implemented to display a no-extractable-text message, but a separate scanned-PDF test has not been recorded here. The reset action and larger-document behavior also need to be checked as part of final deployment testing. Streamlit Community Cloud deployment and the public demo URL are not included in these local test results.

## Overall Learning

The progression across the four weeks helped me understand how the parts of a grounded GenAI application fit together. Week 1 focused on the LLM API, message history, and prompting. Week 2 explained how embeddings represent text and how semantic similarity can retrieve related content. Week 3 connected retrieval with generation and introduced grounding and abstention. Week 4 combined those pieces in a user-facing application with PDF upload, page-aware sources, chat history, and a deployment path.

The main lesson I take from the project is that a fluent answer is not necessarily a supported answer. The quality of the retrieved chunks matters, as do chunk boundaries and the instructions given to the model. Showing the excerpts and page numbers makes the evidence easier to inspect, while the abstention response gives the application a way to handle questions that the uploaded document cannot answer.