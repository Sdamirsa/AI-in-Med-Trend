### Breaking Project to Steps

- [X] **Phase 1**
    - [X] **Step 1** Getting data from pubmed
    - [X] **Step 2** Cleaning and Structuring data (from XML)
    - [loading] **extra Step 2** Embedding, DimenReduction, Clustering, Sampling
        - [X] Embedding
        - [X] Embedding with GPU
        - [X] DimenReduction
        - [X] Clustering
        - [ ] Sampling
        - [ ] Modifiction: change the logic to store pmid at the time of embedding and in future loops so we can retrive the papers after clustering-based sampling

    - [X] **Step 3:** Extraction (LLM-based) with EXCT module I had

- [ ] **Phase 2**
    - [ ] **Step 2-3:** Turning this into CSS/HTML page with filtering feature -> Enhancing Step 1-4

- [ ] **Phase RAG**
    - [ ] Applying RAG

- [ ] **Phase Scoring**
    - [ ] scoring journals (scimago list and score)
    - [ ] scoring papers (citations, etc)
    - [ ] scoring authors/institutes (amount of score for each author, corresponding, first)
    - [ ] scoring clinical speciality (amount of score for each speciality. Probably based on corresponding author speciality/institute)

- [ ] **Phase Trend**
    - [ ] Trend analysis

- [ ] **Phase Patent**
    - [ ] Linking papers to patents

### Resources to see:
- [Knowledge Graph Construction](https://www.youtube.com/watch?v=OsnM8YTFwk4)
