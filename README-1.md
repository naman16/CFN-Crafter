+++CFN-Crafter Application

**CFN-Crafter** is an LLM-powered application that acts like a cloud security engineer, assisting with:

1. Defining AWS service-level security requirements
2. Codifying these requirements into AWS CloudFormation templates

## Environment Setup

Follow these steps to set up the environment:

1. Clone the repository and navigate to the directory:
    ```
    git clone https://github.com/naman16/CFN-Crafter.git
    cd CFN-Crafter
    ```
2. Generate and set the `OPENAI_API_KEY` as an environment variable for the application to access OpenAI models. You can generate an API key [here](https://platform.openai.com/account/api-keys).
3. Create a `.env` file in the repository folder and add your `OPENAI_API_KEY`:
    ```
    OPENAI_API_KEY="sk-…"
    ```
4. Install the necessary packages:
    ```
    pip install -r requirements.txt
    ```
    *Note 1: There may be issues with package installation using the requirements file, which might require manual installation.*

    *Note 2: Package versions may need to be adjusted based on system/OS compatibility.*
5. Unzip the `faiss_index.zip` file to enable RAG (Retrieval-Augmented Generation) based on user queries:
    ```
    unzip faiss_index.zip
    ```
6. Start the application using Streamlit:
    ```
    streamlit run app.py
    ```
+++