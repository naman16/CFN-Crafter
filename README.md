# CFN-Crafter Application

**CFN-Crafter** is LLM powered application that behaves like a cloud security engineer and assists you in:
1. Defining AWS service-level security requirements
2. Codifying the requirements into AWS CloudFormation templates

## Environment Setup

1. Clone this repository in your environment and navigate into this directory:
    ```shell
    git clone https://github.com/naman16/CFN-Crafter.git
    cd CFN-Crafter
    ```
2. `OPENAI_API_KEY` will need to be created and set as an environment variable for the application to access the OpenAI models. API key can be generated [here](https://platform.openai.com/account/api-keys).
3. Create a `.env` file within the repository folder and store the `OPENAI_API_KEY` here 
    ```shell
    OPENAI_API_KEY="sk-…"
    ```
4. Install all the necessary packages required for this application:
    ```shell
    pip install -r requirements.txt
    ```
    Note 1: Sometimes, the packages don’t install correctly using the requirements file and may require manual installation.
    
    Note 2: The exact package versions needed for the application may vary with system / OS types.
5. Unzip the `faiss_index.zip` file within the directory to enable the application to perform RAG based on user questions / query:
    ```shell
    unzip faiss_index.zip
    ```
6. Now the application can be started by using streamlit:
    ```shell
    streamlit run app.py
    ```