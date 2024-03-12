# RAG_a_Gita
This is a GenAI RAG (Retrieval Augmented Generation) based contextualized chatbot app created using:
```
>> Gradio for the app
>> OpenAI embedding model for local document embeddings
>> Pinecone vector database for storing the local document embeddings
>> OpenAI GPT-3.5-Turbo model for getting the contextualized answers
```
## Usage

### Step - 1:
```
pip install requirements.txt
```

### Step - 2:

Get necessary API Keys.

```
PINECONE_API_KEY = "YOUR-PINECONE-API-KEY"
PINECONE_ENVIRONMENT = "YOUR-PINECONE-ENVIRONMENT"

OPENAI_API_KEY = "YOUR-OPENAI-API-KEY"
OPENAI_ORGANIZATION = "YOUR-OPENAI-ORGANIZATION"

HOLY_BOOK = "gita" ('bible'/ 'quran')
```

### Step - 3:

Create index for the selected holybook (gita/ bible/ quran).

```
python createindex.py --holybook $HOLY_BOOK --pinecone_apikey $PINECONE_API_KEY --pinecone_environment $PINECONE_ENVIRONMENT --openaikey $OPENAI_API_KEY --openaiorg $OPENAI_ORGANIZATION
```

### Step - 4:

Launch Gradio app.

```
python app.py --holybook $HOLY_BOOK --pinecone_apikey $PINECONE_API_KEY --pinecone_environment $PINECONE_ENVIRONMENT --openaikey $OPENAI_API_KEY
```
