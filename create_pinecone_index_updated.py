import logging
import argparse
import json
from pinecone import Pinecone, PodSpec  # pip install pinecone-client for pinecone
import openai
import itertools
from openai import OpenAI
client = OpenAI()


def create_logger():
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Create a formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    # Add the formatter to the console handler
    console_handler.setFormatter(formatter)

    # Add the console handler to the logger
    logger.addHandler(console_handler)
    return logger


logger = create_logger()


class CreatePineconeIndex:
    def __init__(self):
        try:
            logger.info("Parsing arguments")
            parser = argparse.ArgumentParser()
            parser.add_argument("--holybook", type=str, required=True)
            parser.add_argument("--pinecone_apikey", type=str, required=True)
            parser.add_argument("--pinecone_environment", type=str, required=False)
            parser.add_argument("--openaikey", type=str, required=True)
            parser.add_argument("--openaiorg", type=str, required=True)

            args = parser.parse_args()
            self.holybook = args.holybook
            self.pinecone_apikey = args.pinecone_apikey
            self.pinecone_environment = args.pinecone_environment
            self.openaikey = args.openaikey
            self.openaiorg = args.openaiorg
            self.pc = Pinecone(api_key=self.pinecone_apikey)
        except Exception as e:
            logger.error("Error while parsing arguments: {}".format(e))

    def read_json(self):
        try:
            with open(f"{self.holybook}.json", "r") as f:
                data = json.loads(f.read())
            return data
        except Exception as e:
            logger.error("Error while reading json: {}".format(e))

    # Initiate pinecone
    def create_pinecone_index(self, name, embeds):
        """

        :type embeds: an iterable
        """
        try:
            self.pc.create_index(
                name=self.holybook,
                dimension=len(embeds[0]),
                metric="cosine",
                spec=PodSpec(
                    environment=self.pinecone_environment,
                    pod_type='s1.x1'
                )
            )
        except Exception as e:
            logger.error("Error while creating pinecone index: {}".format(e))


    # Now create the document embeddings using OpenAI embedding model
    def create_embeddings(self, data):
        try:
            openai.organization = self.openaiorg  # get this from top-right dropdown on OpenAI under organization > settings
            openai.api_key = self.openaikey  # get API key from top-right dropdown on OpenAI website

            def gen_embeddings(data):
                response = client.embeddings.create(
                    input=data,
                    model="text-embedding-3-small"
                )
                embeddings = [response.data[i].embedding for i in range(len(data))]
                return embeddings

            def chunks(iterable, batch_size=100):
                it = iter(iterable)
                chunk = tuple(itertools.islice(it, batch_size))
                while chunk:
                    yield chunk
                    chunk = tuple(itertools.islice(it, batch_size))

            embeddings = []
            for chunk in chunks(data, batch_size=1000):
                embeddings.extend(gen_embeddings(chunk))

            import csv
            with open('embeddings.csv', 'w', newline='') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                wr.writerow(embeddings)
            return embeddings
        except Exception as e:
            logger.error("Error while creating embeddings: {}".format(e))

    # Insert embeddings into pinecone
    def insert_embeddings_pinecone(self, embeddings, data):
        try:
            #self.init_pinecone()
            # Get all the saved indexes on pinecone
            indexes = self.pc.list_indexes()

            if self.holybook in indexes:
                self.pc.delete_index(self.holybook)

            # Create pinecone index
            # pinecone.create_index(self.holybook, dimension=len(embeddings[0]))
            self.create_pinecone_index(self.holybook, embeddings)

            # Connect to index
            index = self.pc.Index(self.holybook)

            keys = list(data.keys())
            to_upsert = [(keys[i], embeddings[i]) for i in range(len(embeddings))]

            def chunks(iterable, batch_size=100):
                """A helper function to break an iterable into chunks of size batch_size."""
                it = iter(iterable)
                chunk = tuple(itertools.islice(it, batch_size))
                while chunk:
                    yield chunk
                    chunk = tuple(itertools.islice(it, batch_size))

            # Upsert data with 100 vectors per upsert request
            for ids_vectors_chunk in chunks(to_upsert, batch_size=100):
                index.upsert(vectors=ids_vectors_chunk)

            logger.info(f"Index on pinecone for {self.holybook} created successfully")
        except Exception as e:
            logger.error("Error while inserting embeddings: {}".format(e))

    # Now create pinecone index and insert embeddings into the index
    def create_index(self):
        try:
            logger.info(f"Reading holybook {self.holybook} data")
            # Read data
            data = self.read_json()
            print(data)

            logger.info(f"Creating index for holybook {self.holybook}")
            # Create embeddings
            embeddings = self.create_embeddings(data)
            print(embeddings)

            logger.info(
                f"Inserting embeddings into Pinecone for holybook {self.holybook}"
            )
            # Insert embeddings into pinecone
            self.insert_embeddings_pinecone(embeddings, data)
        except Exception as e:
            logger.error("Error while creating index: {}".format(e))


if __name__ == "__main__":
    pineconeindex = CreatePineconeIndex()
    pineconeindex.create_index()
