from os import getenv

from dotenv import load_dotenv


load_dotenv()


# API KEYS
OPENAI_API_KEY: str = getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY: str = getenv("OPENROUTER_API_KEY")

# PARAMETERS
CLASSIFICATION_BATCH: int = int(getenv("CLASSIFICATION_BATCH") or 4)

# NEO4J
NEO4J_URI: str = getenv("NEO4J_URI")
NEO4J_USERNAME: str = getenv("NEO4J_USERNAME")
NEO4J_PASSWORD: str = getenv("NEO4J_PASSWORD")
