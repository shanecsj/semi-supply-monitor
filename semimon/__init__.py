"""RAM & GPU supply-chain news monitor."""

from .dotenv import load_dotenv as _load_dotenv

# Load .env before any submodule is imported. classify.py and chat.py read
# os.environ at module level for their model defaults, so a loader called later
# would run after those values had already been frozen.
_load_dotenv()
