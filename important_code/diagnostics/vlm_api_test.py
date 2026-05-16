import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from important_code.shared_control.sentinel import CloudVLMClient


client = CloudVLMClient(provider='deepseek', model='deepseek-v4-flash')
print("API key loaded:", bool(client.api_key))