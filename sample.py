# from langchain_nvidia_ai_endpoints import ChatNVIDIA

# client = ChatNVIDIA(
#   model="meta/llama-3.3-70b-instruct",
#   api_key="NVIDIA_API_KEY",
#   temperature=0.5,
#   top_p=0.7,
#   max_tokens=50,
# )

# for chunk in client.stream([{"role":"user","content":"Write a limerick about the wonders of GPU computing."}]): 
#   print(chunk.content, end="")

from dotenv import load_dotenv
load_dotenv()
import os
nvidia_api_key = os.getenv("NVIDIA_API_KEY")

from langchain_nvidia_ai_endpoints import NVIDIARerank
from langchain_core.documents import Document

query = "What is the GPU memory bandwidth of H100 SXM?"
passages = [
    "The Hopper GPU is paired with the Grace CPU using NVIDIA's ultra-fast chip-to-chip interconnect, delivering 900GB/s of bandwidth, 7X faster than PCIe Gen5. This innovative design will deliver up to 30X higher aggregate system memory bandwidth to the GPU compared to today's fastest servers and up to 10X higher performance for applications running terabytes of data.", 
    "A100 provides up to 20X higher performance over the prior generation and can be partitioned into seven GPU instances to dynamically adjust to shifting demands. The A100 80GB debuts the world's fastest memory bandwidth at over 2 terabytes per second (TB/s) to run the largest models and datasets.", 
    "Accelerated servers with H100 deliver the compute power—along with 3 terabytes per second (TB/s) of memory bandwidth per GPU and scalability with NVLink and NVSwitch™.", 
]

client = NVIDIARerank(
  model="nvidia/llama-3.2-nv-rerankqa-1b-v2", 
  api_key="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",
)

response = client.compress_documents(
  query=query,
  documents=[Document(page_content=passage) for passage in passages]
)

print(response)
