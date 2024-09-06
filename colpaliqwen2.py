import os
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from byaldi import RAGMultiModalModel
from PIL import Image
import io

# Load environment variables
load_dotenv()

# Access environment variables
HF_TOKEN = os.getenv("HF_TOKEN")
RAG_MODEL = os.getenv("RAG_MODEL", "vidore/colpali")
QWN_MODEL = os.getenv("QWN_MODEL", "Qwen/Qwen2-VL-7B-Instruct")
QWN_PROCESSOR = os.getenv("QWN_PROCESSOR", "Qwen/Qwen2-VL-2B-Instruct")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env file")

# Initialize FastAPI app
app = FastAPI()

# Load models and processors
RAG = RAGMultiModalModel.from_pretrained(RAG_MODEL, use_auth_token=HF_TOKEN)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    QWN_MODEL,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
    trust_remote_code=True,
    use_auth_token=HF_TOKEN
).cuda().eval()

processor = AutoProcessor.from_pretrained(QWN_PROCESSOR, trust_remote_code=True, use_auth_token=HF_TOKEN)

# Define request model
class DocumentRequest(BaseModel):
    text_query: str

# Define processing function
def document_rag(text_query, image):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": text_query},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    generated_ids = model.generate(**inputs, max_new_tokens=50)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

# Define API endpoints
@app.post("/process_document")
async def process_document(request: DocumentRequest, file: UploadFile = File(...), x_api_key: Optional[str] = Header(None)):
    # Check API key
    if x_api_key != HF_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid API key")
    
    # Read and process the uploaded file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Process the document
    result = document_rag(request.text_query, image)
    
    return {"result": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
