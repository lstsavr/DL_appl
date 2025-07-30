import os
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import uvicorn
from typing import Optional, List

from infer_unified import load_components, text_to_image, image_to_text

app = FastAPI(title="Flickr8K Retrieval API", version="1.0.0")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
DEVICE = os.environ.get("DEVICE", "cpu")
DUAL_MODEL_PATH = "checkpoints/best.pth"
CROSS_MODEL_PATH = "checkpoints_ca/best_ca.pth"
VOCAB_PATH = "data/vocab.json"
IMAGE_DIR = "data/raw/Flicker8k_Dataset"
CAPTION_FILE = "data/raw/Flickr8k_text/Flickr8k.token.txt"

# 缓存模型
dual_model = None
dual_vocab = None
cross_model = None
cross_vocab = None


@app.on_event("startup")
async def startup_event():
    """启动时加载模型"""
    global dual_model, dual_vocab, cross_model, cross_vocab
    
    # 加载双流模型
    if os.path.exists(DUAL_MODEL_PATH):
        try:
            dual_model, dual_vocab, _ = load_components("dual", DUAL_MODEL_PATH, VOCAB_PATH, DEVICE)
            print("双流模型加载成功")
        except Exception as e:
            print(f"加载双流模型失败: {e}")
    else:
        print(f"双流模型文件不存在: {DUAL_MODEL_PATH}")
    
    # 加载交叉注意力模型
    if os.path.exists(CROSS_MODEL_PATH):
        try:
            cross_model, cross_vocab, _ = load_components("cross", CROSS_MODEL_PATH, VOCAB_PATH, DEVICE)
            print("交叉注意力模型加载成功")
        except Exception as e:
            print(f"加载交叉注意力模型失败: {e}")
    else:
        print(f"交叉注意力模型文件不存在: {CROSS_MODEL_PATH}")


@app.get("/")
async def root():
    """API根路径"""
    return {"message": "Flickr8K Retrieval API", "status": "running"}


@app.get("/models")
async def get_models():
    """获取可用模型列表"""
    models = []
    if dual_model is not None:
        models.append("dual")
    if cross_model is not None:
        models.append("cross")
    
    return {"available_models": models}


@app.post("/text-to-image")
async def text_to_image_endpoint(
    text: str = Form(...),
    model_type: str = Form("dual"),
    k: int = Form(5)
):
    """文本到图像检索API"""
    # 检查模型是否可用
    if model_type == "dual" and dual_model is None:
        raise HTTPException(status_code=404, detail="双流模型未加载")
    elif model_type == "cross" and cross_model is None:
        raise HTTPException(status_code=404, detail="交叉注意力模型未加载")
    
    # 选择模型和词汇表
    model = dual_model if model_type == "dual" else cross_model
    vocab = dual_vocab if model_type == "dual" else cross_vocab
    
    try:
        # 执行检索
        results = text_to_image(text, model, vocab, DEVICE, IMAGE_DIR, k, False)
        
        # 处理结果
        processed_results = []
        for result_list in results:
            items = []
            for idx, score in result_list:
                img_path = os.path.join(IMAGE_DIR, os.path.basename(IMAGE_DIR[idx]))
                items.append({
                    "image_path": img_path,
                    "image_name": os.path.basename(IMAGE_DIR[idx]),
                    "score": score
                })
            processed_results.append(items)
        
        return {"query": text, "results": processed_results[0]}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检索失败: {str(e)}")


@app.post("/image-to-text")
async def image_to_text_endpoint(
    image: UploadFile = File(...),
    model_type: str = Form("dual"),
    k: int = Form(5)
):
    """图像到文本检索API"""
    # 检查模型是否可用
    if model_type == "dual" and dual_model is None:
        raise HTTPException(status_code=404, detail="双流模型未加载")
    elif model_type == "cross" and cross_model is None:
        raise HTTPException(status_code=404, detail="交叉注意力模型未加载")
    
    # 选择模型和词汇表
    model = dual_model if model_type == "dual" else cross_model
    vocab = dual_vocab if model_type == "dual" else cross_vocab
    
    try:
        # 保存上传的图像到临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(await image.read())
            temp_path = temp_file.name
        
        # 执行检索
        results = image_to_text(temp_path, model, vocab, DEVICE, CAPTION_FILE, k, False)
        
        # 删除临时文件
        os.unlink(temp_path)
        
        # 处理结果
        processed_results = []
        for result_list in results:
            items = []
            for idx, score in result_list:
                items.append({
                    "caption": CAPTION_FILE[idx],
                    "score": score
                })
            processed_results.append(items)
        
        return {"query_image": image.filename, "results": processed_results[0]}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检索失败: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True) 