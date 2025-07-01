#!/usr/bin/env python3
"""
医疗保险文档处理FastAPI应用
端到端处理：上传文档 -> 分类 -> 信息抽取 -> 字段验证 -> 返回结构化JSON
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union
import time
import uuid
from datetime import datetime

import torch
from PIL import Image
import pandas as pd
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 导入自定义模块
sys.path.append(str(Path(__file__).parent.parent))

from phase2_classification.train_dit import DiTClassifier
from phase3_extraction.train_layoutlm import MedicalFieldExtractor
from phase4_demo.validation_utils import MedicalFieldValidator

# Transformers imports
from transformers import (
    AutoImageProcessor, LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification, AutoConfig
)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FastAPI应用
app = FastAPI(
    title="医疗保险文档智能处理系统",
    description="自动分类医保文档并抽取关键信息",
    version="1.0.0"
)

# CORS设置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 数据模型
class ProcessingResult(BaseModel):
    """处理结果模型"""
    request_id: str
    status: str
    processing_time: float
    document_type: Optional[str] = None
    confidence: Optional[float] = None
    extracted_fields: Optional[Dict] = None
    validation_results: Optional[Dict] = None
    recommendations: Optional[List[str]] = None
    error_message: Optional[str] = None

class HealthStatus(BaseModel):
    """健康检查模型"""
    status: str
    timestamp: str
    models_loaded: Dict[str, bool]
    version: str

# 全局变量
classification_model = None
classification_processor = None
extraction_model = None
extraction_processor = None
field_extractor = None
validator = None

# 配置
CONFIG = {
    'max_file_size': 10 * 1024 * 1024,  # 10MB
    'allowed_extensions': ['.pdf', '.jpg', '.jpeg', '.png', '.tiff'],
    'upload_dir': Path(__file__).parent / 'uploads',
    'output_dir': Path(__file__).parent / 'outputs'
}

def load_models():
    """加载预训练模型"""
    global classification_model, classification_processor
    global extraction_model, extraction_processor
    global field_extractor, validator
    
    logger.info("开始加载模型...")
    
    try:
        # 验证器
        validator = MedicalFieldValidator()
        logger.info("✓ 验证器加载完成")
        
        # 获取基础目录
        base_dir = Path(__file__).parent.parent
        
        # 查找最佳分类模型
        classification_dir = base_dir / "models" / "classification"
        if classification_dir.exists():
            model_dirs = [d for d in classification_dir.iterdir() if d.is_dir() and 'dit_classifier' in d.name]
            if model_dirs:
                # 选择准确率最高的模型
                best_model_dir = max(model_dirs, key=lambda x: float(x.name.split('_acc_')[-1]) if '_acc_' in x.name else 0)
                
                logger.info(f"加载分类模型: {best_model_dir}")
                
                # 加载分类模型配置
                with open(best_model_dir / "training_config.json", 'r') as f:
                    config = json.load(f)
                
                # 加载处理器
                classification_processor = AutoImageProcessor.from_pretrained(best_model_dir / "dit")
                
                # 加载模型
                classification_model = DiTClassifier(
                    model_name_or_path=str(best_model_dir / "dit"),
                    num_classes=config['num_classes']
                )
                
                # 加载分类头权重
                classifier_path = best_model_dir / "classifier.pth"
                if classifier_path.exists():
                    classification_model.classifier.load_state_dict(torch.load(classifier_path, map_location='cpu'))
                
                classification_model.eval()
                logger.info("✓ 分类模型加载完成")
            else:
                logger.warning("未找到训练好的分类模型，使用预训练模型")
                classification_processor = AutoImageProcessor.from_pretrained("microsoft/dit-base-finetuned-rvlcdip")
        
        # 查找最佳抽取模型
        extraction_dir = base_dir / "models" / "extraction"  
        if extraction_dir.exists():
            model_dirs = [d for d in extraction_dir.iterdir() if d.is_dir() and 'layoutlmv3' in d.name]
            if model_dirs:
                # 选择F1分数最高的模型
                best_model_dir = max(model_dirs, key=lambda x: float(x.name.split('_f1_')[-1]) if '_f1_' in x.name else 0)
                
                logger.info(f"加载抽取模型: {best_model_dir}")
                
                # 加载抽取模型配置
                with open(best_model_dir / "training_config.json", 'r') as f:
                    config = json.load(f)
                
                # 加载处理器和模型
                extraction_processor = LayoutLMv3Processor.from_pretrained(best_model_dir)
                extraction_model = LayoutLMv3ForTokenClassification.from_pretrained(best_model_dir)
                extraction_model.eval()
                
                # 创建字段抽取器
                field_extractor = MedicalFieldExtractor(
                    config['label_info']['label_map'],
                    config['label_info']['medical_fields']
                )
                
                logger.info("✓ 抽取模型加载完成")
            else:
                logger.warning("未找到训练好的抽取模型")
        
        logger.info("🎉 所有模型加载完成")
        
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        raise

def classify_document(image: Image.Image) -> Dict[str, any]:
    """分类文档"""
    if classification_model is None or classification_processor is None:
        return {"document_type": "unknown", "confidence": 0.0, "error": "分类模型未加载"}
    
    try:
        # 预处理图片
        inputs = classification_processor(image, return_tensors="pt")
        
        # 推理
        with torch.no_grad():
            outputs = classification_model(pixel_values=inputs['pixel_values'])
            probabilities = torch.softmax(outputs['logits'], dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # 类别映射
        class_names = ['invoice', 'letter', 'email', 'memo', 'form']
        document_type = class_names[predicted_class] if predicted_class < len(class_names) else 'unknown'
        
        return {
            "document_type": document_type,
            "confidence": confidence,
            "all_scores": {class_names[i]: probabilities[0][i].item() for i in range(len(class_names))}
        }
        
    except Exception as e:
        logger.error(f"文档分类失败: {e}")
        return {"document_type": "unknown", "confidence": 0.0, "error": str(e)}

def extract_information(image: Image.Image) -> Dict[str, any]:
    """抽取信息"""
    if extraction_model is None or extraction_processor is None or field_extractor is None:
        return {"extracted_fields": {}, "error": "抽取模型未加载"}
    
    try:
        # 简单OCR（这里可以集成更好的OCR）
        # 为演示目的，创建模拟的OCR结果
        words = ["Provider:", "Dr.", "John", "Smith", "Phone:", "555-123-4567", "Patient:", "Jane", "Doe", "Amount:", "$250.00"]
        boxes = [[50, 50, 120, 80], [130, 50, 160, 80], [170, 50, 220, 80], [230, 50, 290, 80],
                [50, 100, 110, 130], [120, 100, 250, 130], [50, 150, 120, 180], [130, 150, 180, 180], 
                [190, 150, 240, 180], [50, 200, 110, 230], [120, 200, 200, 230]]
        
        # 预处理
        encoding = extraction_processor(
            image,
            words,
            boxes=boxes,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        # 预测
        with torch.no_grad():
            outputs = extraction_model(**encoding)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            # 获取有效预测
            active_mask = encoding['attention_mask'][0].cpu().numpy() == 1
            active_predictions = predictions[0][active_mask].cpu().numpy()
            
            # 抽取实体
            entities = field_extractor.extract_entities(
                words, boxes, active_predictions[:len(words)]
            )
            
            # 映射到医疗字段
            medical_fields = field_extractor.map_to_medical_fields(entities)
            
            return {
                "extracted_fields": medical_fields,
                "entities": entities,
                "raw_predictions": active_predictions[:len(words)].tolist()
            }
            
    except Exception as e:
        logger.error(f"信息抽取失败: {e}")
        return {"extracted_fields": {}, "error": str(e)}

@app.on_event("startup")
async def startup_event():
    """启动时加载模型"""
    # 创建必要的目录
    CONFIG['upload_dir'].mkdir(exist_ok=True)
    CONFIG['output_dir'].mkdir(exist_ok=True)
    
    # 加载模型
    try:
        load_models()
    except Exception as e:
        logger.error(f"启动时模型加载失败: {e}")

@app.get("/", response_class=HTMLResponse)
async def root():
    """主页"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>医疗保险文档处理系统</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
            .header { text-align: center; color: #2c3e50; }
            .upload-area { border: 2px dashed #3498db; padding: 30px; text-align: center; margin: 20px 0; }
            .upload-area:hover { background-color: #ecf0f1; }
            button { background-color: #3498db; color: white; border: none; padding: 10px 20px; cursor: pointer; border-radius: 5px; }
            button:hover { background-color: #2980b9; }
            .result { margin-top: 20px; padding: 15px; background-color: #f8f9fa; border-radius: 5px; }
            .error { color: #e74c3c; }
            .success { color: #27ae60; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🏥 医疗保险文档智能处理系统</h1>
            <p>上传医保文档，自动识别类型并抽取关键信息</p>
        </div>
        
        <div class="upload-area">
            <p>📄 点击选择文件或拖拽文件到此处</p>
            <input type="file" id="fileInput" accept=".pdf,.jpg,.jpeg,.png,.tiff" style="display: none;">
            <button onclick="document.getElementById('fileInput').click()">选择文件</button>
        </div>
        
        <div id="result" class="result" style="display: none;"></div>
        
        <div style="margin-top: 30px;">
            <h3>🔧 API接口</h3>
            <ul>
                <li><strong>POST /process</strong> - 处理文档</li>
                <li><strong>GET /health</strong> - 健康检查</li>
                <li><strong>GET /docs</strong> - 接口文档</li>
            </ul>
        </div>
        
        <script>
            document.getElementById('fileInput').onchange = function(e) {
                const file = e.target.files[0];
                if (!file) return;
                
                const formData = new FormData();
                formData.append('file', file);
                
                const resultDiv = document.getElementById('result');
                resultDiv.style.display = 'block';
                resultDiv.innerHTML = '<p>⏳ 处理中，请稍候...</p>';
                
                fetch('/process', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        resultDiv.innerHTML = `
                            <h4 class="success">✅ 处理成功</h4>
                            <p><strong>文档类型:</strong> ${data.document_type} (置信度: ${(data.confidence * 100).toFixed(1)}%)</p>
                            <p><strong>处理时间:</strong> ${data.processing_time.toFixed(2)}秒</p>
                            <h5>抽取的信息:</h5>
                            <pre>${JSON.stringify(data.extracted_fields || {}, null, 2)}</pre>
                            <h5>验证结果:</h5>
                            <pre>${JSON.stringify(data.validation_results || {}, null, 2)}</pre>
                        `;
                    } else {
                        resultDiv.innerHTML = `
                            <h4 class="error">❌ 处理失败</h4>
                            <p>${data.error_message}</p>
                        `;
                    }
                })
                .catch(error => {
                    resultDiv.innerHTML = `
                        <h4 class="error">❌ 网络错误</h4>
                        <p>${error.message}</p>
                    `;
                });
            };
        </script>
    </body>
    </html>
    """
    return html_content

@app.get("/health", response_model=HealthStatus)
async def health_check():
    """健康检查"""
    models_status = {
        "classification_model": classification_model is not None,
        "extraction_model": extraction_model is not None,
        "validator": validator is not None
    }
    
    overall_status = "healthy" if all(models_status.values()) else "unhealthy"
    
    return HealthStatus(
        status=overall_status,
        timestamp=datetime.now().isoformat(),
        models_loaded=models_status,
        version="1.0.0"
    )

@app.post("/process", response_model=ProcessingResult)
async def process_document(file: UploadFile = File(...)):
    """处理文档"""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        # 验证文件
        if file.size > CONFIG['max_file_size']:
            raise HTTPException(status_code=413, detail="文件太大")
        
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in CONFIG['allowed_extensions']:
            raise HTTPException(status_code=400, detail="不支持的文件格式")
        
        # 读取并处理图片
        contents = await file.read()
        
        try:
            image = Image.open(io.BytesIO(contents)).convert('RGB')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"无法读取图片: {e}")
        
        # 步骤1: 分类文档
        classification_result = classify_document(image)
        
        # 步骤2: 抽取信息（如果是表单类型）
        extracted_fields = {}
        if classification_result.get('document_type') in ['form', 'invoice']:
            extraction_result = extract_information(image)
            extracted_fields = extraction_result.get('extracted_fields', {})
        
        # 步骤3: 验证字段
        validation_results = {}
        validation_summary = {}
        if extracted_fields and validator:
            validation_results = validator.validate_all_fields(extracted_fields)
            validation_summary = validator.generate_validation_summary(validation_results)
        
        # 计算处理时间
        processing_time = time.time() - start_time
        
        # 构建结果
        result = ProcessingResult(
            request_id=request_id,
            status="success",
            processing_time=processing_time,
            document_type=classification_result.get('document_type'),
            confidence=classification_result.get('confidence'),
            extracted_fields=extracted_fields,
            validation_results={
                'field_results': validation_results,
                'summary': validation_summary
            } if validation_results else None,
            recommendations=validation_summary.get('recommendations', []) if validation_summary else None
        )
        
        # 保存处理结果（可选）
        output_file = CONFIG['output_dir'] / f"{request_id}_result.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result.dict(), f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"文档处理完成 - ID: {request_id}, 类型: {result.document_type}, 时间: {processing_time:.2f}s")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文档处理失败 - ID: {request_id}, 错误: {e}")
        
        return ProcessingResult(
            request_id=request_id,
            status="error",
            processing_time=time.time() - start_time,
            error_message=str(e)
        )

@app.get("/statistics")
async def get_statistics():
    """获取处理统计信息"""
    try:
        output_files = list(CONFIG['output_dir'].glob("*_result.json"))
        
        total_processed = len(output_files)
        document_types = {}
        success_count = 0
        total_processing_time = 0
        
        for file_path in output_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                
                if result.get('status') == 'success':
                    success_count += 1
                    doc_type = result.get('document_type', 'unknown')
                    document_types[doc_type] = document_types.get(doc_type, 0) + 1
                    total_processing_time += result.get('processing_time', 0)
                    
            except Exception as e:
                logger.warning(f"读取结果文件失败 {file_path}: {e}")
        
        avg_processing_time = total_processing_time / success_count if success_count > 0 else 0
        
        return {
            "total_processed": total_processed,
            "success_count": success_count,
            "success_rate": success_count / total_processed if total_processed > 0 else 0,
            "document_types": document_types,
            "average_processing_time": avg_processing_time,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"统计信息获取失败: {e}")

@app.post("/batch_process")
async def batch_process(files: List[UploadFile] = File(...)):
    """批量处理文档"""
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="批量处理最多支持10个文件")
    
    results = []
    
    for file in files:
        try:
            result = await process_document(file)
            results.append(result)
        except Exception as e:
            results.append(ProcessingResult(
                request_id=str(uuid.uuid4()),
                status="error",
                processing_time=0,
                error_message=f"文件 {file.filename} 处理失败: {e}"
            ))
    
    return {"batch_results": results, "total_files": len(files)}

if __name__ == "__main__":
    import uvicorn
    import io
    
    logger.info("启动医疗保险文档处理服务...")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=False,
        log_level="info"
    ) 