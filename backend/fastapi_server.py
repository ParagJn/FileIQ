from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from generate_document_vectors import VectorDBManager
from AzureAIConnection_IBM import AzureAIConnection
import os
import shutil
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# Allow CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VectorRequest(BaseModel):
    filename: str
    folder: str = "."
    case_id: str = "default"

class QuestionRequest(BaseModel):
    question: str
    folder: str = "."
    case_id: str = "default"

class CaseRequest(BaseModel):
    case_id: str

class ClaimAnalysisRequest(BaseModel):
    question: str
    folder: str = "."
    case_id: str = "default"

# Create a documents folder if it doesn't exist
DOCUMENTS_FOLDER = os.path.join(os.path.dirname(__file__), "documents")
os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)

@app.post("/upload-and-generate-vectors")
async def upload_and_generate_vectors(file: UploadFile = File(...), case_id: str = Form("default")):
    try:
        # Create case-specific folder
        case_folder = os.path.join(DOCUMENTS_FOLDER, case_id)
        os.makedirs(case_folder, exist_ok=True)
        
        # Save uploaded file in case-specific folder
        file_path = os.path.join(case_folder, file.filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Check if vectors already exist for this case
        manager = VectorDBManager(folder_path=case_folder, case_id=case_id)
        if manager.vectors_exist(file.filename):
            return {
                "success": True, 
                "filename": file.filename, 
                "case_id": case_id,
                "already_vectorized": True,
                "message": f"Vectors for '{file.filename}' in case '{case_id}' already exist. Skipping vectorization."
            }
        
        # Generate vectors if they don't exist
        success = manager.build_vector_db(file.filename)
        
        return {
            "success": success, 
            "filename": file.filename, 
            "case_id": case_id,
            "already_vectorized": False,
            "message": f"Vectors for '{file.filename}' in case '{case_id}' generated successfully." if success else f"Failed to generate vectors for '{file.filename}' in case '{case_id}'."
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/generate-vectors")
def generate_vectors(req: VectorRequest):
    try:
        # Create case-specific folder
        case_folder = os.path.join(DOCUMENTS_FOLDER, req.case_id)
        os.makedirs(case_folder, exist_ok=True)
        
        manager = VectorDBManager(folder_path=case_folder, case_id=req.case_id)
        
        # Check if vectors already exist
        if manager.vectors_exist(req.filename):
            return {
                "success": True, 
                "case_id": req.case_id,
                "already_vectorized": True,
                "message": f"Vectors for '{req.filename}' in case '{req.case_id}' already exist. Skipping vectorization."
            }
        
        # Generate vectors if they don't exist
        success = manager.build_vector_db(req.filename)
        return {
            "success": success, 
            "case_id": req.case_id,
            "already_vectorized": False,
            "message": f"Vectors for '{req.filename}' in case '{req.case_id}' generated successfully." if success else f"Failed to generate vectors for '{req.filename}' in case '{req.case_id}'."
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# Initialize Azure AI client
def get_azure_ai_client():
    try:
        return AzureAIConnection()
    except Exception as e:
        raise Exception(f"Error initializing Azure AI: {e}")

def get_enhanced_response(question: str, all_results: dict) -> str:
    """Generate enhanced response using Azure AI with vector context."""
    try:
        azure_client = get_azure_ai_client()
        
        # Combine context from all databases
        all_chunks = []
        source_info = []
        for db_name, results in all_results.items():
            for chunk, score, _ in results[:2]:  # Take top 2 from each DB
                all_chunks.append(f"[From {db_name}] {chunk}")
                source_info.append(f"{db_name} (relevance: {score:.3f})")
        
        context = "\n\n".join(all_chunks[:5])  # Limit to top 5 overall
        
        prompt = f"""Based on the following context from documents, please answer the user's question comprehensively and professionally.
        
        Document Context:
        {context}
        
        User Question: {question}
        
        Please provide a comprehensive answer that:
        1. Directly addresses the question using information from the documents
        2. Adds relevant context or explanations only when explicitly asked by the user
        3. Is well-structured in json format
        4. Build on the provided context without making assumptions
        
        Format your response in a clear, professional manner suitable for a business document analysis system."""
        
        response = azure_client.generate_structured_response(
            prompt=prompt,
            system_content="You are an expert in processing Insurance Claims. " \
            "Analyze the provided document context and extract relevant information to structure a comprehensive claim review summary." \
            " Ensure your response is well-structured and professional. " \
            "If any data is missing or unclear, then give ouput as N/A.",
            max_tokens=1000
        )
        
        return response
        
    except Exception as e:
        return f"Error generating enhanced response: {e}"

@app.post("/ask-question")
async def ask_question(req: QuestionRequest):
    try:
        # Get case-specific folder
        case_folder = os.path.join(DOCUMENTS_FOLDER, req.case_id)
        
        # Initialize VectorDBManager for the specific case
        manager = VectorDBManager(folder_path=case_folder, case_id=req.case_id)
        
        # Search across all vector databases for this case
        all_results = manager.search_all_databases(
            query=req.question,
            k=3,
            score_threshold=0.3
        )
        
        total_results = sum(len(results) for results in all_results.values())
        
        if total_results == 0:
            # No relevant context found, ask Azure AI without document context
            try:
                azure_client = get_azure_ai_client()
                response = azure_client.generate_structured_response(
                    prompt=f"Please analyze this question and provide a structured claim review response: {req.question}",
                    system_content="You are an expert in processing Insurance Claims. Provide a structured response even when analyzing general questions.",
                    max_tokens=500
                )
                
                return {
                    "success": True,
                    "answer": response,
                    "sources": [],
                    "context_found": False,
                    "message": "No relevant information found in uploaded documents. Answer provided using general knowledge."
                }
            except Exception as e:
                return {"success": False, "error": f"Error getting response: {e}"}
        
        # Generate enhanced response with document context
        enhanced_response = get_enhanced_response(req.question, all_results)
        
        # Prepare source information
        sources = []
        for db_name, results in all_results.items():
            if results:
                for chunk, score, _ in results[:2]:
                    sources.append({
                        "document": db_name,
                        "relevance_score": round(float(score), 3),  # Convert numpy.float32 to Python float and round
                        "preview": chunk[:200] + "..." if len(chunk) > 200 else chunk
                    })
        
        return {
            "success": True,
            "answer": enhanced_response,
            "sources": sources[:3],  # Limit to top 5 sources
            "context_found": True,
            "total_sources": total_results,
            "case_id": req.case_id,
            "message": f"Answer generated using context from {len(all_results)} document(s) with {total_results} relevant sections for case '{req.case_id}'."
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/analyze-claim-structured")
async def analyze_claim_structured(req: ClaimAnalysisRequest):
    """Analyze documents for structured claim review using the defined schema."""
    try:
        # Get case-specific folder
        case_folder = os.path.join(DOCUMENTS_FOLDER, req.case_id)
        
        # Initialize VectorDBManager for the specific case
        manager = VectorDBManager(folder_path=case_folder, case_id=req.case_id)
        
        # Search across all vector databases for this case
        all_results = manager.search_all_databases(
            query=req.question,
            k=3,
            score_threshold=0.3
        )
        
        # Combine context from all databases
        all_chunks = []
        for db_name, results in all_results.items():
            for chunk, score, _ in results[:2]:  # Take top 2 from each DB
                all_chunks.append(f"[From {db_name}] {chunk}")
        
        context = "\n\n".join(all_chunks[:5])  # Limit to top 5 overall
        
        prompt = f"""Analyze the following insurance claim documents and provide a structured review summary.
        
        Document Context:
        {context}
        
        Analysis Request: {req.question}
        
        Please extract and structure all relevant information from the documents according to the claim review format."""
        
        # Use structured response method
        azure_client = get_azure_ai_client()
        structured_response = azure_client.generate_structured_response(
            prompt=prompt,
            system_content="You are an expert in processing Insurance Claims. Extract information from the provided documents and structure it according to the claim review schema.",
            max_tokens=2000
        )
        
        # Prepare source information
        sources = []
        for db_name, results in all_results.items():
            if results:
                for chunk, score, _ in results[:2]:
                    sources.append({
                        "document": db_name,
                        "relevance_score": round(float(score), 3),
                        "preview": chunk[:200] + "..." if len(chunk) > 200 else chunk
                    })
        
        return {
            "success": True,
            "structured_analysis": structured_response,
            "sources": sources[:5],
            "case_id": req.case_id,
            "message": f"Structured claim analysis completed for case '{req.case_id}' using {len(all_results)} document(s)."
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/refresh-all-vectors")
async def refresh_all_vectors(req: CaseRequest):
    """Refresh/rebuild vectors for all documents in a specific case folder."""
    try:
        case_folder = os.path.join(DOCUMENTS_FOLDER, req.case_id)
        
        if not os.path.exists(case_folder):
            return {
                "success": True,
                "message": f"No documents found for case '{req.case_id}'.",
                "files_processed": []
            }
        
        manager = VectorDBManager(folder_path=case_folder, case_id=req.case_id)
        
        supported_extensions = ['.pdf', '.docx', '.txt', '.json']
        files_to_process = []
        
        if os.path.exists(case_folder):
            for file in os.listdir(case_folder):
                if any(file.lower().endswith(ext) for ext in supported_extensions):
                    files_to_process.append(file)
        
        if not files_to_process:
            return {
                "success": True,
                "message": "No documents found to refresh.",
                "files_processed": []
            }
        
        # Process each file (rebuild vectors)
        results = []
        for filename in files_to_process:
            try:
                # Force rebuild by deleting existing vectors first
                if manager.vectors_exist(filename):
                    paths = manager.get_vectordb_paths(filename)
                    for path in paths.values():
                        if os.path.exists(path):
                            os.remove(path)
                
                # Build new vectors
                success = manager.build_vector_db(filename)
                results.append({
                    "filename": filename,
                    "success": success,
                    "message": "Refreshed successfully" if success else "Failed to refresh"
                })
            except Exception as e:
                results.append({
                    "filename": filename,
                    "success": False,
                    "message": f"Error: {str(e)}"
                })
        
        return {
            "success": True,
            "message": f"Refreshed {len(files_to_process)} document(s) for case '{req.case_id}'.",
            "files_processed": results
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/delete-all-vectors")
async def delete_all_vectors(req: CaseRequest):
    """Delete all vector databases for a specific case."""
    try:
        case_folder = os.path.join(DOCUMENTS_FOLDER, req.case_id)
        
        if not os.path.exists(case_folder):
            return {
                "success": True,
                "message": f"No vectors found for case '{req.case_id}'.",
                "files_deleted": []
            }
        
        manager = VectorDBManager(folder_path=case_folder, case_id=req.case_id)
        
        # Get all files in case folder
        files_deleted = []
        if os.path.exists(case_folder):
            for file in os.listdir(case_folder):
                if manager.vectors_exist(file):
                    try:
                        paths = manager.get_vectordb_paths(file)
                        deleted_count = 0
                        for path in paths.values():
                            if os.path.exists(path):
                                os.remove(path)
                                deleted_count += 1
                        
                        if deleted_count > 0:
                            files_deleted.append(file)
                    except Exception as e:
                        continue
        
        return {
            "success": True,
            "message": f"Deleted vectors for {len(files_deleted)} document(s) in case '{req.case_id}'.",
            "files_deleted": files_deleted
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_server:app", host="0.0.0.0", port=8001, reload=True)
