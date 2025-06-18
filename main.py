from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Union
import uvicorn
import logging

# Import your BM25 encoder classes
from bm25_encoder import BM25Encoder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="BM25 Encoder API",
    description="API for BM25 text encoding and sparse vector generation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global BM25 encoder instance
bm25_encoder: Optional[BM25Encoder] = None


# Pydantic models for request/response
class BM25Config(BaseModel):
    b: float = Field(default=0.75, ge=0.0, le=1.0, description="Length normalization parameter")
    k1: float = Field(default=1.2, ge=0.0, description="Term frequency normalization parameter")
    lower_case: bool = Field(default=True, description="Whether to lowercase tokens")
    remove_punctuation: bool = Field(default=True, description="Whether to remove punctuation")
    remove_stopwords: bool = Field(default=True, description="Whether to remove stopwords")
    stem: bool = Field(default=True, description="Whether to stem tokens")
    language: str = Field(default="english", description="Language for stopwords and stemmer")


class FitRequest(BaseModel):
    corpus: List[str] = Field(..., description="List of documents to fit BM25 with")
    config: Optional[BM25Config] = Field(default=None, description="BM25 configuration")


class EncodeRequest(BaseModel):
    texts: Union[str, List[str]] = Field(..., description="Text(s) to encode")


class SparseVector(BaseModel):
    indices: List[int] = Field(..., description="Token indices")
    values: List[float] = Field(..., description="Token weights")


class EncodeResponse(BaseModel):
    vectors: Union[SparseVector, List[SparseVector]] = Field(..., description="Encoded sparse vectors")


class StatusResponse(BaseModel):
    status: str
    message: str
    is_fitted: bool = False
    n_docs: Optional[int] = None
    avgdl: Optional[float] = None

@app.post("/fit", response_model=StatusResponse)
async def fit_bm25(request: FitRequest, background_tasks: BackgroundTasks):
    """Fit BM25 encoder with a corpus of documents"""
    global bm25_encoder

    try:
        # Validate corpus
        if not request.corpus:
            raise HTTPException(status_code=400, detail="Corpus cannot be empty")

        if len(request.corpus) > 10000:
            raise HTTPException(status_code=400, detail="Corpus too large (max 10,000 documents)")

        # Initialize BM25 encoder with config
        config = request.config or BM25Config()
        bm25_encoder = BM25Encoder(
            b=config.b,
            k1=config.k1,
            lower_case=config.lower_case,
            remove_punctuation=config.remove_punctuation,
            remove_stopwords=config.remove_stopwords,
            stem=config.stem,
            language=config.language
        )

        # Fit the encoder
        logger.info(f"Fitting BM25 encoder with {len(request.corpus)} documents")
        bm25_encoder.fit(request.corpus)

        return StatusResponse(
            status="success",
            message=f"BM25 encoder fitted successfully with {len(request.corpus)} documents",
            is_fitted=True,
            n_docs=bm25_encoder.n_docs,
            avgdl=bm25_encoder.avgdl
        )

    except Exception as e:
        logger.error(f"Error fitting BM25 encoder: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fit BM25 encoder: {str(e)}")


@app.post("/encode/documents", response_model=EncodeResponse)
async def encode_documents(request: EncodeRequest):
    """Encode documents to sparse vectors"""
    global bm25_encoder

    if bm25_encoder is None or bm25_encoder.doc_freq is None:
        raise HTTPException(status_code=400, detail="BM25 encoder not fitted. Call /fit first.")

    try:
        vectors = bm25_encoder.encode_documents(request.texts)

        # Convert to response format
        if isinstance(vectors, dict):
            vectors = SparseVector(indices=vectors["indices"], values=vectors["values"])
        else:
            vectors = [SparseVector(indices=v["indices"], values=v["values"]) for v in vectors]

        return EncodeResponse(vectors=vectors)

    except Exception as e:
        logger.error(f"Error encoding documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to encode documents: {str(e)}")


@app.post("/encode/queries", response_model=EncodeResponse)
async def encode_queries(request: EncodeRequest):
    """Encode queries to sparse vectors"""
    global bm25_encoder

    if bm25_encoder is None or bm25_encoder.doc_freq is None:
        raise HTTPException(status_code=400, detail="BM25 encoder not fitted. Call /fit first.")

    try:
        vectors = bm25_encoder.encode_queries(request.texts)

        # Convert to response format
        if isinstance(vectors, dict):
            vectors = SparseVector(indices=vectors["indices"], values=vectors["values"])
        else:
            vectors = [SparseVector(indices=v["indices"], values=v["values"]) for v in vectors]

        return EncodeResponse(vectors=vectors)

    except Exception as e:
        logger.error(f"Error encoding queries: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to encode queries: {str(e)}")

if __name__ == "__main__":
    # For local development
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )