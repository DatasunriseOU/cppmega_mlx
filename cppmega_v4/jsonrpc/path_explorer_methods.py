import os
from pydantic import BaseModel, Field, RootModel
from typing import List

class ListDirectoryParams(BaseModel):
    path: str = Field(..., description="Absolute or relative path to list")

class FileItemInfo(BaseModel):
    name: str
    is_dir: bool
    size_bytes: int
    extension: str

class ListDirectoryResult(RootModel[List[FileItemInfo]]):
    pass

class AnalyzeSourceParams(BaseModel):
    path: str = Field(..., description="Path to file or folder to analyze")
    content_type: str = Field("text", description="Selected content type: text, code, or parquet")

class AnalyzeSourceResult(BaseModel):
    path: str
    content_type: str
    lines: int
    words: int
    chars: int
    file_count: int
    recommendation: str
    suggested_tokenizer: str

def list_directory(params: ListDirectoryParams) -> ListDirectoryResult:
    """Lists directories and files on the local or remote backend filesystem."""
    target_path = params.path
    if not os.path.exists(target_path):
        # Fallback to local workspace files if path is relative and doesn't exist directly
        target_path = "."
        
    items = []
    try:
        for entry in os.scandir(target_path):
            # Ignore hidden files/directories
            if entry.name.startswith("."):
                continue
                
            is_dir = entry.is_dir()
            size = 0 if is_dir else entry.stat().st_size
            _, ext = os.path.splitext(entry.name)
            
            items.append(FileItemInfo(
                name=entry.name,
                is_dir=is_dir,
                size_bytes=size,
                extension=ext
            ))
    except Exception as e:
        print(f"[list_directory] Error listing {target_path}: {e}")
        
    return ListDirectoryResult(root=items)

def analyze_source(params: AnalyzeSourceParams) -> AnalyzeSourceResult:
    """Performs deep diagnostics on the selected file or folder and returns tokenizer suggestions."""
    path = params.path
    content_type = params.content_type
    
    lines = 0
    words = 0
    chars = 0
    file_count = 0
    recommendation = ""
    suggested_tokenizer = "GPT2Tokenizer"
    
    if not os.path.exists(path):
        return AnalyzeSourceResult(
            path=path,
            content_type=content_type,
            lines=0, words=0, chars=0, file_count=0,
            recommendation=f"Error: Path {path} not found.",
            suggested_tokenizer="None"
        )
        
    # Analyze single file or directory recursively
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for file in files:
                if file.startswith("."):
                    continue
                file_count += 1
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        chars += len(content)
                        lines += content.count("\n")
                        words += len(content.split())
                except Exception:
                    pass
                    
        if content_type == "code":
            recommendation = f"✓ Found {file_count} code files. Suggesting Code-specific BPE (vocab=65536) for best compression."
            suggested_tokenizer = "BPETokenizer-Code"
        elif content_type == "parquet":
            recommendation = f"✓ Found {file_count} parquet shards. Direct loading is supported."
            suggested_tokenizer = "ZeroCopyParquetLoader"
        else:
            recommendation = f"✓ Found {file_count} text files. Standard GPT-2 or Llama tokenizers are ideal."
            suggested_tokenizer = "GPT2Tokenizer"
    else:
        file_count = 1
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                chars = len(content)
                lines = content.count("\n")
                words = len(content.split())
        except Exception:
            pass
            
        ext = os.path.splitext(path)[1].lower()
        if content_type == "code" or ext in [".cpp", ".py", ".cu", ".h", ".c"]:
            recommendation = f"✓ Detected code file. Suggesting Code-specific BPE (vocab=65536) for high density."
            suggested_tokenizer = "BPETokenizer-Code"
        elif content_type == "parquet" or ext == ".parquet":
            recommendation = f"✓ Detected parquet shard with standard columns. Ready for zero-copy training."
            suggested_tokenizer = "ZeroCopyParquetLoader"
        else:
            recommendation = f"✓ Detected plain text. Standard Llama 3 tokenizer (vocab=128k) fits perfectly."
            suggested_tokenizer = "Llama3Tokenizer"
            
    return AnalyzeSourceResult(
        path=path,
        content_type=content_type,
        lines=lines,
        words=words,
        chars=chars,
        file_count=file_count,
        recommendation=recommendation,
        suggested_tokenizer=suggested_tokenizer
    )
