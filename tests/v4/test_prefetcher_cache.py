import os
import shutil
import pytest
from cppmega_v4.data.pre_fetcher import StreamingPreFetcher
from cppmega_v4.jsonrpc.path_explorer_methods import (
    ListDirectoryParams, list_directory,
    AnalyzeSourceParams, analyze_source
)

def test_prefetcher_caching_and_bypass():
    """Verifies that the StreamingPreFetcher generates local parquet files and cache hits bypass network."""
    test_cache_dir = "data/cache/test_datasets"
    if os.path.exists(test_cache_dir):
        shutil.rmtree(test_cache_dir)
        
    try:
        # 1. First run: cache miss, downloads and tokenizes
        fetcher = StreamingPreFetcher(
            dataset_id="HuggingFaceFW/fineweb-edu",
            target_tokens=50000,
            cache_dir=test_cache_dir,
            min_buffer_tokens=25000
        )
        fetcher.start()
        
        # Wait for buffer to be ready
        ready = fetcher.buffer_ready_event.wait(timeout=5.0)
        assert ready, "Pre-fetcher failed to ready the buffer within timeout."
        
        status = fetcher.get_status()
        assert status["progress_percent"] > 0
        assert status["token_offset"] >= 25000
        
        # Wait for completion
        for _ in range(50):
            if fetcher.is_finished:
                break
            import time
            time.sleep(0.1)
            
        assert fetcher.is_finished
        assert len(fetcher.shards_ready) > 0
        
        # 2. Second run: cache hit, instant completion
        fetcher2 = StreamingPreFetcher(
            dataset_id="HuggingFaceFW/fineweb-edu",
            target_tokens=50000,
            cache_dir=test_cache_dir,
            min_buffer_tokens=25000
        )
        fetcher2.start()
        
        assert fetcher2.is_finished, "Cache hit should trigger instant completion."
        assert len(fetcher2.shards_ready) == len(fetcher.shards_ready)
        
    finally:
        if os.path.exists(test_cache_dir):
            shutil.rmtree(test_cache_dir)

def test_path_explorer_rpc_handlers():
    """Verifies that list_directory and analyze_source RPC methods run successfully and return expected types."""
    # Test directory listing
    list_res = list_directory(ListDirectoryParams(path="."))
    assert len(list_res) > 0
    assert any(x.name == "cppmega_v4" for x in list_res)
    
    # Test file analysis
    analyze_res = analyze_source(AnalyzeSourceParams(path="pyproject.toml", content_type="text"))
    assert analyze_res.file_count == 1
    assert analyze_res.lines > 0
    assert "Standard Llama 3" in analyze_res.recommendation
