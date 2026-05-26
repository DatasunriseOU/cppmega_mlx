import mlx.core as mx
from cppmega_v4.buildspec.api import BuiltSequentialModel

class MockNode:
    def __init__(self, name, kind, params=None):
        self.name = name
        self.kind = kind
        self.params = params or {}
        self.module = None

def test_residual_sum_reduction():
    # Build a small graph representing:
    # x0 -> embedding_table -> x1
    # x1 -> active_branch (identity/mlp) -> x2_active
    # x1 -> bypass_branch (identity) -> x2_bypass
    # x2_active + x2_bypass -> residual_add -> x3 (summed!)
    
    nodes = [
        MockNode("tokenizer_1", "tokenizer"),
        MockNode("input_embedder", "embedding_table"),
        MockNode("active_branch", "mlp"), # active branch computes some transform or identity
        MockNode("residual_add", "residual_add"),
        MockNode("output_deembedder", "embedding_table"),
        MockNode("detokenizer_1", "detokenizer")
    ]
    
    edges = [
        ("tokenizer_1", "input_embedder"),
        # Split point from input_embedder
        ("input_embedder", "active_branch"),
        ("input_embedder", "residual_add"), # bypass connection
        # Converge point to residual_add
        ("active_branch", "residual_add"),
        ("residual_add", "output_deembedder"),
        ("output_deembedder", "detokenizer_1")
    ]
    
    class MockGraph:
        def __init__(self, nodes, edges):
            self.nodes = nodes
            self.edges = edges
            
    graph = MockGraph(nodes, edges)
    
    # Instantiate the compiler model
    model = BuiltSequentialModel(graph, hidden_size=64)
    
    # Run forward pass with a dummy input tensor
    x = mx.random.normal((2, 64))
    outputs = model(x)
    
    # Verify outputs
    assert "input_embedder" in outputs
    assert "active_branch" in outputs
    assert "residual_add" in outputs
    
    input_emb = outputs["input_embedder"]
    active = outputs["active_branch"]
    res_add = outputs["residual_add"]
    
    # Since active_branch has no backend block registered, it acts as identity: active == input_emb
    # And since residual_add sums active + input_emb, res_add should mathematically be active + input_emb = 2 * input_emb!
    expected = input_emb + active
    assert mx.allclose(res_add, expected).item()
    print("Residual sum reduction verified successfully!")
