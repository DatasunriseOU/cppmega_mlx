import markdown

text = """
| Path | What it is |
|---|---|
| **Path A** | Native MLX ops — `mx.matmul`, \
`mx.fast.scaled_dot_product_attention` |
"""

extensions = ['tables']
html = markdown.markdown(text, extensions=extensions)
print(html)
