/**
 * Math foundations — "explained for humans" links per concept.
 * Sourced from research agent. Priority: 3Blue1Brown → distill.pub →
 * Raschka → Alammar → EleutherAI → Maarten Grootendorst → Wikipedia.
 *
 * Used by HelpModal to render a "Math foundations" section under each
 * brick / adapter explanation. Each topic_key references concepts a
 * brick's "what/why" text might touch on; the registry below is
 * cross-cutting (a single concept can be linked from many bricks).
 */


export interface MathConcept {
  gloss: string;
  urls: { title: string; url: string }[];
  key_insight: string;
}


export const MATH_FOUNDATIONS: Record<string, MathConcept> = {
  vector_norm: {
    gloss:
      "The norm of a vector is its length — for the L2/Euclidean norm, " +
      "the straight-line distance from the origin, computed as the " +
      "square root of the sum of squared components.",
    urls: [
      { title: "3Blue1Brown — Vectors | Chapter 1, Essence of linear algebra",
        url: "https://www.3blue1brown.com/lessons/vectors" },
      { title: "Khan Academy — Vector magnitude & normalization",
        url: "https://www.khanacademy.org/computing/computer-programming/programming-natural-simulations/programming-vectors/a/vector-magnitude-normalization" },
    ],
    key_insight:
      "Normalizing a vector divides it by its norm so the result has " +
      "length 1, keeping direction while erasing magnitude.",
  },
  rms_norm: {
    gloss:
      "RMSNorm rescales a vector by its root-mean-square so that every " +
      "output vector has the same characteristic size, regardless of how " +
      "big or small the inputs were.",
    urls: [
      { title: "Sebastian Raschka — Why do many modern LLMs use RMSNorm " +
                "instead of LayerNorm?",
        url: "https://sebastianraschka.com/faq/docs/rmsnorm-vs-layernorm.html" },
      { title: "Sebastian Raschka — The Big LLM Architecture Comparison " +
                "(RMSNorm section)",
        url: "https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison" },
    ],
    key_insight:
      "After dividing by RMS(x), the output vector's RMS is exactly 1 — " +
      "a learnable gain then rescales it, but the activation magnitude " +
      "is fixed before that.",
  },
  dot_product: {
    gloss:
      "The dot product of two vectors measures how much they point in " +
      "the same direction — algebraically it's the sum of element-wise " +
      "products, geometrically it's |a||b|cosθ.",
    urls: [
      { title: "3Blue1Brown — Dot products and duality | Chapter 9, " +
                "Essence of linear algebra",
        url: "https://www.3blue1brown.com/lessons/dot-products" },
    ],
    key_insight:
      "A positive dot product means the vectors broadly agree; zero means " +
      "they're perpendicular; negative means they oppose — this is exactly " +
      "what attention uses to score query-key similarity.",
  },
  matrix_multiplication: {
    gloss:
      "Multiplying A·B applies the linear transformation B first, then A; " +
      "each output entry is the dot product of a row of A with a column of B.",
    urls: [
      { title: "3Blue1Brown — Matrix multiplication as composition | Ch. 4",
        url: "https://www.3blue1brown.com/lessons/matrix-multiplication" },
      { title: "3Blue1Brown video — Matrix multiplication as composition",
        url: "https://www.youtube.com/watch?v=XkY2DOUCWMU" },
    ],
    key_insight:
      "Matrix multiplication is function composition on linear maps — " +
      "that's why order matters and why neural-net layers can be stacked " +
      "as one big linear operation between nonlinearities.",
  },
  softmax: {
    gloss:
      "Softmax turns an arbitrary vector of real-valued logits into a " +
      "probability distribution by exponentiating each value and " +
      "dividing by the total.",
    urls: [
      { title: "3Blue1Brown — Attention in transformers (uses softmax) | Ch. 6",
        url: "https://www.3blue1brown.com/lessons/attention" },
      { title: "Wikipedia — Softmax function",
        url: "https://en.wikipedia.org/wiki/Softmax_function" },
    ],
    key_insight:
      "Exponentiation makes the biggest logit dominate while still giving " +
      "smaller logits some non-zero share — it's a 'soft' version of " +
      "argmax that is differentiable.",
  },
  attention_mechanism: {
    gloss:
      "Attention lets each token look at every other token, scoring " +
      "relevance with query·key dot products and pulling in a weighted " +
      "mixture of their value vectors.",
    urls: [
      { title: "3Blue1Brown — Attention in transformers, step-by-step",
        url: "https://www.youtube.com/watch?v=eMlx5fFNoYc" },
      { title: "Jay Alammar — The Illustrated Transformer",
        url: "https://jalammar.github.io/illustrated-transformer/" },
    ],
    key_insight:
      "Q, K, V are three different learned projections of the same token " +
      "— Q asks 'what am I looking for?', K answers 'what do I offer?', V " +
      "carries 'here's what I'd contribute if you chose me'.",
  },
  rotary_position_embedding: {
    gloss:
      "RoPE encodes position by rotating each query and key vector in 2D " +
      "subspaces by an angle proportional to its position in the sequence.",
    urls: [
      { title: "EleutherAI Blog — Rotary Embeddings: A Relative Revolution",
        url: "https://blog.eleuther.ai/rotary-embeddings/" },
    ],
    key_insight:
      "Because the dot product of two rotated vectors depends only on the " +
      "difference of their rotation angles, attention scores end up " +
      "depending on the *relative* distance between tokens.",
  },
  residual_stream: {
    gloss:
      "A residual connection adds a layer's input back to its output, so " +
      "the network always carries forward a running 'stream' that every " +
      "block reads from and writes to.",
    urls: [
      { title: "3Blue1Brown — Transformers, the tech behind LLMs",
        url: "https://www.youtube.com/watch?v=wjZofJX0v4M" },
      { title: "Anthropic — A Mathematical Framework for Transformer Circuits",
        url: "https://transformer-circuits.pub/2021/framework/" },
    ],
    key_insight:
      "Treating the residual stream as a shared communication bus — where " +
      "attention and MLP blocks read from and write into it — explains " +
      "both gradient flow and how features compose across depth.",
  },
  layer_normalization: {
    gloss:
      "Layer normalization subtracts the mean and divides by the standard " +
      "deviation across a token's features, then applies a learned scale " +
      "and shift.",
    urls: [
      { title: "Sebastian Raschka — RMSNorm vs LayerNorm",
        url: "https://sebastianraschka.com/faq/docs/rmsnorm-vs-layernorm.html" },
      { title: "Wikipedia — Layer normalization",
        url: "https://en.wikipedia.org/wiki/Layer_normalization" },
    ],
    key_insight:
      "Normalizing per token keeps activation magnitudes from exploding " +
      "or vanishing as they pass through dozens of layers.",
  },
  gradient_descent: {
    gloss:
      "Gradient descent tunes a model by repeatedly nudging its " +
      "parameters a small step in the direction that most decreases the " +
      "loss — downhill on the loss surface.",
    urls: [
      { title: "3Blue1Brown — Gradient descent, how neural networks learn",
        url: "https://www.3blue1brown.com/lessons/gradient-descent" },
      { title: "Distill — Why Momentum Really Works",
        url: "https://distill.pub/2017/momentum/" },
    ],
    key_insight:
      "SGD takes the same step but using a noisy estimate of the gradient " +
      "from a mini-batch — both faster per step and a regularizer.",
  },
  backpropagation: {
    gloss:
      "Backpropagation computes the gradient of the loss with respect to " +
      "every parameter by applying the chain rule layer by layer.",
    urls: [
      { title: "3Blue1Brown — Backpropagation, intuitively",
        url: "https://www.3blue1brown.com/lessons/backpropagation" },
      { title: "3Blue1Brown — Backpropagation calculus",
        url: "https://www.3blue1brown.com/lessons/backpropagation-calculus" },
    ],
    key_insight:
      "Each parameter's gradient is the product of local derivatives along " +
      "the path back from the loss — one forward + backward pass tells " +
      "every weight which direction to push.",
  },
  selective_ssm: {
    gloss:
      "Selective state-space models like Mamba replace attention with a " +
      "recurrent state whose update rules themselves depend on the input " +
      "— so the model selectively remembers or forgets per token.",
    urls: [
      { title: "Maarten Grootendorst — A Visual Guide to Mamba and SSMs",
        url: "https://www.maartengrootendorst.com/blog/mamba/" },
    ],
    key_insight:
      "Because the recurrence is linear, it can be evaluated in parallel " +
      "during training (transformer-like) but run with O(1) state per step " +
      "at inference (RNN-like).",
  },
  mixture_of_experts: {
    gloss:
      "A mixture-of-experts layer holds many parallel feed-forward " +
      "'experts' and, for each token, a router picks the top-k experts " +
      "whose outputs get combined.",
    urls: [
      { title: "Maarten Grootendorst — A Visual Guide to Mixture of Experts",
        url: "https://www.maartengrootendorst.com/blog/moe/" },
      { title: "Sebastian Raschka — Mixture of Experts (MoE)",
        url: "https://sebastianraschka.com/llms-from-scratch/ch04/07_moe/" },
    ],
    key_insight:
      "Only k of N experts run per token, so total parameters can be huge " +
      "while compute per token stays small.",
  },
  low_rank_approximation: {
    gloss:
      "LoRA freezes the original weight matrix W and learns a tiny " +
      "additive update ΔW = B·A where B and A are skinny matrices, so " +
      "the effective rank of the update is small.",
    urls: [
      { title: "Sebastian Raschka — Parameter-Efficient LLM Finetuning " +
                "With LoRA",
        url: "https://sebastianraschka.com/blog/2023/llm-finetuning-lora.html" },
    ],
    key_insight:
      "Empirically, the change needed to adapt a pretrained model to a " +
      "new task lies in a low-dimensional subspace.",
  },
};


/**
 * Per-topic → list-of-concept-keys mapping. When a HelpModal renders,
 * we look up the topic here and surface the listed math concepts as
 * a "Math foundations" section under the diagram.
 */
export const TOPIC_FOUNDATIONS: Record<string, string[]> = {
  brick_attention: [
    "attention_mechanism", "dot_product", "softmax",
    "matrix_multiplication", "residual_stream",
  ],
  brick_gated_attention: [
    "attention_mechanism", "rotary_position_embedding", "softmax",
    "rms_norm", "residual_stream",
  ],
  brick_mla: [
    "attention_mechanism", "low_rank_approximation",
    "rotary_position_embedding", "softmax", "residual_stream",
  ],
  brick_mla_absorb: [
    "attention_mechanism", "low_rank_approximation",
    "rotary_position_embedding", "matrix_multiplication",
  ],
  brick_mistral4_mla: [
    "attention_mechanism", "low_rank_approximation",
    "rotary_position_embedding",
  ],
  brick_gqa_sliding: [
    "attention_mechanism", "softmax", "dot_product",
  ],
  brick_cca_attention: [
    "attention_mechanism", "matrix_multiplication", "softmax",
  ],
  brick_mlp: [
    "matrix_multiplication", "residual_stream",
  ],
  brick_moe: [
    "mixture_of_experts", "softmax", "matrix_multiplication",
  ],
  brick_bailing_moe: ["mixture_of_experts", "softmax"],
  brick_mamba3: [
    "selective_ssm", "matrix_multiplication", "residual_stream",
  ],
  brick_mlstm: ["selective_ssm", "matrix_multiplication"],
  brick_gdn: ["selective_ssm", "matrix_multiplication"],
  brick_kda: ["selective_ssm", "matrix_multiplication"],
  adapter_rmsnorm: [
    "rms_norm", "vector_norm", "layer_normalization",
  ],
  brick_abs_pos_embed: ["residual_stream"],
  // Training-related topics surface foundational math too.
  train_grad_clip: ["gradient_descent", "backpropagation"],
  train_loss_scaler: ["gradient_descent"],
  metric_perplexity: ["softmax"],
};
