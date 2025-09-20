# test_graphcs.py
# Run with: python test_graphcs.py --ckpt graphcs.pt --max_samples 50 --beam 4

import argparse
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score

# -----------------------------------------------------
# Import the functions/classes from your training file.
# If your training code is in the same notebook or module,
# adjust imports accordingly. For example:
# from graphcs_train_file import (build_graph_from_python_ast, Vocab,
#                                CodeSummaryDataset, collate_batch, GraphCS, beam_search)
# -----------------------------------------------------
# For demonstration assume the functions and classes are in scope:
# build_graph_from_python_ast, Vocab, CodeSummaryDataset, collate_batch, GraphCS, beam_search

# If not running in the same module, import them:
# from your_training_module import build_graph_from_python_ast, Vocab, CodeSummaryDataset, collate_batch, GraphCS, beam_search

smooth_bleu = SmoothingFunction().method4

def compute_bleu(ref_tokens, hyp_tokens):
    if len(hyp_tokens) == 0:
        return 0.0
    return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smooth_bleu)

def compute_meteor(ref, hyp):
    # meteor_score expects strings (reference list, hypothesis string)
    # we'll join tokens with space
    return meteor_score([' '.join(ref)], ' '.join(hyp))

def lcs(a, b):
    # Longest common subsequence length (tokens)
    # Classic DP O(len(a)*len(b))
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0
    dp = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la - 1, -1, -1):
        for j in range(lb - 1, -1, -1):
            if a[i] == b[j]:
                dp[i][j] = 1 + dp[i + 1][j + 1]
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
    return dp[0][0]

def compute_rouge_l(ref_tokens, hyp_tokens):
    if len(ref_tokens) == 0 or len(hyp_tokens) == 0:
        return 0.0
    l = lcs(ref_tokens, hyp_tokens)
    prec = l / (len(hyp_tokens) + 1e-12)
    rec = l / (len(ref_tokens) + 1e-12)
    if prec + rec == 0:
        return 0.0
    beta = 1.0  # F1 harmonizes precision and recall equally
    score = (1 + beta**2) * prec * rec / (rec + beta**2 * prec + 1e-12)
    return score

def detokenize(tokens):
    """Convert list of token strings to a string (for METEOR)."""
    return " ".join(tokens)

def evaluate(model, test_pairs, code_vocab, sum_vocab, device="cpu", beam=4, max_samples=None):
    model.to(device)
    model.eval()

    n = len(test_pairs)
    if max_samples is not None:
        n = min(n, max_samples)
        test_pairs = test_pairs[:n]

    bleu_scores = []
    meteor_scores = []
    rouge_l_scores = []

    for code, reference in tqdm(test_pairs, desc="Evaluating"):
        # Build ids and adjacency for single example (reuse your helper)
        nodes, edges = build_graph_from_python_ast(code)
        code_ids = torch.tensor([code_vocab.encode(nodes)], dtype=torch.long)
        L = code_ids.size(1)
        adj = torch.zeros(1, L, L)
        for u, v in edges:
            if u < L and v < L:
                adj[0, u, v] = 1

        # Generate via beam search (returns string)
        gen = beam_search(model, code_ids, adj, sum_vocab, beam=beam, max_len=50, device=device)

        # Tokenize generated string and reference for metrics
        gen_tokens = nltk.word_tokenize(gen.lower())
        ref_tokens = nltk.word_tokenize(reference.lower())

        bleu = compute_bleu(ref_tokens, gen_tokens)
        meteor = compute_meteor(ref_tokens, gen_tokens)
        rouge_l = compute_rouge_l(ref_tokens, gen_tokens)

        bleu_scores.append(bleu)
        meteor_scores.append(meteor)
        rouge_l_scores.append(rouge_l)

    # Convert to percentages (conventional)
    avg_bleu = 100.0 * (sum(bleu_scores) / len(bleu_scores)) if bleu_scores else 0.0
    avg_meteor = 100.0 * (sum(meteor_scores) / len(meteor_scores)) if meteor_scores else 0.0
    avg_rouge_l = 100.0 * (sum(rouge_l_scores) / len(rouge_l_scores)) if rouge_l_scores else 0.0

    return {
        "BLEU": avg_bleu,
        "METEOR": avg_meteor,
        "ROUGE-L": avg_rouge_l,
        "num_samples": len(bleu_scores)
    }

def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default=None, help="Path to saved model state_dict (optional)")
    parser.add_argument('--max_samples', type=int, default=None, help="Max number of test samples")
    parser.add_argument('--beam', type=int, default=4, help="Beam size for generation")
    parser.add_argument('--device', type=str, default=None, help="device (cpu or cuda)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- Example: load test set ----------
    # Replace with your real test set loader. For demo we re-use the toy pairs.
    test_pairs = [
        ("def add(a,b): return a+b","add two numbers"),
        ("def power(a,b): return a**b","raise number to power"),
        ("def is_even(n): return n%2==0","check if number is even"),
        ("def factorial(n): return 1 if n==0 else n*factorial(n-1)","compute factorial"),
        # Add more test examples or load from file...
    ]

    # ---------- Build dataset / vocabs same as training ----------
    ds = CodeSummaryDataset(test_pairs)  # uses build_graph_from_python_ast inside
    # We must build vocabs consistent with model training vocab.
    # If you saved vocabs during training, load them here instead. For demo, build from test set:
    code_vocab = Vocab(); code_vocab.build([c for c, _ in ds])
    sum_vocab = Vocab(); sum_vocab.build([s for _, s in ds])

    # ---------- Initialize model architecture ----------
    model = GraphCS(len(code_vocab), len(sum_vocab))
    if args.ckpt and os.path.isfile(args.ckpt):
        print(f"Loading checkpoint: {args.ckpt}")
        model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    else:
        print("No checkpoint provided or not found — evaluating untrained model (demo only).")

    # ---------- Evaluate ----------
    metrics = evaluate(model, test_pairs, code_vocab, sum_vocab, device=device, beam=args.beam, max_samples=args.max_samples)
    print("-" * 40)
    print(f"Evaluated {metrics['num_samples']} samples")
    print(f"BLEU (avg):   {metrics['BLEU']:.4f}")
    print(f"METEOR (avg): {metrics['METEOR']:.4f}")
    print(f"ROUGE-L (avg):{metrics['ROUGE-L']:.4f}")
    print("-" * 40)

if __name__ == "__main__":
    # Ensure NLTK punkt and meteor resources available
    nltk.download('punkt', quiet=True)
    # meteor_score may require wordnet; if needed download:
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)

    main_cli()
