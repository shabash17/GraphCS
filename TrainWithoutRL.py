# ===============================
# Colab-ready GraphCS with RAML (BLEU reward)
# ===============================
!pip install -q nltk tqdm

import ast, math, random, nltk, torch, time
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True) # Download the missing resource

# ----------------------------
# 1. Preprocessing (tokenization + AST->graph)
# ----------------------------
def split_identifier(tok):
    parts=[]
    for piece in tok.split('_'):
        cur=''
        for ch in piece:
            if ch.isupper() and cur and not cur.isupper():
                parts.append(cur.lower()); cur=ch
            else:
                cur+=ch
        if cur: parts.append(cur.lower())
    return [p for p in parts if p]

def code_tokenize(code):
    tokens,cur=[], ''
    for ch in code:
        if ch.isalnum() or ch=='_': cur+=ch
        else:
            if cur: tokens.extend(split_identifier(cur)); cur=''
            if not ch.isspace(): tokens.append(ch)
    if cur: tokens.extend(split_identifier(cur))
    return tokens

def build_graph_from_python_ast(code):
    try:
        tree=ast.parse(code)
    except Exception as e:
        print(f"Error parsing code: {code}")
        print(f"Error details: {e}")
        toks=code_tokenize(code)
        return toks, [(i,i) for i in range(len(toks))]

    nodes,edges,node_idx,idx=[],[],{},0
    def add_node(n):
        nonlocal idx
        key = (type(n), getattr(n,'lineno',None), getattr(n,'col_offset',None), str(getattr(n,'id', '') or getattr(n,'arg','') or getattr(n,'value','')))
        if key in node_idx:
            return node_idx[key]
        node_idx[key]=idx
        label=type(n).__name__
        if isinstance(n,ast.Name): label=n.id
        elif isinstance(n,ast.Constant): label=str(n.value)
        elif isinstance(n,ast.arg): label=n.arg
        nodes.append(label); idx+=1; return node_idx[key]

    def walk(parent,node):
        if node is None: return
        u=add_node(node)
        if parent is not None:
            v=add_node(parent); edges.extend([(u,v),(v,u)])
        for child in ast.iter_child_nodes(node): walk(node,child)

    walk(None,tree)
    if not nodes:
        nodes,edges=code_tokenize(code),[(i,i) for i in range(len(code_tokenize(code)))]
    edges.extend([(i,i) for i in range(len(nodes))])
    return nodes,edges

# ----------------------------
# 2. Vocab + Dataset
# ----------------------------
class Vocab:
    def __init__(self,specials=['<pad>','<unk>','<s>','</s>']):
        self.specials=specials; self.freq=Counter(); self.itos=[]; self.stoi={}
    def build(self,seqs,min_freq=1,max_size=None):
        for seq in seqs:
            # sequences might be lists of tokens
            self.freq.update(seq)
        items=[t for t,c in self.freq.items() if c>=min_freq]
        items=sorted(items,key=lambda t:(-self.freq[t],t))
        if max_size:
            items=items[:max_size-len(self.specials)]
        self.itos=list(self.specials)+items
        self.stoi={t:i for i,t in enumerate(self.itos)}
    def encode(self,seq): return [self.stoi.get(t,self.stoi.get('<unk>',1)) for t in seq]
    def __len__(self): return len(self.itos)

class CodeSummaryDataset(Dataset):
    def __init__(self,pairs):
        self.codes,self.graphs,self.summaries=[],[],[]
        for code,summ in pairs:
            nodes,edges=build_graph_from_python_ast(code)
            self.codes.append(nodes); self.graphs.append(edges)
            self.summaries.append(nltk.word_tokenize(summ.lower()))
    def __len__(self): return len(self.codes)
    def __getitem__(self,i): return self.codes[i],self.graphs[i],self.summaries[i]

def collate_batch(batch,vocab_code,vocab_sum,max_code_len=150,max_sum_len=50):
    codes,graphs,sums=zip(*batch)
    # sequential tokens
    code_ids=[torch.tensor(vocab_code.encode(c[:max_code_len])) for c in codes]
    code_ids=pad_sequence(code_ids,batch_first=True,padding_value=vocab_code.stoi['<pad>'])
    # adjacency matrix
    B,L=code_ids.size(); adjs=torch.zeros(B,L,L)
    for i,(nodes,edges) in enumerate(zip(codes,graphs)):
        n=min(len(nodes),L)
        for u,v in edges:
            if u<n and v<n: adjs[i,u,v]=1
    # summaries
    sum_ids=[]
    for s in sums:
        toks=['<s>']+s[:max_sum_len-2]+['</s>']
        ids=torch.tensor(vocab_sum.encode(toks))
        sum_ids.append(ids)
    sum_ids=pad_sequence(sum_ids,batch_first=True,padding_value=vocab_sum.stoi['<pad>'])
    return code_ids,adjs,sum_ids

# ----------------------------
# 3. GraphCS Model (encoders, GAT, fusion, decoder)
# ----------------------------
class SeqEncoder(nn.Module):
    def __init__(self,vocab_size,d=256):
        super().__init__()
        self.emb=nn.Embedding(vocab_size,d, padding_idx=0)
        self.gru=nn.GRU(d,d//2,batch_first=True,bidirectional=True)
    def forward(self,ids):
        emb=self.emb(ids)
        out,_=self.gru(emb)
        return out

class GATLayer(nn.Module):
    def __init__(self,d,dropout=0.1):
        super().__init__()
        self.W=nn.Linear(d,d); self.a=nn.Linear(2*d,1); self.drop=nn.Dropout(dropout)
    def forward(self,h,adj):
        Wh=self.W(h); B,N,F=Wh.size()
        Wh_i=Wh.unsqueeze(2).expand(B,N,N,F)
        Wh_j=Wh.unsqueeze(1).expand(B,N,N,F)
        e=torch.tanh(self.a(torch.cat([Wh_i,Wh_j],-1))).squeeze(-1)
        e=e+((adj==0).float()*-1e9)
        attn=e.softmax(-1); attn=self.drop(attn)
        return torch.bmm(attn,Wh)

class GraphEncoder(nn.Module):
    def __init__(self,vocab_size,d=256,layers=2):
        super().__init__()
        self.emb=nn.Embedding(vocab_size,d, padding_idx=0)
        self.layers=nn.ModuleList([GATLayer(d) for _ in range(layers)])
        self.norm=nn.LayerNorm(d)
    def forward(self,ids,adj):
        h=self.emb(ids)
        for gat in self.layers: h=self.norm(h+gat(h,adj))
        return h

class FusionAttention(nn.Module):
    def __init__(self,d,h=4):
        super().__init__()
        self.h=h; self.dk=d//h
        self.Wq=nn.Linear(d,d); self.Wg=nn.Linear(d,d)
    def forward(self,x,g):
        B,T,D=x.size()
        Q=(self.Wq(x)+self.Wg(g)).view(B,T,self.h,self.dk).transpose(1,2)
        K=x.view(B,T,self.h,self.dk).transpose(1,2)
        V=K
        scores=torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(self.dk)
        attn=F.softmax(scores,-1)
        out=torch.matmul(attn,V).transpose(1,2).reshape(B,T,D)
        return out

class GAEncoder(nn.Module):
    def __init__(self,code_vocab_size,d=256,h=4,layers=2):
        super().__init__()
        self.seq=SeqEncoder(code_vocab_size,d)
        self.gat=GraphEncoder(code_vocab_size,d)
        self.fusion=nn.ModuleList([FusionAttention(d,h) for _ in range(layers)])
        self.norm=nn.LayerNorm(d)
    def forward(self,ids,adj):
        x=self.seq(ids); g=self.gat(ids,adj)
        for f in self.fusion: x=self.norm(x+f(x,g))
        return x

class TransformerDecoder(nn.Module):
    def __init__(self,sum_vocab_size,d=256,h=4,layers=2):
        super().__init__()
        self.emb=nn.Embedding(sum_vocab_size,d, padding_idx=0)
        self.pos=nn.Embedding(512,d)
        dec_layer=nn.TransformerDecoderLayer(d,h,dropout=0.1,batch_first=True)
        self.dec=nn.TransformerDecoder(dec_layer,layers)
        self.out=nn.Linear(d,sum_vocab_size)
    def forward(self,tgt,mem):
        B,T=tgt.size()
        pos=torch.arange(T,device=tgt.device).unsqueeze(0)
        y=self.emb(tgt)+self.pos(pos)
        out=self.dec(y,mem)
        return self.out(out)

class GA_SCS(nn.Module):
    def __init__(self,code_vocab_size,sum_vocab_size, d=256):
        super().__init__()
        self.enc=GAEncoder(code_vocab_size,d)
        self.dec=TransformerDecoder(sum_vocab_size,d)
    def forward(self,code,adj,tgt):
        mem=self.enc(code,adj)
        return self.dec(tgt[:,:-1],mem)   # predict next tokens

# ----------------------------
# 4. Losses + Training (MLE + RAML)
# ----------------------------
def compute_loss(logits,targets,pad):
    return F.cross_entropy(logits.reshape(-1,logits.size(-1)),
                           targets[:,1:].reshape(-1),ignore_index=pad)

smooth_bleu = SmoothingFunction().method4
def compute_bleu(reference_tokens, hypothesis_tokens):
    # reference_tokens: list of tokens (strings), hypothesis_tokens: list of tokens
    if len(hypothesis_tokens)==0:
        return 0.0
    return sentence_bleu([reference_tokens], hypothesis_tokens, smoothing_function=smooth_bleu)

@torch.no_grad()
def sample_sequences(model, code, adj, sum_vocab, device="cpu", max_len=50, sample_topk=None):
    """
    Sample one sequence per batch-item using model (multinomial) and also return
    the per-token log-probs for each sampled token so we can compute sequence log-prob.
    """
    model.eval()
    mem = model.enc(code.to(device), adj.to(device))
    B = code.size(0)
    seqs = [[] for _ in range(B)]
    seq_logprobs = [0.0 for _ in range(B)]
    finished = [False]*B
    # initialize with <s>
    cur_tokens = [sum_vocab.stoi['<s>']]*B
    for t in range(max_len):
        tgt = torch.tensor([ [cur] for cur in cur_tokens ], dtype=torch.long, device=device)  # shape (B,1)
        logits = model.dec(tgt, mem)  # (B,1,V)
        logp = F.log_softmax(logits[:, -1, :], dim=-1)  # (B,V)
        # optionally restrict to topk
        if sample_topk is not None:
            topk_vals, topk_idx = torch.topk(logp, sample_topk, dim=-1)
            probs = topk_vals.exp()
            # renormalize
            probs = probs / probs.sum(dim=-1, keepdim=True)
            next_idx_in_topk = [torch.multinomial(probs[i],1).item() for i in range(B)]
            next_tokens = [ topk_idx[i, next_idx_in_topk[i]].item() for i in range(B) ]
            next_logps = [ topk_vals[i, next_idx_in_topk[i]].item() for i in range(B) ]
        else:
            probs = logp.exp()
            next_samples = torch.multinomial(probs,1).squeeze(-1)  # (B,)
            next_tokens = next_samples.tolist()
            next_logps = [ logp[i, next_samples[i]].item() for i in range(B) ]

        for i in range(B):
            if finished[i]:
                continue
            tok = next_tokens[i]
            seqs[i].append(tok)
            seq_logprobs[i] += float(next_logps[i])
            if tok == sum_vocab.stoi['</s>']:
                finished[i] = True
        cur_tokens = [ seqs[i][-1] if len(seqs[i])>0 else sum_vocab.stoi['<s>'] for i in range(B) ]
        if all(finished):
            break
    return seqs, seq_logprobs

def raml_loss(model, code, adj, target, sum_vocab, pad_idx, device="cpu", lam=0.85, n_samples=3, max_len=50):
    """
    RAML: for each batch sample, draw n_samples sequences and compute their BLEU rewards.
    Then compute reward-weighted negative log-prob (sequence-level) as RAML loss.
    """
    model.eval()
    B = code.size(0)
    # Collect sampled sequences and their log-probs (we will sample n_samples times)
    all_sampled_seqs = []    # list length n_samples, each element list of seqs for the batch
    all_sampled_logps = []   # parallel: list length n_samples, each element list of seq_logprob for batch
    for _ in range(n_samples):
        seqs, seq_logps = sample_sequences(model, code, adj, sum_vocab, device=device, max_len=max_len)
        all_sampled_seqs.append(seqs)
        all_sampled_logps.append(seq_logps)

    # Convert references (list of token strings) for BLEU
    refs = []
    for tgt in target:
        ref = [sum_vocab.itos[idx.item()] for idx in tgt if idx.item() not in [pad_idx, sum_vocab.stoi['<s>'], sum_vocab.stoi['</s>']]]
        refs.append(ref)

    # compute reward and sequence log-probs arrays shape (B, n_samples)
    rewards = torch.zeros(B, n_samples, device=device)
    seq_logps_tensor = torch.zeros(B, n_samples, device=device)
    for s_i in range(n_samples):
        for i in range(B):
            hyp_ids = all_sampled_seqs[s_i][i]
            hyp_tokens = [ sum_vocab.itos[t] for t in hyp_ids if t not in [pad_idx, sum_vocab.stoi['<s>'], sum_vocab.stoi['</s>']] ]
            bleu = compute_bleu(refs[i], hyp_tokens)
            rewards[i, s_i] = float(bleu)
            seq_logps_tensor[i, s_i] = float(all_sampled_logps[s_i][i])

    # Form RAML weights per sample: w ∝ exp(reward / lambda)
    with torch.no_grad():
        w = torch.exp(rewards / lam)    # (B, n_samples)
        w = w / (w.sum(dim=1, keepdim=True) + 1e-12)  # normalize per example

    # Compute RAML loss: - sum_i sum_s w_is * log p(y_is | x_i)
    # Note: seq_logps_tensor are sums of log probs; they represent log p(seq). We want expected negative log-likelihood.
    raml = - (w * seq_logps_tensor).sum(dim=1).mean()
    return raml

def train_model_raml(model,train_loader,opt,sum_vocab,epochs=10,device="cpu",
                     theta1=0.1,theta2=0.9, raml_lambda=0.85, raml_samples=2):
    pad=sum_vocab.stoi['<pad>']; model.to(device)
    for ep in range(1,epochs+1):
        model.train(); tot=0.0
        pbar = tqdm(train_loader, desc=f"Epoch{ep}")
        for code,adj,sums in pbar:
            code,adj,sums = code.to(device),adj.to(device),sums.to(device)
            opt.zero_grad()
            logits = model(code,adj,sums)            # (B, T-1, V)
            mle_loss = compute_loss(logits,sums,pad) # LNULL

            # RAML: sample sequences and compute reward-weighted seq log-prob loss
            raml = raml_loss(model, code, adj, sums, sum_vocab, pad_idx=pad, device=device, lam=raml_lambda, n_samples=raml_samples)

            loss = theta1*mle_loss + theta2*raml
            loss.backward()
            opt.step()
            tot += loss.item()
            pbar.set_postfix({"loss": tot/ (pbar.n+1) })
        print(f"Epoch {ep}: avg loss = {tot/len(train_loader):.4f}")
    torch.save(model.state_dict(),"best_gascs_raml.pt")

# ----------------------------
# 5. Beam search for generation (keeps your previous function)
# ----------------------------
def beam_search(model,code_ids,adj,sum_vocab,beam=4,max_len=50,device="cpu"):
    model.eval()
    with torch.no_grad():
        mem=model.enc(code_ids.to(device),adj.to(device))
        beams=[([sum_vocab.stoi["<s>"]],0.0)]; finished=[]
        for _ in range(max_len):
            new=[]
            for seq,score in beams:
                if seq[-1]==sum_vocab.stoi["</s>"]:
                    finished.append((seq,score)); continue
                tgt=torch.tensor(seq).unsqueeze(0).to(device)
                logits=model.dec(tgt,mem)
                log_probs=F.log_softmax(logits[:,-1,:],-1).squeeze(0)
                topk=torch.topk(log_probs,beam)
                for idx,val in zip(topk.indices,topk.values):
                    new.append((seq+[idx.item()],score+val.item()))
            beams=sorted(new,key=lambda x:x[1],reverse=True)[:beam]
        finished+=beams
        best=max(finished,key=lambda x:x[1])[0]
        toks=[sum_vocab.itos[i] for i in best
              if i not in [sum_vocab.stoi[t] for t in ["<s>","</s>","<pad>"]]]
        return " ".join(toks)

# ----------------------------
# 6. Main (build vocabs, dataloaders, train)
# ----------------------------
def main():
    # small demo training data (use your full dataset for real experiments)
    train_pairs = [
        ("def add(a,b): return a+b", "add two numbers"),
        ("def subtract(a,b): return a-b", "subtract one number from another"),
        ("def multiply(a,b): return a*b", "multiply two numbers"),
        ("def divide(a,b): return a/b", "divide one number by another"),
        ("def square(x): return x*x", "square a number"),
        ("def cube(x): return x**3", "cube of a number"),
        ("def factorial(n): return 1 if n==0 else n*factorial(n-1)", "compute factorial"),
        ("def power(a,b): return a**b", "raise number to power"),
        ("def is_even(n): return n%2==0", "check if number is even"),
        ("def is_odd(n): return n%2!=0", "check if number is odd"),
        ("def maximum(a,b): return a if a>b else b", "find maximum of two numbers"),
        ("def minimum(a,b): return a if a<b else b", "find minimum of two numbers"),
        ("def absolute(x): return abs(x)", "get absolute value"),
        ("def reverse_string(s): return s[::-1]", "reverse a string"),
        ("def is_palindrome(s): return s==s[::-1]", "check palindrome string"),
        ("def to_lower(s): return s.lower()", "convert string to lowercase"),
        ("def to_upper(s): return s.upper()", "convert string to uppercase"),
        ("def capitalize_words(s): return s.title()", "capitalize each word in string"),
        ("def count_words(s): return len(s.split())", "count words in string"),
        ("def join_words(lst): return ' '.join(lst)", "join list of words into string"),
        ("def split_by_space(s): return s.split()", "split string by spaces"),
        ("def split_by_comma(s): return s.split(',')", "split string by comma"),
        ("def get_first(lst): return lst[0]", "get first element of list"),
        ("def get_last(lst): return lst[-1]", "get last element of list"),
        ("def list_sum(lst): return sum(lst)", "sum of list elements"),
        ("def list_max(lst): return max(lst)", "maximum element in list"),
        ("def list_min(lst): return min(lst)", "minimum element in list"),
        ("def unique_elements(lst): return list(set(lst))", "get unique elements of list"),
        ("def sort_list(lst): return sorted(lst)", "sort list ascending"),
        ("def sort_list_desc(lst): return sorted(lst, reverse=True)", "sort list descending"),
        ("def merge_dicts(d1,d2): return {**d1,**d2}", "merge two dictionaries"),
        ("def dict_keys(d): return list(d.keys())", "get dictionary keys"),
        ("def dict_values(d): return list(d.values())", "get dictionary values"),
        ("def dict_items(d): return list(d.items())", "get dictionary items"),
        ("def fibonacci(n): return n if n<=1 else fibonacci(n-1)+fibonacci(n-2)", "compute fibonacci number"),
        ("def gcd(a,b): return a if b==0 else gcd(b,a%b)", "compute greatest common divisor"),
        ("def lcm(a,b): return abs(a*b)//gcd(a,b)", "compute least common multiple"),
        ("def average(lst): return sum(lst)/len(lst)", "calculate average of list"),
        ("def flatten(list_of_lists): return [item for sub in list_of_lists for item in sub]", "flatten nested list"),
        ("def count_occurrences(lst,x): return lst.count(x)", "count occurrences of element"),
        ("def remove_duplicates(lst): return list(dict.fromkeys(lst))", "remove duplicates from list"),
        ("def reverse_list(lst): return lst[::-1]", "reverse a list"),
        ("def sum_even_numbers(lst): return sum(x for x in lst if x%2==0)", "sum even numbers in list"),
        ("def sum_odd_numbers(lst): return sum(x for x in lst if x%2!=0)", "sum odd numbers in list"),
        ("def filter_positive(lst): return [x for x in lst if x>0]", "filter positive numbers from list"),
        ("def filter_negative(lst): return [x for x in lst if x<0]", "filter negative numbers from list"),
        ("def square_list(lst): return [x*x for x in lst]", "square each element in list"),
        ("def cube_list(lst): return [x**3 for x in lst]", "cube each element in list"),
        ("def double_list(lst): return [2*x for x in lst]", "double each element in list"),
        ("def triple_list(lst): return [3*x for x in lst]", "triple each element in list"),
        ("def sum_matrix(matrix): return sum(sum(row) for row in matrix)", "sum all elements in matrix"),
        ("def transpose(matrix): return list(map(list, zip(*matrix)))", "transpose a matrix"),
        ("def flatten_matrix(matrix): return [x for row in matrix for x in row]", "flatten matrix to list"),
        ("def diagonal_sum(matrix): return sum(matrix[i][i] for i in range(len(matrix)))", "sum of diagonal elements"),
        ("def reverse_words(s): return ' '.join(s.split()[::-1])", "reverse words in string"),
        ("def word_lengths(s): return [len(w) for w in s.split()]", "get length of each word"),
        ("def longest_word(s): return max(s.split(), key=len)", "find longest word in string"),
        ("def shortest_word(s): return min(s.split(), key=len)", "find shortest word in string"),
        ("def starts_with(s,prefix): return s.startswith(prefix)", "check if string starts with prefix"),
        ("def ends_with(s,suffix): return s.endswith(suffix)", "check if string ends with suffix"),
        ("def replace_spaces(s): return s.replace(' ','_')", "replace spaces with underscore"),
        ("def remove_spaces(s): return s.replace(' ','')", "remove spaces from string"),
        ("def count_vowels(s): return sum(1 for c in s if c in 'aeiou')", "count vowels in string"),
        ("def count_consonants(s): return sum(1 for c in s if c.isalpha() and c not in 'aeiou')", "count consonants"),
        ("def is_digit(s): return s.isdigit()", "check if string is a digit"),
        ("def is_alpha(s): return s.isalpha()", "check if string is alphabetic"),
        ("def is_alnum(s): return s.isalnum()", "check if string is alphanumeric"),
        ("def reverse_tuple(t): return t[::-1]", "reverse a tuple"),
        ("def tuple_max(t): return max(t)", "find maximum in tuple"),
        ("def tuple_min(t): return min(t)", "find minimum in tuple"),
        ("def set_union(a,b): return a.union(b)", "union of two sets"),
        ("def set_intersection(a,b): return a.intersection(b)", "intersection of two sets"),
        ("def set_difference(a,b): return a.difference(b)", "difference of two sets"),
        ("def set_symmetric_difference(a,b): return a.symmetric_difference(b)", "symmetric difference of sets"),
        ("def count_chars(s): return {c:s.count(c) for c in set(s)}", "count character frequencies"),
        ("def unique_chars(s): return set(s)", "get unique characters in string"),
        ("def merge_lists(a,b): return a+b", "merge two lists"),
        ("def repeat_string(s,n): return s*n", "repeat string n times"),
        ("def repeat_list(lst,n): return lst*n", "repeat list n times"),
        ("def reverse_dict(d): return {v:k for k,v in d.items()}", "swap keys and values in dictionary"),
        ("def dict_length(d): return len(d)", "get dictionary size"),
        ("def clear_dict(d): d.clear(); return d", "clear dictionary"),
        ("def pop_dict(d,key): return d.pop(key)", "remove key from dictionary"),
        ("def update_dict(d,k,v): d[k]=v; return d", "update dictionary value"),
        ("def char_codes(s): return [ord(c) for c in s]", "get ASCII codes of characters"),
        ("def from_char_codes(lst): return ''.join(chr(x) for x in lst)", "build string from ASCII codes"),
        ("def reverse_words_in_sentence(s): return ' '.join(word[::-1] for word in s.split())", "reverse each word"),
        ("def is_anagram(a,b): return sorted(a)==sorted(b)", "check if two strings are anagrams"),
        ("def factorial_iter(n):\n    result=1\n    for i in range(1,n+1): result*=i\n    return result", "factorial using loop"),
        ("def sum_natural(n): return n*(n+1)//2", "sum of first n natural numbers"),
        ("def sum_squares(n): return sum(i*i for i in range(1,n+1))", "sum of squares up to n"),
        ("def sum_cubes(n): return sum(i**3 for i in range(1,n+1))", "sum of cubes up to n"),
        ("def nth_triangle(n): return n*(n+1)//2", "nth triangular number"),
        ("def harmonic_sum(n): return sum(1/i for i in range(1,n+1))", "harmonic series sum"),
        ("def is_prime(n): return all(n%i for i in range(2,int(n**0.5)+1)) and n>1", "check if number is prime"),
        ("def primes_up_to(n): return [x for x in range(2,n+1) if all(x%i for i in range(2,int(x**0.5)+1))]", "list primes up to n")
    ]
    train_ds=CodeSummaryDataset(train_pairs)
    # Build vocabularies
    code_tokens = [c for c,_,_ in train_ds]  # note dataset returns (code_nodes, edges, summary)
    # build code vocab from nodes lists
    code_vocab = Vocab(); code_vocab.build(code_tokens, min_freq=1)
    sum_vocab = Vocab(); sum_vocab.build([s for _,_,s in train_ds], min_freq=1)

    train_loader=DataLoader(train_ds,batch_size=2,shuffle=True,
        collate_fn=lambda b:collate_batch(b,code_vocab,sum_vocab))

    device="cuda" if torch.cuda.is_available() else "cpu"
    model=GA_SCS(len(code_vocab),len(sum_vocab), d=256)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)

    # Train with RAML
    print("Training with RAML (BLEU reward) ...")
    train_model_raml(model,train_loader,opt,sum_vocab,epochs=5,device=device,
                     theta1=0.1,theta2=0.9, raml_lambda=0.85, raml_samples=2)

    # ---- Test ----
    code="def power(a,b): return a**b"
    nodes,edges=build_graph_from_python_ast(code)
    ids=torch.tensor([code_vocab.encode(nodes)])
    L=ids.size(1); adj=torch.zeros(1,L,L)
    for u,v in edges:
        if u<L and v<L: adj[0,u,v]=1

    model.load_state_dict(torch.load("best_gascs_raml.pt",map_location=device))
    summary=beam_search(model,ids,adj,sum_vocab,device=device)
    print("Generated summary:",summary)

if __name__=="__main__":
    main()