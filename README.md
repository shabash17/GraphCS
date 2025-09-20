# GraphCS with Multi-Head Structure-Informed Self-Attention

This repository implements **GraphCS** (Graph Attention-Network based Source Code Summarization).

## ✨ Features
- **AST-based graph construction** for source code.
- **Graph Attention Networks (GAT)** to encode structural information.
- **Multi-Head Structure-Informed Self-Attention** to fuse sequential and structural code representations.
- **Transformer-based decoder** with causal masks for autoregressive summarization.
- **Reward Augmented Maximum Likelihood (RAML)** with BLEU rewards for training.
- **Beam search decoding** for high-quality summaries.
- Includes a **large toy dataset of Python functions** with natural language summaries.

---

## 📦 Installation
Clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/graphcs-raml.git
cd graphcs-raml
pip install -r requirements.txt
```

Or, if running in Google Colab:

```python
!pip install -q nltk tqdm
```

---

## 🚀 Usage
### Training
Run the script to train the model. This code trains the model for 3 epochs using the default parameters. You can optimize the model by increasing the number of epochs and fine-tuning the parameters.

```bash
python graphcs.py
```

Training uses:
- **MLE loss** (maximum likelihood estimation)  
- **RAML loss** (BLEU-based reward optimization)

By default, the model trains for **3 epochs** on the toy dataset.

### Testing
After training, the script generates summaries using **beam search**:

---

## 📂 Project Structure
```
graphcs.py              # Main training & inference script
README.md               # Project documentation
requirements.txt        # Dependencies
```

---

## 📊 Example Results
On the included toy dataset, the model successfully learns to generate functional summaries such as:

- `def add(a,b): return a+b` → **"add two numbers"**  
- `def factorial(n): return 1 if n==0 else n*factorial(n-1)` → **"compute factorial"**  
- `def reverse_string(s): return s[::-1]` → **"reverse a string"**

---

## 🔧 Requirements
- Python 3.8+
- PyTorch ≥ 1.10
- NLTK
- tqdm

Install with:

```bash
pip install torch nltk tqdm
```

## Running on GPU/CPU

- If CUDA is available, the script will automatically use GPU.  
- Otherwise, training falls back to CPU.  
- Multi-GPU training is not yet enabled.  

---

## Acknowledgement

This code is inspired by ideas from:  
- [Graph Attention Networks (GAT)]  
- [Transformer Decoder (Vaswani et al.)]  
- [RAML for Reinforcement Learning]  
- [DrQA, OpenNMT projects]  

We extend these ideas with **structure-informed self-attention** for source code understanding.  


