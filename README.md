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


---

## 📂 Project Structure
```
graphcs.py              # Main training & inference script
README.md               # Project documentation
```

---

## 📊 Example Results
On the included toy dataset, the model successfully learns to generate functional summaries such as:

- `def add(a,b): return a+b` → **"add two numbers"**  
- `def factorial(n): return 1 if n==0 else n*factorial(n-1)` → **"compute factorial"**  
- `def reverse_string(s): return s[::-1]` → **"reverse a string"**

---


# Testing

---

## 1. Prerequisites

Make sure you have Python 3.8+ installed and the following packages:

```bash
pip install torch torchvision torchaudio
pip install nltk tqdm rouge-score
```

Download NLTK data if not already installed:

```python
import nltk
nltk.download('punkt')
```

---

## 2. Files

- `test_graphcs.py` : Test script for evaluating the model.
- `graphcs.pt` : Trained model checkpoint.
- `your_training_module.py` : Contains GraphCS model, dataset, collate functions, etc.

---

## 3. Running the Test

Run the test script using:

```bash
python test_graphcs.py --ckpt graphcs.pt --beam 4 --device cpu
```

or if you have a GPU:

```bash
python test_graphcs.py --ckpt graphcs.pt --beam 4 --device cuda
```

**Arguments:**

- `--ckpt` : Path to the trained model checkpoint.
- `--beam` : Beam size for beam search (default = 4).
- `--device` : Device to run the test on (`cpu` or `cuda`).

---

## 4. Test Output

The script will print for each sample:

- Reference summary (`Ref`)
- Predicted summary (`Pred`)

At the end, it prints average metrics over all test samples:

```
Evaluated <num_samples> samples
BLEU (avg):   XX.XX
METEOR (avg): XX.XX
ROUGE-L (avg): XX.XX
```

---

## 5. Notes

- Replace the test dataset inside `test_graphcs.py` with your own code-summary pairs.
- Ensure the checkpoint `graphcs.pt` matches the vocabulary used in training.
- You can adjust `beam` size to improve output quality (larger beam → slower inference).

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


