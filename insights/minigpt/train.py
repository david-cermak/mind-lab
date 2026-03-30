"""
Train minigpt on input.txt and export weights to data.c / data.h for embedded inference.
Run: python train.py
Output: data.c, data.h
"""

import os
import math
import random

random.seed(42)

# Load dataset
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')

docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# Tokenizer
uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
print(f"vocab size: {vocab_size}")

# Autograd
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

# Config (must match infer.c)
n_layer = 1
n_embd = 16
block_size = 16
n_head = 4
head_dim = n_embd // n_head

matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"num params: {len(params)}")

def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    for li in range(n_layer):
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict['lm_head'])
    return logits

# Training
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params)
v = [0.0] * len(params)
num_steps = 1000

for step in range(num_steps):
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)
    loss = (1 / n) * sum(losses)

    loss.backward()

    lr_t = learning_rate * (1 - step / num_steps)
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0

    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}", end='\r')

print("\n--- exporting to data.c / data.h ---")

def emit_array(name, data, rows, cols):
    """Emit C array as row-major float values."""
    lines = ["const float %s[%d][%d] = {" % (name, rows, cols)]
    for r in range(rows):
        row_vals = ["%.8f" % data[r][c] for c in range(cols)]
        lines.append("    {" + ", ".join(row_vals) + "},")
    lines.append("};")
    return "\n".join(lines)

def emit_3d_array(name, data, d0, d1, d2):
    """Emit C 3D array [d0][d1][d2]."""
    lines = ["const float %s[%d][%d][%d] = {" % (name, d0, d1, d2)]
    for i in range(d0):
        lines.append("    {")
        for j in range(d1):
            row_vals = ["%.8f" % data[i][j][k] for k in range(d2)]
            lines.append("        {" + ", ".join(row_vals) + "},")
        lines.append("    },")
    lines.append("};")
    return "\n".join(lines)

# Build data.h
data_h = """/* Auto-generated by train.py. Do not edit. */
#ifndef MINIGPT_DATA_H
#define MINIGPT_DATA_H

#define N_LAYER     %(n_layer)d
#define N_EMBD      %(n_embd)d
#define BLOCK_SIZE  %(block_size)d
#define N_HEAD      %(n_head)d
#define HEAD_DIM    (N_EMBD / N_HEAD)
#define VOCAB_SIZE  %(vocab_size)d
#define BOS_TOKEN   %(BOS)d

extern const char uchars[VOCAB_SIZE];

extern const float wte[VOCAB_SIZE][N_EMBD];
extern const float wpe[BLOCK_SIZE][N_EMBD];
extern const float lm_head[VOCAB_SIZE][N_EMBD];
extern const float attn_wq[N_LAYER][N_EMBD][N_EMBD];
extern const float attn_wk[N_LAYER][N_EMBD][N_EMBD];
extern const float attn_wv[N_LAYER][N_EMBD][N_EMBD];
extern const float attn_wo[N_LAYER][N_EMBD][N_EMBD];
extern const float mlp_fc1[N_LAYER][4*N_EMBD][N_EMBD];
extern const float mlp_fc2[N_LAYER][N_EMBD][4*N_EMBD];

#endif
""" % {
    'n_layer': n_layer, 'n_embd': n_embd, 'block_size': block_size,
    'n_head': n_head, 'vocab_size': vocab_size, 'BOS': BOS
}

# Build data.c
def mat_to_list(mat):
    return [[p.data for p in row] for row in mat]

data_c = """/* Auto-generated by train.py. Do not edit. */
#include "data.h"

"""

# uchars: index 0..vocab_size-2 = character for that token. Index BOS = vocab_size-1, store 0 (not used for decoding)
uchars_arr = list(uchars) + ['\0']  # vocab_size elements
data_c += "const char uchars[VOCAB_SIZE] = {\n    "
data_c += ", ".join(str(ord(c)) if c != '\0' else "0" for c in uchars_arr)
data_c += "\n};\n\n"

# Weights
data_c += emit_array("wte", mat_to_list(state_dict['wte']), vocab_size, n_embd) + "\n\n"
data_c += emit_array("wpe", mat_to_list(state_dict['wpe']), block_size, n_embd) + "\n\n"
data_c += emit_array("lm_head", mat_to_list(state_dict['lm_head']), vocab_size, n_embd) + "\n\n"

# Layer weights as 3D arrays
attn_wq_data = [mat_to_list(state_dict[f'layer{li}.attn_wq']) for li in range(n_layer)]
attn_wk_data = [mat_to_list(state_dict[f'layer{li}.attn_wk']) for li in range(n_layer)]
attn_wv_data = [mat_to_list(state_dict[f'layer{li}.attn_wv']) for li in range(n_layer)]
attn_wo_data = [mat_to_list(state_dict[f'layer{li}.attn_wo']) for li in range(n_layer)]
mlp_fc1_data = [mat_to_list(state_dict[f'layer{li}.mlp_fc1']) for li in range(n_layer)]
mlp_fc2_data = [mat_to_list(state_dict[f'layer{li}.mlp_fc2']) for li in range(n_layer)]

data_c += emit_3d_array("attn_wq", attn_wq_data, n_layer, n_embd, n_embd) + "\n\n"
data_c += emit_3d_array("attn_wk", attn_wk_data, n_layer, n_embd, n_embd) + "\n\n"
data_c += emit_3d_array("attn_wv", attn_wv_data, n_layer, n_embd, n_embd) + "\n\n"
data_c += emit_3d_array("attn_wo", attn_wo_data, n_layer, n_embd, n_embd) + "\n\n"
data_c += emit_3d_array("mlp_fc1", mlp_fc1_data, n_layer, 4 * n_embd, n_embd) + "\n\n"
data_c += emit_3d_array("mlp_fc2", mlp_fc2_data, n_layer, n_embd, 4 * n_embd) + "\n\n"

with open('data.h', 'w') as f:
    f.write(data_h)
with open('data.c', 'w') as f:
    f.write(data_c)

print("Wrote data.h and data.c")
