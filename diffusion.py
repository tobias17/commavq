from datasets import load_dataset # type: ignore
from tinygrad.tensor import Tensor # type: ignore
from tinygrad.nn import Linear, LayerNorm, Embedding # type: ignore
from tinygrad.helpers import dtypes, prod # type: ignore
from tinygrad.nn.state import get_parameters, safe_save, safe_load, get_state_dict, load_state_dict # type: ignore
from tinygrad.nn.optim import Adam # type: ignore
import onnxruntime as ort # type: ignore
import numpy as np
from datetime import datetime
import math
from typing import Optional
from tqdm import tqdm, trange # type: ignore
import os
import time
import matplotlib.pyplot as plt
import cv2

SAVE_FOLDER    = f"weights/{datetime.now()}".replace(" ", "_").replace(":", "_").replace("-", "_").replace(".", "_")
MODEL_FILENAME = "model_{0}.safetensor"
ITER_FOLDER    = "iteration_{0}"
IMG_FILENAME   = "{0}_{1}.png"
LOSS_FILENAME  = "graph_loss.png"
ACCS_FILENAME  = "graph_acc.png"
LOAD_COUNT     = 20

ITERATIONS  = 100000
SAVE_EVERY  = 4000
TEST_EVERY  = 100
TEST_COUNT  = 4
LR = 2**-12
BS = 5

DIM = 256
N_HEADS = 4
D_HEADS = 64
VOCAB_SIZE = 1024
CTX_LAYERS = 2
DEC_LAYERS = 2

FRAME_SIZE = 8*16
CTX_FRAMES = 20
CTX_SIZE   = CTX_FRAMES * FRAME_SIZE

class CrossAttention:
   def __init__(self, query_dim, context_dim, n_heads, d_head, dropout=0.1):
      self.to_q = Linear(query_dim, n_heads*d_head, bias=False)
      self.to_k = Linear(context_dim, n_heads*d_head, bias=False)
      self.to_v = Linear(context_dim, n_heads*d_head, bias=False)
      self.num_heads = n_heads
      self.head_size = d_head
      self.to_out = [Linear(n_heads*d_head, query_dim)]
      self.dropout = dropout

   def __call__(self, x, context=None):
      context = x if context is None else context
      q,k,v = self.to_q(x), self.to_k(context), self.to_v(context)
      q,k,v = [y.reshape(x.shape[0], -1, self.num_heads, self.head_size).transpose(1,2) for y in (q,k,v)]
      attention = Tensor.scaled_dot_product_attention(q, k, v).dropout(self.dropout).transpose(1,2)
      h_ = attention.reshape(shape=(x.shape[0], -1, self.num_heads * self.head_size))
      return h_.sequential(self.to_out)

class GEGLU:
   def __init__(self, dim_in, dim_out):
      self.proj = Linear(dim_in, dim_out * 2)
      self.dim_out = dim_out

   def __call__(self, x):
      x, gate = self.proj(x).chunk(2, dim=-1)
      return x * gate.gelu()

class FeedForward:
   def __init__(self, dim, mult=4):
      self.net = [
         GEGLU(dim, dim*mult),
         lambda x: x,  # needed for weights loading code to work
         Linear(dim*mult, dim)
      ]

   def __call__(self, x):
      return x.sequential(self.net)

class BasicTransformerBlock:
   def __init__(self, dim, context_dim, n_heads, d_head, dropout=0.1):
      self.attn1 = CrossAttention(dim, dim, n_heads, d_head)
      self.ff = FeedForward(dim)
      self.attn2 = CrossAttention(dim, context_dim, n_heads, d_head)
      self.norm1 = LayerNorm(dim)
      self.norm2 = LayerNorm(dim)
      self.norm3 = LayerNorm(dim)
      self.dropout = dropout

   def __call__(self, x, context=None):
      x = self.attn1(self.norm1(x)) + x
      x = self.attn2(self.norm2(x), context=context) + x
      x = self.ff(self.norm3(x)).dropout(self.dropout) + x
      return x

class CommaVQDiffuser:
   def __init__(self, vocab_size, ctx_size, ctx_layers, dec_size, dec_layers, dim, n_heads, d_heads):
      self.ctx_tok_embed = Embedding(vocab_size, dim)
      self.ctx_pos_embed = Embedding(ctx_size,   dim)
      self.ctx_layers = []
      for _ in range(ctx_layers):
         self.ctx_layers.append(BasicTransformerBlock(dim, dim, n_heads, d_heads))

      self.dec_size = dec_size
      self.dec_pos_embed = Embedding(dec_size, dim)
      self.dec_layers = []
      for _ in range(dec_layers):
         self.dec_layers.append(BasicTransformerBlock(dim, dim, n_heads, d_heads))
      self.class_head = Linear(dim, vocab_size)

   def make_context_from(self, x:Tensor) -> Tensor:
      tok_emb = self.ctx_tok_embed(x)
      pos_emb = self.ctx_pos_embed(Tensor.arange(0, x.shape[-1], requires_grad=False).reshape([1]*(len(x.shape)-1)+[x.shape[-1]]))
      return (tok_emb + pos_emb).sequential(self.ctx_layers)
   
   def __call__(self, context:Tensor):
      x = self.dec_pos_embed(Tensor.arange(0, self.dec_size, requires_grad=False).reshape((1,-1))).expand((context.shape[0],self.dec_size,-1))
      for layer in self.dec_layers:
         x = layer(x, context)
      return self.class_head(x).log_softmax()

MODEL_PARAMS = [VOCAB_SIZE, CTX_SIZE, CTX_LAYERS, FRAME_SIZE, DEC_LAYERS, DIM, N_HEADS, D_HEADS]




def _to_image(output):
   if not hasattr(_to_image, 'session'):
      options  = ort.SessionOptions()
      provider = 'CPUExecutionProvider' # 'CUDAExecutionProvider'
      _to_image.session = ort.InferenceSession(f'gpt2m/decoder.onnx', options, [provider])
   outputs = _to_image.session.run(None, {'encoding_indices': output.reshape(1,8,16).numpy().astype(np.int64)})
   img = transpose_and_clip(outputs[0].reshape(outputs[0].shape[1:]))
   return img

def transpose_and_clip(tensors):
  tensors = np.transpose(tensors, (1,2,0))
  tensors = np.clip((tensors + 1.0) * 127.5, 0, 255).astype(np.uint8)
  return tensors
def _save_decoded_image(output, filepath):
   if not os.path.exists(os.path.dirname(filepath)):
      os.makedirs(os.path.dirname(filepath))
   img = _to_image(output)
   plt.clf()
   plt.imshow(img)
   plt.savefig(filepath)

def save_test_snapshot(step, outputs, baselines, train_loss, test_loss, train_acc, test_acc):
   for i in range(TEST_COUNT):
      baseline_path = os.path.join(SAVE_FOLDER, IMG_FILENAME.format('test_baseline', i))
      if not os.path.exists(baseline_path):
         _save_decoded_image(baselines[i], baseline_path)
      _save_decoded_image(outputs[i], os.path.join(SAVE_FOLDER, ITER_FOLDER.format(step), IMG_FILENAME.format('decoded_test', i)))
   
   plt.clf()
   plt.plot(np.arange(0, len(train_loss)), train_loss, label='train')
   plt.plot(np.arange(TEST_EVERY, len(test_loss)*TEST_EVERY+1, TEST_EVERY), test_loss, label='test')
   plt.ylim(0, None)
   plt.legend()
   plt.savefig(os.path.join(SAVE_FOLDER, LOSS_FILENAME))
   
   plt.clf()
   plt.plot(np.arange(0, len(train_acc)), train_acc, label='train')
   plt.plot(np.arange(TEST_EVERY, len(test_acc)*TEST_EVERY+1, TEST_EVERY), test_acc, label='test')
   plt.ylim(0.0, 1.0)
   plt.legend()
   plt.savefig(os.path.join(SAVE_FOLDER, ACCS_FILENAME))


def train():
   np.random.seed(1337)
   model = CommaVQDiffuser(*MODEL_PARAMS)
   sd = get_state_dict(model)
   opt = Adam(get_parameters(model), LR)
   
   s_time = time.time()
   dataset = {}
   for i in list(range(LOAD_COUNT)) + ['40[:10]']:
      dataset[str(i).split("[")[0]] = load_dataset("commaai/commavq", split=str(i))
   train_datas = []
   for split in dataset:
      if split in ['27', '38', '40']:
         continue
      for i in range(dataset[split].shape[0]):
         try:
            train_datas.append(np.load(dataset[split][i,0]['path'][0]))
         except Exception as ex:
            print(f"encountered error at dataset[{split}][{i}]: {ex}")
   test_datas = [np.load(dataset['40'][i,0]['path'][0]) for i in range(TEST_COUNT)]
   print(f"loaded dataset in {time.time() - s_time:.1f} seconds")

   s_time = time.time()
   step, is_test = 0, False
   train_loss, test_loss = [], []
   train_acc,  test_acc  = [], []
   while step < ITERATIONS:
      if is_test:
         datas = test_datas
         index = 0
         size = TEST_COUNT
      else:
         datas = [train_datas[i % len(train_datas)] for i in range(step*BS, (step+1)*BS)]
         index = np.random.randint(datas[0].shape[0] - CTX_FRAMES - 1)
         size = BS

      prev_frames = np.array([d[index:index+CTX_FRAMES] for d in datas]).reshape(size,-1)
      prev_frames = Tensor(prev_frames, dtype=dtypes.float32, requires_grad=False)
      target_frames = np.array([d[index+CTX_FRAMES] for d in datas]).reshape(size,-1)
      target_frames = Tensor(target_frames, dtype=dtypes.float32)

      context = model.make_context_from(prev_frames)
      output  = model(context)
      loss    = output.sparse_categorical_crossentropy(target_frames)

      if is_test:
         test_loss.append(loss.numpy())
         test_acc.append((output.argmax(axis=-1) == target_frames).mean().numpy())
      else:
         loss.realize()
         opt.zero_grad()
         loss.backward()
         opt.step()

         train_loss.append(loss.numpy())
         train_acc.append((output.argmax(axis=-1) == target_frames).mean().numpy())
      

      if (step+1) % TEST_EVERY == 0:
         if is_test:
            step += 1
            print(f"Step {str(step): >5} | Train Loss: {sum(train_loss[-TEST_EVERY:])/TEST_EVERY:.4f} | Train Accuracy: {100.0*sum(train_acc[-TEST_EVERY:])/TEST_EVERY:.2f}% | Test Loss: {test_loss[-1]:.4f} | Test Accuracy: {100.0*test_acc[-1]:.2f}% | {(time.time() - s_time) / float(TEST_EVERY):.2f} sec/iter")
            s_time = time.time()
            save_test_snapshot(step, output.argmax(axis=-1), target_frames, train_loss, test_loss, train_acc, test_acc)
         is_test = not is_test
      else:
         step += 1

      if step % SAVE_EVERY == 0:
         if not os.path.exists(SAVE_FOLDER):
            os.makedirs(SAVE_FOLDER)
         safe_save(get_state_dict(model), os.path.join(SAVE_FOLDER, MODEL_FILENAME.format(step)))




def generate(gen_count=10):
   Tensor.no_grad = True

   np.random.seed(1337)
   model = CommaVQDiffuser(*MODEL_PARAMS)
   load_state_dict(model, safe_load(os.path.join("weights", "2023_11_13_20_42_41_823383", MODEL_FILENAME.format("40000"))))

   data = np.load(load_dataset("commaai/commavq", split='40[:2]')[1,0]['path'][0])
   prev_frames = np.zeros((CTX_FRAMES+gen_count,FRAME_SIZE))
   prev_frames[:CTX_FRAMES] = data[:CTX_FRAMES].reshape((CTX_FRAMES,-1))

   times = []
   for i in range(gen_count):
      s_time = time.time()

      prev_frames_t = Tensor(prev_frames[i:i+CTX_FRAMES].reshape((1,-1)), dtype=dtypes.float32)
      context = model.make_context_from(prev_frames_t.detach())
      output = model(context).argmax(axis=-1)

      prev_frames[CTX_FRAMES+i,:] = output.numpy()[0,:]
      img = _to_image(output[0])

      # plt.clf()
      # plt.imshow(img)
      # plt.savefig("outputs/frame.png")

      cv2.imshow('frames', img[...,::-1])
      cv2.waitKey()

      times.append(time.time() - s_time)
      while len(times) > 4:
         times.pop(0)
      fps = 1.0 / (sum(times)/len(times))
      print( f"{fps/20.0:.2f}x realtime (avg {fps:.2f} fps)" )

   # write_video(prev_frames.reshape(CTX_FRAMES+gen_count,8,16), "outputs/video.mp4")


if __name__ == "__main__":
   generate()

