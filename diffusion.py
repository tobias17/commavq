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
MODEL_FILENAME = "commavq_diffusion_{0}.safetensor"
ITER_FOLDER    = "iteration_{0}"
IMG_FILENAME   = "{1}_decoded_{0}.png"
LOSS_FILENAME  = "loss.png"
ACCS_FILENAME  = "acc.png"
LOAD_COUNT     = 20

ITERATIONS  = 100000
SAVE_EVERY  = 4000
TEST_EVERY  = 100
TEST_COUNT  = 4
LR = 2**-11
BS = 128

DIM = 384
N_HEADS = 6
D_HEADS = 64
VOCAB_SIZE = 1024
CTX_LAYERS = 2
DEN_LAYERS = 2
TIMESTEPS  = 1000

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

def timestep_embedding(timesteps, dim, max_period=10000):
  half = dim // 2
  freqs = (-math.log(max_period) * Tensor.arange(half) / half).exp().reshape(1,-1)
  args = timesteps * freqs
  return Tensor.cat(args.cos(), args.sin()).reshape(timesteps.shape[0], -1)

class TimeFusionBlock:
   def __init__(self, dim, emb_dim, dropout=0.1):
      self.in_layers = [
         Tensor.silu,
         Linear(dim, emb_dim),
      ]
      self.time_emb_layers = [
         Tensor.silu,
         Linear(dim, emb_dim),
      ]
      self.pos_emb_layers = [
         Tensor.silu,
         Linear(dim, emb_dim),
      ]
      self.out_layers = [
         Tensor.silu,
         Linear(emb_dim, dim),
      ]
      self.dropout = dropout
   
   def __call__(self, x:Tensor, time_emb:Tensor, pos_emb:Tensor) -> Tensor:
      h_in   = x.sequential(self.in_layers).dropout(self.dropout)
      h_time = time_emb.sequential(self.time_emb_layers).dropout(self.dropout)
      h_pos  = pos_emb.sequential(self.pos_emb_layers).dropout(self.dropout)
      h = (h_in + h_time + h_pos).sequential(self.out_layers)
      return x + h

class Transformer:
   def __init__(self, vocab_size, ctx_size, ctx_layers, den_size, den_layers, dim, n_heads, d_heads):
      self.time_embed = [
         Linear(320, dim),
         Tensor.silu,
         Linear(dim, dim),
      ]

      self.ctx_tok_embed = Embedding(vocab_size, dim)
      self.ctx_pos_embed = Embedding(ctx_size,   dim)
      self.ctx_layers = []
      for _ in range(ctx_layers):
         self.ctx_layers.append(BasicTransformerBlock(dim, dim, n_heads, d_heads))

      self.den_tok_embed = [
         Embedding(vocab_size, dim),
         LayerNorm(dim),
      ]
      self.den_pos_embed = Embedding(den_size, dim)
      self.den_layers = []
      for _ in range(den_layers):
         self.den_layers.append(TimeFusionBlock(dim, dim*2))
         self.den_layers.append(BasicTransformerBlock(dim, dim, n_heads, d_heads))

   def make_context_from(self, x:Tensor) -> Tensor:
      tok_emb = self.ctx_tok_embed(x)
      pos_emb = self.ctx_pos_embed(Tensor.arange(0, x.shape[-1], requires_grad=False).reshape([1]*(len(x.shape)-1)+[x.shape[-1]]))
      return (tok_emb + pos_emb).sequential(self.ctx_layers)
   
   def __call__(self, x:Tensor, timestep:Tensor, context:Tensor):
      B,T,C = x.shape
      t_emb = timestep_embedding(timestep, 320).sequential(self.time_embed).reshape(B,1,C)
      p_emb = self.den_pos_embed(Tensor.arange(0, T, requires_grad=False).reshape((1,T)))
      for layer in self.den_layers:
         if   isinstance(layer, TimeFusionBlock):       x = layer(x, t_emb, p_emb)
         elif isinstance(layer, BasicTransformerBlock): x = layer(x, context)
         else: raise ValueError(f"Found unknown denoising layer {type(layer)}")
      return x

MIN_ALPHA = 0.00046
MAX_ALPHA = 0.99915
def make_alphas_cumprod():
   alphas_cumprod = np.zeros(shape=(TIMESTEPS,))
   for i in range(TIMESTEPS): alphas_cumprod[i] = MIN_ALPHA + (MAX_ALPHA - MIN_ALPHA) * ((TIMESTEPS - i - 1) / (TIMESTEPS - 1))
   return alphas_cumprod

class CommaVQDiffuser:
   def __init__(self):
      self.alphas_cumprod = Tensor(make_alphas_cumprod(), dtype=dtypes.float32).realize()
      self.model = Transformer(VOCAB_SIZE, CTX_SIZE, CTX_LAYERS, FRAME_SIZE, DEN_LAYERS, DIM, N_HEADS, D_HEADS)
      self.decoder = Linear(DIM, VOCAB_SIZE)
      self.make_context_from = self.model.make_context_from

   def make_latent_from(self, x:Tensor) -> Tensor:
      return x.sequential(self.model.den_tok_embed)

   def get_x_prev_and_pred_x0(self, x, e_t, a_t, a_prev):
      sigma_t = 0
      sqrt_one_minus_at = (1-a_t).sqrt()
      pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()

      # direction pointing to x_t
      dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
      x_prev = a_prev.sqrt() * pred_x0 + dir_xt

      return x_prev, pred_x0

   def __call__(self, latent, context, timestep, alphas, alphas_prev):
      e_t = self.model(latent, timestep, context)
      x_prev, _ = self.get_x_prev_and_pred_x0(latent, e_t, alphas, alphas_prev)
      return x_prev

# def test(steps=20, model=None, save_path=None, baseline_path=None, ret_loss=False):
#    if model is None:
#       model = CommaVQDiffuser()

#    dataset = load_dataset("commaai/commavq", split=f'40[:1]')
#    path = dataset[0,0]['path']
#    data = np.load(path[0])
#    context = model.make_context_from(Tensor(data[:CTX_FRAMES].reshape(1,-1)))
#    target = data[CTX_FRAMES:CTX_FRAMES+1]

#    timesteps = list(range(1, 1000, 1000//steps))
#    alphas = model.alphas_cumprod[Tensor(timesteps)]
#    alphas_prev = Tensor([1.0]).cat(alphas[:-1])

#    latent = Tensor.randn(1, FRAME_SIZE, DIM)
#    for index, timestep in list(enumerate(timesteps))[::-1]:
#       # t.set_description(f"{index:3d} {timestep:3d}")
#       tid = Tensor([index])
#       latent = model(latent, context, Tensor([timestep]), alphas[tid], alphas_prev[tid])
#    output = model.decoder(latent)
#    tokens_am = output.argmax(axis=-1)
#    acc = (tokens_am == target).mean().numpy().item()
#    tokens = tokens_am.cast(dtypes.int64).numpy()
#    # del context
#    # del alphas
#    # del alphas_prev
#    # del latent
#    # del tid
#    # del output
#    # if ret_loss:
#    #    losses, accs = [], []
#    #    for i in trange(TEST_COUNT):
#    #       index = np.random.randint(data.shape[0] - CTX_FRAMES - 1)
#    #       l_path = dataset[i,0]['path']
#    #       l_data = np.load(l_path[0])

#    #       prev_frames  = Tensor(l_data[index:index+CTX_FRAMES].reshape(1,-1), requires_grad=False)
#    #       target_frame = Tensor(l_data[index+CTX_FRAMES].reshape(1,-1))
#    #       loss, output = create_loss(model, prev_frames, target_frame)
#    #       acc = (output.argmax(axis=-1) == target_frame).mean().numpy().item()
#    #       losses.append(loss)
#    #       accs.append(acc)
#    #    loss = sum(losses) / len(losses)
#    #    acc  = sum(acc)    / len(acc)

#    options  = ort.SessionOptions()
#    # options.log_severity_level = 3
#    provider = 'CPUExecutionProvider' # 'CUDAExecutionProvider'
#    session  = ort.InferenceSession(f'gpt2m/decoder.onnx', options, [provider])
#    if baseline_path:
#       outputs = session.run(None, {'encoding_indices': target.astype(np.int64)})
#       img = transpose_and_clip(outputs[0].reshape(outputs[0].shape[1:]))
#       plt.clf()
#       plt.imshow(img)
#       plt.savefig(baseline_path)

#    outputs = session.run(None, {'encoding_indices': tokens[0].reshape(1,8,16)})
#    img = transpose_and_clip(outputs[0].reshape(outputs[0].shape[1:]))
#    plt.clf()
#    plt.imshow(img)
#    if save_path: plt.savefig(save_path)
#    else:         plt.show()

#    if ret_loss:
#       return acc




def create_loss(model:CommaVQDiffuser, prev_frames:Tensor, target_frame:Tensor, timesteps:Optional[Tensor]=None):
   context = model.make_context_from(prev_frames.detach())
   if timesteps is None:
      timesteps = Tensor(np.random.randint(0, TIMESTEPS, size=(BS,)), dtype=dtypes.float32, requires_grad=False)

   x_0 = model.make_latent_from(target_frame.detach())
   x_0 = x_0.expand((timesteps.shape[0],*x_0.shape[1:]))
   B,T,C = x_0.shape
   alphas_t = Tensor.full((B,1,1), MIN_ALPHA)
   alphas_prev = model.alphas_cumprod[timesteps].reshape((B,1,1))
   x_t = (x_0 * alphas_prev) + (Tensor.randn(*x_0.shape) * (1 - alphas_prev))

   e_t = model.model(x_t, timesteps.reshape(-1,1), context)
   loss_1 = (e_t - (x_t - x_0)).pow(2).sum().sqrt()

   _, pred_x0 = model.get_x_prev_and_pred_x0(x_t, e_t, alphas_t, alphas_prev)
   output = model.decoder(pred_x0).log_softmax()
   loss_2 = output.sparse_categorical_crossentropy(target_frame)

   return loss_1 + loss_2, output


test_timesteps = [int((i+1)/TEST_COUNT*TIMESTEPS)-1 for i in range(TEST_COUNT)]
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

def save_train_snapshot(step, outputs, baseline):
   _save_decoded_image(baseline, os.path.join(SAVE_FOLDER, ITER_FOLDER.format(step), IMG_FILENAME.format('0', 'train')))
   for i in range(TEST_COUNT):
      _save_decoded_image(outputs[i], os.path.join(SAVE_FOLDER, ITER_FOLDER.format(step), IMG_FILENAME.format(test_timesteps[i], 'train')))

def save_test_snapshot(step, outputs, baseline, train_losses, test_losses, train_accs, test_accs):
   _save_decoded_image(baseline, os.path.join(SAVE_FOLDER, ITER_FOLDER.format(step), IMG_FILENAME.format('0', 'test')))
   for i in range(TEST_COUNT):
      _save_decoded_image(outputs[i], os.path.join(SAVE_FOLDER, ITER_FOLDER.format(step), IMG_FILENAME.format(test_timesteps[i], 'test')))
   
   plt.clf()
   for loss, label in [(train_losses, 'train'), (test_losses, 'test')]:
      x, y = np.arange(TEST_EVERY, len(loss)*TEST_EVERY+1, TEST_EVERY), loss
      plt.plot(x, y, label=label)
   plt.ylim(0, None)
   plt.legend()
   plt.savefig(os.path.join(SAVE_FOLDER, ITER_FOLDER.format(step), LOSS_FILENAME))

   plt.clf()
   for accs, label in [(train_accs, 'train_{0}'), (test_accs, 'test_{0}')]:
      for i in range(TEST_COUNT):
         acc = accs[i]
         x, y = np.arange(TEST_EVERY, len(acc)*TEST_EVERY+1, TEST_EVERY), acc
         plt.plot(x, y, label=label.format(test_timesteps[i]))
   plt.ylim(0.0, 1.0)
   plt.legend()
   plt.savefig(os.path.join(SAVE_FOLDER, ITER_FOLDER.format(step), ACCS_FILENAME))



def train():
   np.random.seed(1337)
   model = CommaVQDiffuser()
   sd = get_state_dict(model)
   opt = Adam(get_parameters(model), LR)
   
   s_time = time.time()
   dataset = {}
   for i in list(range(LOAD_COUNT)) + ['40[:5]']:
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
   test_datas = [np.load(dataset['40'][1,0]['path'][0])]
   print(f"loaded dataset in {time.time() - s_time:.1f} seconds")

   s_time = time.time()
   step, test_index = 0, 0
   train_loss, test_loss = [], []
   train_accs, test_accs = [[] for _ in range(TEST_COUNT)], [[] for _ in range(TEST_COUNT)]
   while step < ITERATIONS:
      if test_index < 2:
         data = train_datas[step % len(train_datas)]
         index = np.random.randint(data.shape[0] - CTX_FRAMES - 1)
      else:
         data = test_datas[0]
         index = 0

      prev_frames  = Tensor(data[index:index+CTX_FRAMES].reshape(1,-1), dtype=dtypes.float32, requires_grad=False)
      target_frame = Tensor(data[index+CTX_FRAMES].reshape(1,-1), dtype=dtypes.float32)
      timesteps = None if test_index == 0 else Tensor(test_timesteps, dtype=dtypes.float32, requires_grad=False)
      loss, output = create_loss(model, prev_frames, target_frame, timesteps)

      if test_index == 0:
         loss.realize()
         opt.zero_grad()
         loss.backward()
         opt.step()
      
      if test_index > 0:
         loss_list, accs_list = (train_loss, train_accs) if test_index == 1 else (test_loss, test_accs)
         loss_list.append(loss.numpy().item())
         accs = [(output.argmax(axis=-1) == target_frame)[i].mean().numpy() for i in range(TEST_COUNT)]
         for i in range(TEST_COUNT):
            accs_list[i].append(accs[i])
         if test_index == 1:
            save_train_snapshot(step+1, output.argmax(axis=-1), target_frame)

      if (step+1) % TEST_EVERY == 0:
         test_index += 1
         if test_index > 2:
            print(f"Step {str(step+1): >4} | Train Loss: {train_loss[-1]:.4f} | Train Accuracy: {100.0*(sum(a[-1] for a in train_accs)/float(TEST_COUNT)):.2f}% | Test Loss: {test_loss[-1]:.4f} | Test Accuracy: {100.0*(sum(a[-1] for a in test_accs)/float(TEST_COUNT)):.2f}% | {(time.time() - s_time) / float(TEST_EVERY):.2f} sec/iter")
            save_test_snapshot(step+1, output.argmax(axis=-1), target_frame, train_loss, test_loss, train_accs, test_accs)
            test_index = 0
            step += 1
            s_time = time.time()
      else:
         step += 1

      if (step+1) % SAVE_EVERY == 0:
         if not os.path.exists(SAVE_FOLDER):
            os.makedirs(SAVE_FOLDER)
         safe_save(get_state_dict(model), os.path.join(SAVE_FOLDER, MODEL_FILENAME.format(step)))




def generate(gen_count=100, start_t=999, end_t=0):
   Tensor.no_grad = True

   np.random.seed(1337)
   model = CommaVQDiffuser()
   load_state_dict(model, safe_load(os.path.join("weights", "2023_11_12_20_22_53_147121", MODEL_FILENAME.format('39999'))))

   data = np.load(load_dataset("commaai/commavq", split='40[:2]')[1,0]['path'][0])
   prev_frames = np.zeros((CTX_FRAMES+gen_count+1,FRAME_SIZE))
   prev_frames[:CTX_FRAMES] = data[:CTX_FRAMES].reshape((CTX_FRAMES,-1))

   times = []
   for i in range(gen_count):
      s_time = time.time()

      prev_frames_t = Tensor(prev_frames[i:i+CTX_FRAMES].reshape((1,-1)), dtype=dtypes.float32)
      target_frame  = Tensor(data[CTX_FRAMES:CTX_FRAMES+1].reshape((1,-1)))
      context = model.make_context_from(prev_frames_t.detach())

      timesteps = Tensor([TIMESTEPS-1])
      a_t = model.alphas_cumprod[Tensor([end_t])].reshape((1,1,1))
      a_p = model.alphas_cumprod[Tensor([start_t])].reshape((1,1,1))
      x_0 = model.make_latent_from(target_frame.detach())
      x_t = (x_0 * a_p) + Tensor.randn(1,FRAME_SIZE,DIM) * (1 - a_p)
      e_t = model.model(x_t, timesteps.reshape((1,1)), context)
      _, pred_x0 = model.get_x_prev_and_pred_x0(x_t, e_t, a_t, a_p)
      output = model.decoder(pred_x0).log_softmax().argmax(axis=-1)

      prev_frames[CTX_FRAMES+i+1,:] = output.numpy()[0,:]

      print(x_0.abs().sum().numpy())
      print((x_0 * a_p).abs().sum().numpy())

      img = _to_image(output[0])
      # img = _to_image(Tensor(prev_frames[CTX_FRAMES-1:CTX_FRAMES]))

      plt.clf()
      plt.imshow(img)
      plt.show()

      # cv2.imshow('frames', img)

      times.append(time.time() - s_time)
      while len(times) > 20:
         times.pop(0)
      fps = 1.0 / (sum(times)/len(times))
      print( f"{fps/20.0:.2f}x realtime (avg {fps:.2f} fps)" )




if __name__ == "__main__":
   generate()

