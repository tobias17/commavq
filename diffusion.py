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

ITERATIONS  = 200000
SAVE_EVERY  = 2000
TEST_EVERY  = 100
TEST_COUNT  = 1
LR = 2**-17
BS = 1

DIM     = 256
FF_DIM  = DIM*2
N_HEADS = 8
DEC_LAYERS = 4

BOS_TOKEN  = 1024
VOCAB_SIZE = 1024 + 1
FRAME_SIZE = 8*16 + 1
CTX_FRAMES = 20
MAX_CTX_SIZE = CTX_FRAMES * FRAME_SIZE

class TransformerBlock:
   def __init__(self, embed_dim, num_heads, ff_dim, act=lambda x: x.relu(), dropout=0.1):
      assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

      self.num_heads = num_heads
      self.head_size = embed_dim // num_heads
      self.act = act
      self.dropout = dropout

      self.query = (Tensor.scaled_uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))
      self.key   = (Tensor.scaled_uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))
      self.value = (Tensor.scaled_uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))

      self.out = (Tensor.scaled_uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))

      self.ff1 = (Tensor.scaled_uniform(embed_dim, ff_dim), Tensor.zeros(ff_dim))
      self.ff2 = (Tensor.scaled_uniform(ff_dim, embed_dim), Tensor.zeros(embed_dim))

      self.ln1 = (Tensor.ones(embed_dim), Tensor.zeros(embed_dim))
      self.ln2 = (Tensor.ones(embed_dim), Tensor.zeros(embed_dim))

   def attn(self, x):
      # x: (bs, time, embed_dim) -> (bs, time, embed_dim)
      query, key, value = [x.linear(*y).reshape(shape=(x.shape[0], -1, self.num_heads, self.head_size)).transpose(1,2) for y in [self.query, self.key, self.value]]
      attention = Tensor.scaled_dot_product_attention(query, key, value, is_causal=True).transpose(1,2)
      return attention.reshape(shape=(x.shape[0], -1, self.num_heads * self.head_size)).linear(*self.out)

   def __call__(self, x):
      x = x + self.attn(x).dropout(self.dropout)
      x = x.layernorm().linear(*self.ln1)
      x = x + self.act(x.linear(*self.ff1)).linear(*self.ff2).dropout(self.dropout)
      x = x.layernorm().linear(*self.ln2)
      return x

class CommaVQDecoder:
   def __init__(self):
      self.tok_embed = Embedding(VOCAB_SIZE, DIM)
      self.pos_embed = Embedding(MAX_CTX_SIZE + FRAME_SIZE, DIM)
      self.layers = [TransformerBlock(DIM, N_HEADS, FF_DIM) for _ in range(DEC_LAYERS)]
      self.class_head = Linear(DIM, VOCAB_SIZE)

   def __call__(self, x:Tensor) -> Tensor:
      x = self.tok_embed(x) + self.pos_embed(Tensor.arange(0, x.shape[1], requires_grad=False).reshape((1,-1)))
      x = x.sequential(self.layers)
      x = self.class_head(x)
      return x


def _to_image(output, is_numpy=False):
   if not hasattr(_to_image, 'session'):
      options  = ort.SessionOptions()
      provider = 'CPUExecutionProvider' # 'CUDAExecutionProvider'
      _to_image.session = ort.InferenceSession(f'gpt2m/decoder.onnx', options, [provider])
   to_np = lambda x: x if is_numpy else x.numpy()
   outputs = _to_image.session.run(None, {'encoding_indices': to_np(output.reshape(1,8,16)).astype(np.int64)})
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
   for i in range(1):
      baseline_path = os.path.join(SAVE_FOLDER, IMG_FILENAME.format('test_baseline', i))
      if not os.path.exists(baseline_path):
         _save_decoded_image(baselines[i], baseline_path)
      _save_decoded_image(outputs[i], os.path.join(SAVE_FOLDER, ITER_FOLDER.format(step), IMG_FILENAME.format('decoded_test', i)))
   
   plt.clf()
   plt.plot(np.arange(1, len(train_loss)+1), train_loss, label='train')
   plt.plot(np.arange(TEST_EVERY, len(test_loss)*TEST_EVERY+1, TEST_EVERY), test_loss, label='test')
   plt.ylim(0, None)
   plt.legend()
   plt.savefig(os.path.join(SAVE_FOLDER, LOSS_FILENAME))
   
   plt.clf()
   plt.plot(np.arange(1, len(train_acc)+1), train_acc, label='train')
   plt.plot(np.arange(TEST_EVERY, len(test_acc)*TEST_EVERY+1, TEST_EVERY), test_acc, label='test')
   plt.ylim(0.0, 1.0)
   plt.legend()
   plt.savefig(os.path.join(SAVE_FOLDER, ACCS_FILENAME))


def load_datasets():
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
   test_datas = [np.load(dataset['40'][i,0]['path'][0]) for i in range(10)]
   print(f"loaded dataset in {time.time() - s_time:.1f} seconds")

   return train_datas, test_datas

def get_latest_weights_folder():
   root = "weights"
   subdirs = [os.path.join(root, d) for d in os.listdir(root)]
   return max([d for d in subdirs if os.path.isdir(d)], key=os.path.getmtime)

def inject_latest_weights_folder(restore_count=2):
   latest = get_latest_weights_folder()
   global SAVE_FOLDER, LOSS_FILENAME, ACCS_FILENAME
   SAVE_FOLDER    = latest
   LOSS_FILENAME  = f"graph_loss_restore_{restore_count}.png"
   ACCS_FILENAME  = f"graph_acc_restore_{restore_count}.png"

def train(pickup_step=None):
   np.random.seed(1337)
   model = CommaVQDecoder()

   step, is_test = 0, False
   if pickup_step is not None:
      step = pickup_step
      inject_latest_weights_folder()
      model_dir = f"{SAVE_FOLDER}/{MODEL_FILENAME.format(step)}"
      print(f"Using {model_dir}")
      load_state_dict(model, safe_load(model_dir))

   params = get_parameters(model)
   print(f"{sum(prod(p.shape) for p in params)/1e6:.2f} million parameters")
   opt = Adam(params, LR)
   
   train_datas, test_datas = load_datasets()

   s_time = time.time()
   train_loss, test_loss = [], []
   train_acc,  test_acc  = [], []
   while step < ITERATIONS:
      if is_test:
         data = test_datas[1]
         index = 0
      else:
         data = train_datas[step % len(train_datas)]
         index = np.random.randint(data.shape[0] - CTX_FRAMES - 1)

      frames = np.full((BS,MAX_CTX_SIZE+FRAME_SIZE), BOS_TOKEN, dtype=np.float32)
      for i in range(CTX_FRAMES+1):
         frames[0,1+i*FRAME_SIZE:(i+1)*FRAME_SIZE] = data[index + i].flatten()

      inputs  = Tensor(frames[:,:-1], dtype=dtypes.float32, requires_grad=False)
      targets = Tensor(frames[:,-(FRAME_SIZE-1):], dtype=dtypes.float32)
      output  = model(inputs)[:,-(FRAME_SIZE-1):]
      loss    = output.sparse_categorical_crossentropy(targets)

      if is_test:
         test_loss.append(loss.numpy())
         test_acc.append((output.argmax(axis=-1) == targets).mean().numpy())
      else:
         loss.realize()
         opt.zero_grad()
         loss.backward()
         opt.step()

         train_loss.append(loss.numpy())
         train_acc.append((output.argmax(axis=-1) == targets).mean().numpy())
      
      if (step+1) % TEST_EVERY == 0:
         if is_test:
            step += 1
            print(f"Step {str(step): >5} | Train Loss: {sum(train_loss[-TEST_EVERY:])/TEST_EVERY:.4f} | Train Accuracy: {100.0*sum(train_acc[-TEST_EVERY:])/TEST_EVERY:.2f}% | Test Loss: {test_loss[-1]:.4f} | Test Accuracy: {100.0*test_acc[-1]:.2f}% | {(time.time() - s_time) / float(TEST_EVERY):.2f} sec/iter")
            s_time = time.time()
            save_test_snapshot(step, output.argmax(axis=-1), targets, train_loss, test_loss, train_acc, test_acc)
         is_test = not is_test
      else:
         step += 1

      if step % SAVE_EVERY == 0:
         if not os.path.exists(SAVE_FOLDER):
            os.makedirs(SAVE_FOLDER)
         safe_save(get_state_dict(model), os.path.join(SAVE_FOLDER, MODEL_FILENAME.format(step)))


def generate(gen_count=50, iter=152000):
   Tensor.no_grad = True

   np.random.seed(1337)
   model = CommaVQDecoder()
   latest = get_latest_weights_folder()
   load_state_dict(model, safe_load(os.path.join(latest, MODEL_FILENAME.format(iter))))

   data = np.load(load_dataset("commaai/commavq", split='40[:2]')[0,0]['path'][0])
   prev_frames = np.zeros((CTX_FRAMES+gen_count,FRAME_SIZE-1))
   prev_frames[:CTX_FRAMES] = data[:CTX_FRAMES].reshape((CTX_FRAMES,-1))

   times = []
   for i in range(gen_count):
      s_time = time.time()

      data = prev_frames[i:i+CTX_FRAMES]
      frames = np.full((BS,MAX_CTX_SIZE+FRAME_SIZE), BOS_TOKEN, dtype=np.float32)
      for i in range(CTX_FRAMES):
         frames[0,1+i*FRAME_SIZE:(i+1)*FRAME_SIZE] = data[i].flatten()

      for j in trange(FRAME_SIZE-1, 0, -1):
         inputs = Tensor(frames[:,:-j], dtype=dtypes.float32, requires_grad=False)
         frames[:,-j] = model(inputs).argmax(axis=-1)[:,-1].numpy()
         del inputs

      output = frames[0,-(FRAME_SIZE-1):]

      prev_frames[CTX_FRAMES+i,:] = output
      img = _to_image(output, is_numpy=True)

      cv_img = img[...,::-1]
      cv2.imwrite(f'outputs/frame_{i}.png', cv_img)
      cv2.imshow('frames', cv2.resize(cv_img, (cv_img.shape[1]*4, cv_img.shape[0]*4)))
      cv2.waitKey()

      times.append(time.time() - s_time)
      while len(times) > 4:
         times.pop(0)
      fps = 1.0 / (sum(times)/len(times))
      print( f"{fps/20.0:.2f}x realtime (avg {fps:.2f} fps)" )

   # write_video(prev_frames.reshape(CTX_FRAMES+gen_count,8,16), "outputs/video.mp4")


if __name__ == "__main__":
   generate()

