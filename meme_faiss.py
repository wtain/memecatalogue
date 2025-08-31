import os
import torch
import clip
import faiss
from PIL import Image


# https://github.com/dmlc/xgboost/issues/1715
# https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

# -------- CONFIG --------
TEMPLATE_DIR = "memes"
QUERY_DIR = "new_memes"


def decide_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


DEVICE = decide_device()

print("Running on:", DEVICE)

# -------- LOAD CLIP MODEL --------
print(f"Loading CLIP on {DEVICE}...")
model, preprocess = clip.load("ViT-B/32", device=DEVICE)

# -------- HELPERS --------
def get_image_embeddings(folder, extensions=(".jpg", ".png", ".jpeg"), device=DEVICE):
    image_paths, embeddings, labels = [], [], []
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        if not fname.lower().endswith(extensions):
            continue
        img = preprocess(Image.open(path)).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.encode_image(img)
        emb = emb / emb.norm(dim=-1, keepdim=True)  # normalize
        embeddings.append(emb.cpu().numpy())
        image_paths.append(path)
        # label = filename without number (e.g. "distracted_boyfriend")
        image_labels = "".join([c for c in os.path.splitext(fname)[0] if not c.isdigit()])
        labels.append(image_labels)
        print(fname, image_labels)
    return image_paths, labels, torch.vstack([torch.tensor(e) for e in embeddings]).numpy()

# -------- PREPARE TEMPLATE INDEX --------
print("Encoding template memes...")
template_paths, template_labels, template_embeddings = get_image_embeddings(TEMPLATE_DIR)

# Create FAISS index
dim = template_embeddings.shape[1]
index = faiss.IndexFlatIP(dim)  # inner product
index.add(template_embeddings)

# -------- QUERY NEW MEMES --------
print("Classifying new memes...")
query_paths, _, query_embeddings = get_image_embeddings(QUERY_DIR)

D, I = index.search(query_embeddings, k=3)  # top-3 matches

for q_idx, q_path in enumerate(query_paths):
    print(f"\n🔍 Query: {q_path}")
    for rank, idx in enumerate(I[q_idx]):
        similarity = D[q_idx][rank]
        print(f"   {rank+1}. {template_labels[idx]}  (score: {similarity:.3f})")
