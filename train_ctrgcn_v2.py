"""
KARSL-502 Sign Language Recognition  —  CTR-GCN v6
====================================================
Architecture : CTR-GCN + Temporal Transformer Head
Streams      : Joint + Bone + Motion  (3-stream score fusion)

التغييرات في v6 (مقارنةً بـ v5):
  1. ✅ FIX SWA: شيلنا SWALR واستبدلناها بـ constant LR صغير → يمنع الـ collapse
  2. ✅ FIX SWA_START: من 55 → 60 عشان الـ cosine يكمل قبل الـ averaging
  3. ✅ ADD Signer Augmentation: يحاكي signer جديد عن طريق
       - random global translation (يحاكي اختلاف وضع الجسم)
       - random inter-joint scale (يحاكي اختلاف حجم اليد/الوجه)
       - random speed offset per-joint (يحاكي اختلاف سرعة الحركة)
     ده الأهم لـ signer-independent generalization
  4. ✅ ADD z-axis noise أكبر 3x في الـ spatial augmentation
  5. ✅ ADD face partial dropout (10%)
  6. ✅ TUNE Dropout: FC layer من 0.4 → 0.5
"""

import os
os.environ['TORCH_COMPILE_DISABLE'] = '1'

import math, logging, random, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datetime import datetime
from torch.optim.swa_utils import AveragedModel

# ── استيراد الـ graph ────────────────────────────────────────────────────────
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
    sys.path.insert(0, os.path.join(BASE_DIR, "src"))

try:
    from data_prep.graph_karsl_75_v2 import Graph as KARSLGraph, GRAPH_PAIRS as BONE_PAIRS_SOURCE
    EXTERNAL_GRAPH = True
except ImportError:
    EXTERNAL_GRAPH = False
    print("⚠️  graph_karsl_75_v2.py مش موجود — هيتستخدم الـ fallback graph")

# ══════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════
NUM_CLASSES   = 502
NUM_JOINTS    = 75
TARGET_FRAMES = 32        # لازم يتطابق مع 2AGCN_input.py
IN_CHANNELS   = 3

BATCH_SIZE    = 32
TOTAL_EPOCHS  = 80
WARMUP_EPOCHS = 5
BASE_LR       = 0.05
WEIGHT_DECAY  = 5e-4
GRAD_CLIP     = 1.0
LABEL_SMOOTH  = 0.1
DROPOUT       = 0.5        # ✅ رفعناه من 0.4 → 0.5 لـ signer generalization

# ── 3-stream fusion weights ──────────────────────────────────────
WEIGHT_J = 0.35
WEIGHT_B = 0.40
WEIGHT_M = 0.25

# ── SWA ─────────────────────────────────────────────────────────
# ✅ SWA_START من 55 → 60 عشان الـ cosine يكمل
# ✅ SWA_LR صغير جداً ومثبّت → يمنع الـ collapse تماماً
SWA_START = 60
SWA_LR    = 5e-5

DATA_ROOT      = r"C:\KARSL-502\core\output\data\agcn_final_input"
CHECKPOINT_DIR = r"C:\KARSL-502\core\output\checkpoints_v6"

# ── Joint index ranges (لازم تتطابق مع 2AGCN_input.py و graph_karsl_75_v2.py) ──
# 0–32   → Face (33 joints)
# 33–53  → Left Hand (21 joints)
# 54–74  → Right Hand (21 joints)
FACE_JOINTS = list(range(0,  33))
LH_JOINTS   = list(range(33, 54))
RH_JOINTS   = list(range(54, 75))

# ══════════════════════════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════════════════════════
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(CHECKPOINT_DIR, "train_log.txt"),
                            encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
#  GRAPH  —  نفس الـ build_adjacency بالظبط
# ══════════════════════════════════════════════════════════════════
def build_adjacency():
    """
    يرجع:
      A_np  : np.ndarray shape (K, V, V)
      K     : int — عدد الـ subsets
      pairs : list of (int, int) — للـ bone stream
    """
    if EXTERNAL_GRAPH:
        g = KARSLGraph(strategy='spatial', max_hop=1)
        logger.info(f"  ✅ Graph مستوردة من graph_karsl_75_v2.py  "
                    f"A.shape={g.A.shape}  pairs={len(g.pairs)}")
        return g.A, g.A.shape[0], g.pairs

    # ── Fallback ────────────────────────────────────────────────
    logger.warning("  ⚠️  Fallback graph — نتائج أقل دقة")
    num_node  = NUM_JOINTS
    self_link = [(i, i) for i in range(num_node)]
    hand_raw  = [
        (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
    ]
    lh        = [(i+33, j+33) for i,j in hand_raw]
    rh        = [(i+54, j+54) for i,j in hand_raw]
    all_edges = lh + rh + self_link
    all_edges = [(i,j) for i,j in all_edges if i < num_node and j < num_node]

    A = np.zeros((num_node, num_node), dtype=np.float32)
    for i, j in all_edges:
        A[i,j] = 1.0; A[j,i] = 1.0
    D     = np.sum(A, axis=0)
    D     = np.where(D > 0, D ** -0.5, 0)
    A_norm = np.diag(D) @ A @ np.diag(D)
    A_np  = A_norm[np.newaxis, ...]   # (1, V, V)
    pairs = [(i,j) for i,j in all_edges if i != j]
    return A_np, 1, pairs


# ══════════════════════════════════════════════════════════════════
#  CTR-GCN BLOCKS  —  نفس البنية بالظبط
# ══════════════════════════════════════════════════════════════════
class CTRGC(nn.Module):
    def __init__(self, in_c, out_c, num_subsets=3, rel_reduction=8):
        super().__init__()
        self.num_subsets = num_subsets
        rel_c = max(in_c // rel_reduction, 16)
        self.conv1 = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_c, rel_c, 1),
                          nn.BatchNorm2d(rel_c), nn.ReLU())
            for _ in range(num_subsets)
        ])
        self.conv2 = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_c, rel_c, 1),
                          nn.BatchNorm2d(rel_c), nn.ReLU())
            for _ in range(num_subsets)
        ])
        self.conv3 = nn.ModuleList([
            nn.Conv2d(in_c, out_c, 1)
            for _ in range(num_subsets)
        ])
        self.tanh = nn.Tanh()
        self.bn   = nn.BatchNorm2d(out_c)

    def forward(self, x, A):
        # x: (N, C, T, V)  |  A: (K, V, V)
        out = None
        for k in range(min(A.shape[0], self.num_subsets)):
            A_k = A[k]
            z1  = self.conv1[k](x).mean(-2)   # (N, rel_c, V)
            z2  = self.conv2[k](x).mean(-2)
            B   = self.tanh(
                torch.einsum('nci,ncj->nij', z1, z2) / math.sqrt(z1.shape[1])
            )
            y_k = torch.einsum('nctv,nvw->nctw', x, A_k + B)
            y_k = self.conv3[k](y_k)
            out = y_k if out is None else out + y_k
        return self.bn(out)


class CTRGCNBlock(nn.Module):
    def __init__(self, in_c, out_c, A, stride=1, residual=True,
                 kernel_size=5, dilations=(1, 2)):
        super().__init__()
        self.register_buffer('A', torch.from_numpy(A.astype(np.float32)))
        self.ctrgc = CTRGC(in_c, out_c, num_subsets=A.shape[0])
        self.relu  = nn.ReLU(inplace=True)

        n_branches = len(dilations) + 1
        branch_c   = out_c // n_branches
        self.tcn_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_c, branch_c, (kernel_size, 1),
                          stride=(stride, 1),
                          padding=((kernel_size + (kernel_size-1)*(d-1))//2, 0),
                          dilation=(d, 1)),
                nn.BatchNorm2d(branch_c),
            ) for d in dilations
        ])
        rem_c = out_c - branch_c * len(dilations)
        self.tcn_point = nn.Sequential(
            nn.Conv2d(out_c, rem_c, 1, stride=(stride, 1)),
            nn.BatchNorm2d(rem_c),
        )

        if not residual:
            self.residual = lambda x: 0
        elif in_c == out_c and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=(stride, 1)),
                nn.BatchNorm2d(out_c),
            )

    def forward(self, x):
        res      = self.residual(x)
        y        = self.ctrgc(x, self.A)
        branches = [b(y) for b in self.tcn_branches] + [self.tcn_point(y)]
        y        = torch.cat(branches, dim=1)
        if isinstance(res, torch.Tensor):
            if y.shape[2] != res.shape[2]:
                res = F.interpolate(res, size=(y.shape[2], y.shape[3]), mode='nearest')
            return self.relu(y + res)
        return self.relu(y)


# ══════════════════════════════════════════════════════════════════
#  TEMPORAL TRANSFORMER HEAD
# ══════════════════════════════════════════════════════════════════
class TemporalTransformerHead(nn.Module):
    def __init__(self, d_model=256, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.pos_enc     = nn.Embedding(256, d_model)
        enc_layer        = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model*2,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        N, C, T, V = x.shape
        x   = x.mean(-1).permute(0, 2, 1)       # (N, T, C)
        cls = self.cls_token.expand(N, -1, -1)
        x   = torch.cat([cls, x], dim=1)         # (N, T+1, C)
        pos = torch.arange(x.shape[1], device=x.device)
        x   = x + self.pos_enc(pos)
        x   = self.transformer(x)
        return x[:, 0]                            # CLS token


# ══════════════════════════════════════════════════════════════════
#  FULL MODEL
# ══════════════════════════════════════════════════════════════════
class CTRGCN(nn.Module):
    def __init__(self, num_class=NUM_CLASSES, num_point=NUM_JOINTS,
                 in_channels=IN_CHANNELS, A=None, dropout=DROPOUT):
        super().__init__()
        A = A.astype(np.float32)
        self.data_bn = nn.BatchNorm1d(in_channels * num_point)
        self.blocks  = nn.ModuleList([
            CTRGCNBlock( in_channels,  64, A, residual=False),
            CTRGCNBlock( 64,  64, A),
            CTRGCNBlock( 64,  64, A),
            CTRGCNBlock( 64,  64, A),
            CTRGCNBlock( 64, 128, A, stride=2),
            CTRGCNBlock(128, 128, A),
            CTRGCNBlock(128, 128, A),
            CTRGCNBlock(128, 256, A, stride=2),
            CTRGCNBlock(256, 256, A),
            CTRGCNBlock(256, 256, A),
        ])
        self.temp_attn = TemporalTransformerHead(d_model=256, nhead=4, num_layers=2)
        self.drop      = nn.Dropout(p=dropout)
        self.fc        = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2.0 / num_class))

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0,4,3,1,2).contiguous().reshape(N*M, V*C, T)
        x = self.data_bn(x)
        x = x.reshape(N,M,V,C,T).permute(0,1,3,4,2).contiguous().reshape(N*M, C, T, V)
        for blk in self.blocks:
            x = blk(x)
        x = self.temp_attn(x)
        x = self.drop(x)
        x = x.reshape(N, M, -1).mean(1)
        return self.fc(x)


# ══════════════════════════════════════════════════════════════════
#  BONE / MOTION STREAMS
# ══════════════════════════════════════════════════════════════════
def get_bone_stream(data: torch.Tensor, pairs) -> torch.Tensor:
    """data: (N, C, T, V, M) | pairs من graph_karsl_75_v2.py"""
    bone = torch.zeros_like(data)
    for v1, v2 in pairs:
        if v1 < data.shape[3] and v2 < data.shape[3]:
            bone[:, :, :, v1, :] = data[:, :, :, v1, :] - data[:, :, :, v2, :]
    return bone


def get_motion_stream(data: torch.Tensor) -> torch.Tensor:
    motion = torch.zeros_like(data)
    motion[:, :, 1:, :, :] = data[:, :, 1:, :, :] - data[:, :, :-1, :, :]
    return motion


# ══════════════════════════════════════════════════════════════════
#  DATASET + AUGMENTATION
#  input shape من 2AGCN_input.py: (3, 32, 75, 1)
# ══════════════════════════════════════════════════════════════════
class KARSL_SkeletonDataset(Dataset):
    def __init__(self, data_path, label_path, use_augmentation=False,
                 target_frames=TARGET_FRAMES):
        self.data  = np.load(data_path)    # (N, 3, 32, 75, 1)
        self.label = np.load(label_path)
        self.aug   = use_augmentation
        self.T     = target_frames
        logger.info(f"  {data_path.split(os.sep)[-1]} → {self.data.shape},"
                    f" samples={len(self.label)}")

    def __len__(self): return len(self.label)

    def __getitem__(self, index):
        x = self.data[index].copy()   # (3, 32, 75, 1)
        if self.aug:
            x = self._speed_warp(x)
            x = self._signer_aug(x)   # ✅ الجديد — قبل الـ spatial
            x = self._spatial(x)
        return torch.from_numpy(x).float(), torch.tensor(self.label[index]).long()

    # ── Speed Warp ───────────────────────────────────────────────
    def _speed_warp(self, x):
        """تسريع/تبطيء الحركة — يحاكي اختلاف سرعة الـ signer"""
        C, T, V, M = x.shape
        min_s = max(0.85, 10.0 / T)
        max_s = min(1.30, (T+6) / T)
        scale = np.random.uniform(min_s, max_s)
        new_t = max(int(T * scale), 10)
        tmp   = F.interpolate(
            torch.from_numpy(x).permute(0,2,1,3),   # (C, V, T, M)
            size=(new_t, M), mode='bilinear', align_corners=False
        ).permute(0,2,1,3).numpy()                   # (C, new_t, V, M)
        cur_t = tmp.shape[1]
        if cur_t >= self.T:
            start = np.random.randint(0, cur_t - self.T + 1)
            return tmp[:, start:start+self.T, :, :]
        return np.pad(tmp, ((0,0),(0,self.T-cur_t),(0,0),(0,0)), mode='reflect')

    # ── Signer Augmentation ──────────────────────────────────────
    def _signer_aug(self, x):
        """
        يحاكي signer جديد مختلف — نسخة آمنة:

        1. Global translation: offset صغير على كل الـ joints مع بعض
           يحاكي اختلاف وضع الجسم أمام الكاميرا

        2. Per-region scale: scale مختلف للوجه واليدين بشكل مستقل
           يحاكي اختلاف حجم أعضاء الجسم بين الـ signers

        ملاحظة: شيلنا الـ per-joint temporal roll لأنه كان بيخرب
        الـ bone stream (bone = فرق بين joints في نفس الـ frame)
        """
        C, T, V, M = x.shape   # (3, 32, 75, 1)

        # 1. Global translation — صغير جداً عشان مش نكسر الـ normalization
        if np.random.random() < 0.7:
            shift = np.random.uniform(-0.03, 0.03, (C, 1, 1, M)).astype(np.float32)
            x = x + shift

        # 2. Per-region scale — نطاق أضيق من الأول عشان نحافظ على الـ bone ratios
        if np.random.random() < 0.5:
            face_scale = np.random.uniform(0.93, 1.07)
            x[:, :, FACE_JOINTS, :] *= face_scale
            lh_scale = np.random.uniform(0.92, 1.08)
            x[:, :, LH_JOINTS, :] *= lh_scale
            rh_scale = np.random.uniform(0.92, 1.08)
            x[:, :, RH_JOINTS, :] *= rh_scale

        return x

    # ── Spatial Augmentation ─────────────────────────────────────
    def _spatial(self, x):
        """يحاكي اختلاف الكاميرا والإضاءة"""
        # noise: xy عادي، z أكبر عشان depth أقل دقة في الكاميرا
        noise = np.random.normal(0, 0.005, x.shape).astype(np.float32)
        noise[2] *= 3.0   # ✅ z-axis noise أكبر

        # scale: بعد/قرب من الكاميرا
        scale = np.random.uniform(0.85, 1.15)

        # 2D rotation: ميل الجسم
        if np.random.random() < 0.4:
            angle        = np.random.uniform(-20, 20) * math.pi / 180
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            x_rot = cos_a * x[0] - sin_a * x[1]
            y_rot = sin_a * x[0] + cos_a * x[1]
            x[0]  = x_rot
            x[1]  = y_rot

        # hand dropout: حجب اليد
        if np.random.random() < 0.20:
            hand  = np.random.choice([0, 1])
            start = 33 + hand * 21
            x[:, :, start:start+21, :] = 0.0

        # face partial dropout ✅ جديد: إضاءة وحشة تأثر على الوجه
        if np.random.random() < 0.10:
            x[:, :, FACE_JOINTS, :] *= np.random.uniform(0.5, 1.0)

        # temporal jitter: frame drops
        if np.random.random() < 0.3:
            _, T, _, _ = x.shape
            n_drop   = np.random.randint(1, max(2, T//8))
            drop_idx = np.random.choice(T, n_drop, replace=False)
            for idx in drop_idx:
                prev = max(0, idx-1); nxt = min(T-1, idx+1)
                x[:, idx, :, :] = (x[:, prev, :, :] + x[:, nxt, :, :]) / 2.0

        return (x + noise) * scale


# ══════════════════════════════════════════════════════════════════
#  TRAINING HELPERS
# ══════════════════════════════════════════════════════════════════
def topk_accuracy(output, target, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        bs   = target.size(0)
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        correct = pred.t().eq(target.view(1,-1).expand_as(pred.t()))
        return [correct[:k].reshape(-1).float().sum().item()*100.0/bs for k in topk]


def cosine_warmup_lr(epoch: int) -> float:
    """Cosine schedule مع linear warmup"""
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS
    prog = (epoch - WARMUP_EPOCHS) / max(TOTAL_EPOCHS - WARMUP_EPOCHS, 1)
    return 0.5 * (1.0 + math.cos(math.pi * prog))


# ══════════════════════════════════════════════════════════════════
#  TEMPERATURE CALIBRATION
# ══════════════════════════════════════════════════════════════════
def calibrate_temperature(model_j, model_b, model_m, val_loader, device, bone_pairs):
    logger.info("\n── Temperature Calibration ──────────────────────────────")
    temperature = nn.Parameter(torch.ones(1, device=device) * 1.5)
    optimizer   = torch.optim.LBFGS([temperature], lr=0.01, max_iter=100)
    criterion   = nn.CrossEntropyLoss()

    logits_list, labels_list = [], []
    model_j.eval(); model_b.eval(); model_m.eval()

    with torch.no_grad():
        for data, label in val_loader:
            data, label = data.to(device), label.to(device)
            bone   = get_bone_stream(data, bone_pairs)
            motion = get_motion_stream(data)
            fused  = (WEIGHT_J * model_j(data)
                    + WEIGHT_B * model_b(bone)
                    + WEIGHT_M * model_m(motion))
            logits_list.append(fused.cpu())
            labels_list.append(label.cpu())

    logits = torch.cat(logits_list).to(device)
    labels = torch.cat(labels_list).to(device)

    def eval_fn():
        optimizer.zero_grad()
        loss = criterion(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(eval_fn)
    T_opt = temperature.item()
    logger.info(f"  Optimal temperature: {T_opt:.4f}  "
                f"(>1 = overconfident | <1 = underconfident)")
    return T_opt


# ══════════════════════════════════════════════════════════════════
#  MAIN TRAINING LOOP
# ══════════════════════════════════════════════════════════════════
def train():
    logger.info("=" * 65)
    logger.info(f"  CTR-GCN v6  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    logger.info(f"  3 streams: J×{WEIGHT_J} + B×{WEIGHT_B} + M×{WEIGHT_M}")
    logger.info(f"  TARGET_FRAMES={TARGET_FRAMES}  NUM_JOINTS={NUM_JOINTS}")
    logger.info(f"  SWA from ep {SWA_START}  SWA_LR={SWA_LR}  (fixed, no cycling)")
    logger.info(f"  DROPOUT={DROPOUT}  External graph: {EXTERNAL_GRAPH}")
    logger.info("=" * 65)

    # ── Adjacency ────────────────────────────────────────────────
    A_matrix, K, bone_pairs = build_adjacency()
    logger.info(f"  A.shape={A_matrix.shape}  K={K}  bone_pairs={len(bone_pairs)}")

    # ── Data ─────────────────────────────────────────────────────
    train_ds = KARSL_SkeletonDataset(
        os.path.join(DATA_ROOT, "train_data.npy"),
        os.path.join(DATA_ROOT, "train_label.npy"),
        use_augmentation=True)
    eval_ds  = KARSL_SkeletonDataset(
        os.path.join(DATA_ROOT, "eval_data.npy"),
        os.path.join(DATA_ROOT, "eval_label.npy"),
        use_augmentation=False)

    # نص الـ eval data للـ validation، النص التاني للـ test الحقيقي
    all_idx = list(range(len(eval_ds)))
    val_idx, _ = train_test_split(all_idx, test_size=0.5, random_state=42)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(Subset(eval_ds, val_idx), batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=True)

    logger.info(f"  Train={len(train_ds):,}  Val={len(val_idx):,}"
                f"  steps/ep={len(train_loader)}")

    # ── Device ───────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"  Device: {device}")
    if device.type == "cuda":
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── Models ───────────────────────────────────────────────────
    model_j = CTRGCN(A=A_matrix).to(device)
    model_b = CTRGCN(A=A_matrix).to(device)
    model_m = CTRGCN(A=A_matrix).to(device)

    total_p = sum(p.numel() for p in model_j.parameters() if p.requires_grad)
    logger.info(f"  Params per stream: {total_p:,}  (×3 = {total_p*3:,} total)")

    def make_opt(m):
        return optim.SGD(m.parameters(), lr=BASE_LR,
                         momentum=0.9, weight_decay=WEIGHT_DECAY, nesterov=True)

    opt_j = make_opt(model_j); opt_b = make_opt(model_b); opt_m = make_opt(model_m)
    sch_j = optim.lr_scheduler.LambdaLR(opt_j, cosine_warmup_lr)
    sch_b = optim.lr_scheduler.LambdaLR(opt_b, cosine_warmup_lr)
    sch_m = optim.lr_scheduler.LambdaLR(opt_m, cosine_warmup_lr)

    # ✅ SWA بدون SWALR — بس AveragedModel
    swa_j = AveragedModel(model_j)
    swa_b = AveragedModel(model_b)
    swa_m = AveragedModel(model_m)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    use_amp   = device.type == "cuda"
    scaler_j  = GradScaler(enabled=use_amp)
    scaler_b  = GradScaler(enabled=use_amp)
    scaler_m  = GradScaler(enabled=use_amp)

    # ── Resume ───────────────────────────────────────────────────
    ckpt_path   = os.path.join(CHECKPOINT_DIR, "last_checkpoint.pth")
    start_epoch = 1
    best_acc    = 0.0

    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model_j.load_state_dict(ck["model_j"])
        model_b.load_state_dict(ck["model_b"])
        model_m.load_state_dict(ck["model_m"])
        opt_j.load_state_dict(ck["opt_j"]); opt_b.load_state_dict(ck["opt_b"])
        opt_m.load_state_dict(ck["opt_m"])
        sch_j.load_state_dict(ck["sch_j"]); sch_b.load_state_dict(ck["sch_b"])
        sch_m.load_state_dict(ck["sch_m"])
        start_epoch = ck["epoch"] + 1
        best_acc    = ck.get("best_acc", 0.0)
        logger.info(f"  Resumed from epoch {start_epoch-1}  (best={best_acc:.2f}%)")

    # ── Epoch Loop ───────────────────────────────────────────────
    for epoch in range(start_epoch, TOTAL_EPOCHS + 1):
        using_swa = (epoch >= SWA_START)
        cur_lr    = opt_j.param_groups[0]["lr"]
        logger.info(f"\nEpoch {epoch}/{TOTAL_EPOCHS}  lr={cur_lr:.5f}"
                    + ("  [SWA]" if using_swa else ""))

        # ── Train ────────────────────────────────────────────────
        model_j.train(); model_b.train(); model_m.train()
        loss_j_sum = loss_b_sum = loss_m_sum = 0.0

        pbar = tqdm(train_loader, desc=f"  Train {epoch}", ncols=95)
        for data, label in pbar:
            data, label = data.to(device), label.to(device)
            bone   = get_bone_stream(data, bone_pairs)
            motion = get_motion_stream(data)

            for model, opt, scaler, x, attr in [
                (model_j, opt_j, scaler_j, data,   "j"),
                (model_b, opt_b, scaler_b, bone,   "b"),
                (model_m, opt_m, scaler_m, motion, "m"),
            ]:
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    out  = model(x)
                    loss = criterion(out, label)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(opt)
                scaler.update()
                if attr == "j":   loss_j_sum += loss.item()
                elif attr == "b": loss_b_sum += loss.item()
                else:             loss_m_sum += loss.item()

            pbar.set_postfix(
                Lj=f"{loss_j_sum/(pbar.n+1):.3f}",
                Lb=f"{loss_b_sum/(pbar.n+1):.3f}",
                Lm=f"{loss_m_sum/(pbar.n+1):.3f}",
            )

        # ── LR / SWA Step ────────────────────────────────────────
        if using_swa:
            # ✅ بنثبت الـ LR على قيمة صغيرة جداً — مش بنرفعه زي SWALR
            for opt in [opt_j, opt_b, opt_m]:
                for pg in opt.param_groups:
                    pg['lr'] = SWA_LR
            swa_j.update_parameters(model_j)
            swa_b.update_parameters(model_b)
            swa_m.update_parameters(model_m)
        else:
            sch_j.step(); sch_b.step(); sch_m.step()

        # ── Validate ─────────────────────────────────────────────
        eval_j = swa_j if using_swa else model_j
        eval_b = swa_b if using_swa else model_b
        eval_m = swa_m if using_swa else model_m
        eval_j.eval(); eval_b.eval(); eval_m.eval()

        t1j=t5j=t1b=t5b=t1m=t5m=t1f=t5f=n=0.0
        with torch.no_grad():
            for data, label in val_loader:
                data, label = data.to(device), label.to(device)
                bs = label.size(0); n += bs
                bone   = get_bone_stream(data, bone_pairs)
                motion = get_motion_stream(data)
                oj = eval_j(data); ob = eval_b(bone); om = eval_m(motion)
                of = WEIGHT_J*oj + WEIGHT_B*ob + WEIGHT_M*om

                a = topk_accuracy(oj, label); t1j+=a[0]*bs; t5j+=a[1]*bs
                a = topk_accuracy(ob, label); t1b+=a[0]*bs; t5b+=a[1]*bs
                a = topk_accuracy(om, label); t1m+=a[0]*bs; t5m+=a[1]*bs
                a = topk_accuracy(of, label); t1f+=a[0]*bs; t5f+=a[1]*bs

        t1j/=n; t5j/=n; t1b/=n; t5b/=n; t1m/=n; t5m/=n; t1f/=n; t5f/=n

        flag = "  ← BEST ✅" if t1f > best_acc else ""
        logger.info(
            f"  Joint  top-1={t1j:.2f}%  top-5={t5j:.2f}%\n"
            f"  Bone   top-1={t1b:.2f}%  top-5={t5b:.2f}%\n"
            f"  Motion top-1={t1m:.2f}%  top-5={t5m:.2f}%\n"
            f"  Fused  top-1={t1f:.2f}%  top-5={t5f:.2f}%{flag}"
        )

        # ── Checkpoint ───────────────────────────────────────────
        state = dict(
            epoch=epoch,
            model_j=model_j.state_dict(), model_b=model_b.state_dict(),
            model_m=model_m.state_dict(),
            opt_j=opt_j.state_dict(),     opt_b=opt_b.state_dict(),
            opt_m=opt_m.state_dict(),
            sch_j=sch_j.state_dict(),     sch_b=sch_b.state_dict(),
            sch_m=sch_m.state_dict(),
            best_acc=best_acc, top1_fused=t1f,
            bone_pairs=bone_pairs,
        )
        torch.save(state, ckpt_path)
        if t1f > best_acc:
            best_acc = t1f
            torch.save(state, os.path.join(CHECKPOINT_DIR, "best_model.pth"))
            logger.info("  ✅ Best model saved.")

    # ── SWA BatchNorm Update ──────────────────────────────────────
    logger.info("\nUpdating SWA BatchNorm statistics...")
    swa_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=0)
    with torch.no_grad():
        for data, _ in tqdm(swa_loader, desc="  BN update", ncols=70):
            data = data.to(device)
            swa_j(data)
            swa_b(get_bone_stream(data, bone_pairs))
            swa_m(get_motion_stream(data))

    torch.save({
        "swa_j":      swa_j.module.state_dict(),
        "swa_b":      swa_b.module.state_dict(),
        "swa_m":      swa_m.module.state_dict(),
        "bone_pairs": bone_pairs,
    }, os.path.join(CHECKPOINT_DIR, "swa_final.pth"))

    # ── Temperature Calibration ───────────────────────────────────
    logger.info("\nRunning temperature calibration...")
    best_ck = torch.load(os.path.join(CHECKPOINT_DIR, "best_model.pth"),
                         map_location=device)
    model_j.load_state_dict(best_ck["model_j"])
    model_b.load_state_dict(best_ck["model_b"])
    model_m.load_state_dict(best_ck["model_m"])

    T_opt = calibrate_temperature(model_j, model_b, model_m,
                                  val_loader, device, bone_pairs)
    best_ck["temperature"] = T_opt
    torch.save(best_ck, os.path.join(CHECKPOINT_DIR, "best_model.pth"))
    logger.info(f"  Temperature {T_opt:.4f} saved in best_model.pth")

    logger.info(f"\n{'='*65}")
    logger.info(f"  Done. Best fused top-1: {best_acc:.2f}%")
    logger.info(f"{'='*65}")


if __name__ == "__main__":
    train()