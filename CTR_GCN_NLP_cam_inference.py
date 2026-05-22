"""
KARSL-502 — CTR-GCN + Arabic NLP Pipeline  [v6 — EXACT TRAINING MATCH]

"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import argparse, collections, math, re, time, warnings
import cv2, mediapipe as mp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import ImageFont, ImageDraw, Image
from typing import List, Tuple, Dict

try:
    from arabic_reshaper import reshape
    from bidi.algorithm import get_display
    ARABIC_SUPPORT = True
except ImportError:
    ARABIC_SUPPORT = False
    def reshape(t): return t
    def get_display(t): return t

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    Groq = None

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════
#  CONFIG train_ctrgcn_v2.py
# ══════════════════════════════════════════════════════════════════
NUM_CLASSES   = 502
NUM_JOINTS    = 75
IN_CHANNELS   = 3
WINDOW_FRAMES = 32         
STRIDE        = 4
THRESHOLD     = 0.30
TOP_K_DISPLAY = 3

WEIGHT_J      = 0.35       
WEIGHT_B      = 0.40
WEIGHT_M      = 0.25

# landmark indices  extract_landmarks.py + CTRGCN_input.py
N_FACE_SELECT = 33
POSE_START    = 468
LH_START      = 501
RH_START      = 522

GROQ_API_KEY  = "YOUR_GROQ_API_KEY"
GROQ_MODEL    = "llama-3.3-70b-versatile"

# joint groups (نفس التريننج)
FACE_JOINTS = list(range(0,  33))
LH_JOINTS   = list(range(33, 54))
RH_JOINTS   = list(range(54, 75))

# ══════════════════════════════════════════════════════════════════
#  GRAPH ـ graph_karsl_75_v2.py 
# ══════════════════════════════════════════════════════════════════

# Face edges (33 points — chain + closure)
FACE_EDGES = [
    (0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),
    (10,11),(11,12),(12,13),(13,14),(14,15),(15,16),(16,17),(17,18),
    (18,19),(19,20),(20,21),(21,22),(22,23),(23,24),(24,25),(25,26),
    (26,27),(27,28),(28,29),(29,30),(30,31),(31,32),
    (0,32),   # close the loop
    (0,4),(0,28),(0,16),
]

def _hand_edges(base):
    wrist = base
    edges = []
    finger_bases = [base+1, base+5, base+9, base+13, base+17]
    for fb in finger_bases:
        edges.append((wrist, fb))
    for fb in finger_bases:
        for i in range(3):
            edges.append((fb+i, fb+i+1))
    return edges

LH_EDGES = _hand_edges(33)
RH_EDGES = _hand_edges(54)

CROSS_EDGES = [
    (0, 33),(0, 54),(33, 54),
    (41, 0),(41, 4),(41, 16),(62, 0),(62, 28),(62, 16),
    (37, 16),(37, 4),(58, 16),(58, 28),
    (41, 9),(62, 23),
    (41, 62),(37, 58),(45, 66),
]

ALL_EDGES   = FACE_EDGES + LH_EDGES + RH_EDGES + CROSS_EDGES
GRAPH_PAIRS = ALL_EDGES   #  bone stream


def build_karsl_adjacency():
    """
    Spatial Configuration Partitioning — نفس KARSLGraph(strategy='spatial')
    يرجع A shape (3, 75, 75)
    """
    N = NUM_JOINTS
    graph = {i: [] for i in range(N)}
    for v1, v2 in ALL_EDGES:
        graph[v1].append(v2)
        graph[v2].append(v1)

    # BFS hop distances
    hop_dis = np.full((N, N), np.inf)
    np.fill_diagonal(hop_dis, 0)
    for root in range(N):
        queue, visited, dist = [root], {root}, {root: 0}
        while queue:
            node = queue.pop(0)
            for nb in graph[node]:
                if nb not in visited:
                    visited.add(nb)
                    dist[nb] = dist[node] + 1
                    hop_dis[root, nb] = dist[nb]
                    queue.append(nb)

    # Spatial partitioning (center = joint 0, nose tip)
    A = np.zeros((3, N, N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            if hop_dis[i, j] != 1 and i != j:
                continue
            if i == j:
                A[0, i, j] = 1.0
            elif hop_dis[0, j] <= hop_dis[0, i]:
                A[1, i, j] = 1.0
            else:
                A[2, i, j] = 1.0

   
    def normalize(a):
        d = a.sum(axis=0)
        dn = np.where(d > 0, 1.0 / d, 0.0)
        return np.diag(dn) @ a @ np.diag(dn)

    for k in range(3):
        A[k] = normalize(A[k])

    return A   # (3, 75, 75)



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
        out = None
        for k in range(min(A.shape[0], self.num_subsets)):
            A_k = A[k]
            z1  = self.conv1[k](x).mean(-2)
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


class TemporalTransformerHead(nn.Module):
    def __init__(self, d_model=256, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.pos_enc     = nn.Embedding(256, d_model)
        enc_layer        = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model*2,
            dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        N, C, T, V = x.shape
        x   = x.mean(-1).permute(0, 2, 1)
        cls = self.cls_token.expand(N, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        pos = torch.arange(x.shape[1], device=x.device)
        x   = x + self.pos_enc(pos)
        x   = self.transformer(x)
        return x[:, 0]


class CTRGCN(nn.Module):
    def __init__(self, num_class=NUM_CLASSES, num_point=NUM_JOINTS,
                 in_channels=IN_CHANNELS, A=None, dropout=0.5):
        super().__init__()
        A = A.astype(np.float32)
        self.data_bn = nn.BatchNorm1d(in_channels * num_point)
        self.blocks  = nn.ModuleList([
            CTRGCNBlock( in_channels,  64, A, residual=False),
            CTRGCNBlock( 64,  64, A), CTRGCNBlock( 64,  64, A),
            CTRGCNBlock( 64,  64, A),
            CTRGCNBlock( 64, 128, A, stride=2),
            CTRGCNBlock(128, 128, A), CTRGCNBlock(128, 128, A),
            CTRGCNBlock(128, 256, A, stride=2),
            CTRGCNBlock(256, 256, A), CTRGCNBlock(256, 256, A),
        ])
        self.temp_attn = TemporalTransformerHead(d_model=256, nhead=4, num_layers=2)
        self.drop      = nn.Dropout(p=dropout)
        self.fc        = nn.Linear(256, num_class)

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0,4,3,1,2).contiguous().reshape(N*M, V*C, T)
        x = self.data_bn(x)
        x = x.reshape(N,M,V,C,T).permute(0,1,3,4,2).contiguous().reshape(N*M,C,T,V)
        for blk in self.blocks:
            x = blk(x)
        x = self.temp_attn(x)
        x = self.drop(x)
        x = x.reshape(N, M, -1).mean(1)
        return self.fc(x)



def get_bone_stream(data: torch.Tensor, pairs) -> torch.Tensor:
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
#  ARABIC NLP PIPELINE
# ══════════════════════════════════════════════════════════════════
class ArabicGrammarRules:
    FEMININE_NOUNS = {
        'أم','أخت','بنت','امرأة','مرأة','فتاة','شابة','جدة','عمة','خالة',
        'ابنة','زوجة','بنات','نساء','سيدة','عروس','أرملة','مطلقة',
        'ممرضة','طبيبة','معلمة','مديرة','موظفة','مهندسة','محامية',
        'صيدلانية','أستاذة','مدربة','سكرتيرة','عاملة','فنانة',
        'عين','أذن','يد','كف','ساق','ركبة','قدم','رجل','رئة','كلية',
        'شمس','أرض','نار','ريح','روح','بئر','دار','نفس','حرب',
        'مصر','سوريا','تونس','ليبيا','السودان',
    }
    KNOWN_VERBS = {
        'يأكل','يشرب','ينام','يستيقظ','يلبس','يغسل','يستحم','يمشط',
        'يطبخ','يلعب','يركض','يمشي','يجري','يسبح','يركب','يقود',
        'يسافر','يرجع','يتكلم','يقول','يسمع','يقرأ','يكتب','يشرح',
        'يسأل','يجاوب','يعلم','يدرس','يحفظ','يفهم','يتعلم','يعالج',
        'يفحص','يشتري','يبيع','يدفع','يأخذ','يعطي','يحضر','يجلب',
        'يرسل','يطلب','يستلم','يقدم','يساعد','يفتح','يغلق','يرى',
        'يصعد','ينزل','يدخل','يخرج','يذهب','يأتي','يعود','يقف',
        'يجلس','يحب','يكره','يخاف','يفرح','يحزن','يغضب','يبكي',
        'يضحك','يرسم','يحسب','يحل','يجيب','يشارك',
    }
    PLACE_WORDS = {
        'سوق','مستشفى','مدرسة','جامعة','بيت','منزل','مسجد','كنيسة',
        'مطعم','صيدلية','حديقة','شارع','غرفة','مطبخ','حمام','صف',
        'فصل','ملعب','مزرعة','مكتب','محل','دكان','مصنع','معمل',
    }
    TIME_WORDS = {
        'صباح','مساء','ظهر','ليل','نهار','عصر','فجر','يوم','أسبوع',
        'شهر','سنة','عام','صيف','شتاء','ربيع','خريف','أمس','اليوم','غد',
    }
    EVERY_WORDS = {'يوم','أسبوع','شهر','سنة','عام'}
    BODY_PARTS  = {
        'شعر','رأس','وجه','جبهة','خد','أنف','فم','لسان','أسنان',
        'عين','أذن','رقبة','كتف','ذراع','يد','كف','إصبع','صدر',
        'بطن','ظهر','خصر','ورك','رجل','ركبة','ساق','قدم','قلب',
    }
    ADJECTIVES = {
        'كبير','صغير','طويل','قصير','جميل','سريع','بطيء','ذكي',
        'قوي','ضعيف','صحي','مريض','نظيف','غني','فقير','سعيد',
        'حزين','جديد','قديم','ثقيل','خفيف','حار','بارد','حلو',
    }
    FROM_PLACE_VERBS = {
        'يشتري','تشتري','يأخذ','تأخذ','يجلب','تجلب','يحضر','تحضر',
        'يطلب','تطلب','يستلم','تستلم','يخرج','تخرج','يعود','تعود',
    }
    TO_PLACE_VERBS = {
        'يذهب','تذهب','يسافر','تسافر','يرسل','ترسل','يصل','تصل',
    }
    INDIRECT_OBJ_VERBS = {
        'يكتب','تكتب','يعطي','تعطي','يقدم','تقدم','يرسل','ترسل',
        'يشرح','تشرح','يقول','تقول','يخبر','تخبر','يعلم','تعلم',
        'يبيع','تبيع','يهدي','تهدي',
    }

    @staticmethod
    def clean_word(word): return word.split('/')[0].strip()

    @classmethod
    def get_core(cls, word):
        w = cls.clean_word(word)
        return w[2:] if w.startswith('ال') else w

    @classmethod
    def is_feminine(cls, word):
        core = cls.get_core(word)
        if core in cls.FEMININE_NOUNS: return True
        if core.endswith('ة'): return True
        if core.endswith('ات') and len(core) > 3: return True
        if core.endswith('اء') and len(core) > 3: return True
        return False

    @classmethod
    def get_gender(cls, word): return 'f' if cls.is_feminine(word) else 'm'

    @classmethod
    def feminize_verb(cls, verb):
        verb = cls.clean_word(verb)
        return ('ت' + verb[1:]) if verb.startswith('ي') else verb

    @classmethod
    def conjugate_verb(cls, verb, gender, number='singular'):
        verb = cls.clean_word(verb)
        base = ('ي' + verb[1:]) if verb.startswith('ت') and len(verb) > 2 else verb
        if gender == 'f':
            fem = 'ت' + base[1:]
            return fem + 'ن' if number == 'plural' else fem
        return base + 'ون' if number == 'plural' else base

    @classmethod
    def agree_adjective(cls, adj, gender):
        adj = cls.clean_word(adj)
        if gender == 'm' or adj.endswith('ة'): return adj
        if adj.startswith('أ') and len(adj) == 4: return adj[1:] + 'اء'
        if adj.endswith('ان') and len(adj) > 3: return adj[:-2] + 'ى'
        return adj + 'ة'

    @classmethod
    def add_article(cls, word):
        w = cls.clean_word(word)
        return w if w.startswith('ال') else 'ال' + w

    @classmethod
    def attach_preposition(cls, prep, word):
        with_art = cls.add_article(word)
        if prep in ('لـ','ل'): return 'لل' + with_art[2:]
        if prep in ('بـ','ب'): return 'بال' + with_art[2:]
        return prep + with_art

    @classmethod
    def possessive(cls, word, gender):
        w = cls.clean_word(word)
        return w + ('ها' if gender == 'f' else 'ه')

    @classmethod
    def get_preposition(cls, place, verb):
        verb_m = ('ي' + verb[1:]) if verb.startswith('ت') else verb
        if verb_m in cls.FROM_PLACE_VERBS: return 'من'
        if verb_m in cls.TO_PLACE_VERBS:   return 'إلى'
        return 'في'

    @classmethod
    def classify_word(cls, word):
        core = cls.get_core(word)
        if core in cls.KNOWN_VERBS:  return 'verb'
        if core in cls.PLACE_WORDS:  return 'place'
        if core in cls.TIME_WORDS:   return 'time'
        if core in cls.ADJECTIVES:   return 'adj'
        return 'noun'


class WordSelector:
    def __init__(self, vocabulary: set):
        self.vocab = vocabulary

    def select(self, predictions):
        selected = []
        for top_k in predictions:
            for word, conf in top_k:
                if word in self.vocab:
                    selected.append(word)
                    break
            else:
                if top_k:
                    selected.append(top_k[0][0])
        return selected


class NLPPipeline:
    def __init__(self, vocabulary: set, mode: str = 'rules'):
        self.grammar  = ArabicGrammarRules()
        self.selector = WordSelector(vocabulary)
        self.mode     = mode
        self.groq_client = None
        if mode == 'hybrid' and GROQ_AVAILABLE:
            try:
                self.groq_client = Groq(api_key=GROQ_API_KEY)
            except Exception:
                self.mode = 'rules'

    def _build_sentence(self, words, hints):
        subject, verb, direct_obj = None, None, None
        indirect_objs, adjectives, places, times = [], [], [], []
        g = 'f' if hints.get('is_feminine') else 'm'

        for w in words:
            wtype = self.grammar.classify_word(w)
            if wtype == 'verb' and verb is None:       verb = w
            elif wtype == 'place':                     places.append(w)
            elif wtype == 'time':                      times.append(w)
            elif wtype == 'adj':                       adjectives.append(w)
            elif wtype == 'noun':
                if subject is None:                    subject = w
                elif direct_obj is None:               direct_obj = w
                else:                                  indirect_objs.append(w)

        parts = []
        if subject:    parts.append(self.grammar.add_article(subject))
        if verb:       parts.append(self.grammar.conjugate_verb(verb, g))
        if direct_obj:
            core = self.grammar.get_core(direct_obj)
            if core in self.grammar.BODY_PARTS:
                parts.append(self.grammar.possessive(direct_obj, g))
            else:
                parts.append(self.grammar.add_article(direct_obj))

        verb_m = ('ي' + verb[1:]) if verb and verb.startswith('ت') and len(verb) >= 4 else (verb or '')
        for ind in indirect_objs:
            core = self.grammar.get_core(ind)
            if core in self.grammar.BODY_PARTS:
                parts.append(self.grammar.possessive(ind, g))
            elif verb_m in self.grammar.INDIRECT_OBJ_VERBS:
                parts.append(self.grammar.attach_preposition('لـ', ind))
            else:
                parts.append(self.grammar.add_article(ind))

        for adj in adjectives:
            parts.append(self.grammar.add_article(self.grammar.agree_adjective(adj, g)))
        for place in places:
            prep = self.grammar.get_preposition(place, verb or '')
            parts.append(self.grammar.attach_preposition(prep, place))
        for t in times:
            core = self.grammar.get_core(t)
            if core in self.grammar.EVERY_WORDS: parts.append('كل ' + core)
            else: parts.append(self.grammar.attach_preposition('في', t))

        return ' '.join(parts)

    def run(self, predictions):
        selected = self.selector.select(predictions)
        hints = {'subject': None, 'is_feminine': False, 'verbs': []}
        for w in selected:
            wtype = self.grammar.classify_word(w)
            if wtype == 'noun' and hints['subject'] is None:
                hints['subject']     = w
                hints['is_feminine'] = self.grammar.is_feminine(w)
            elif wtype == 'verb':
                hints['verbs'].append(w)

        processed = []
        for w in selected:
            if self.grammar.classify_word(w) == 'verb' and hints['is_feminine']:
                w = self.grammar.feminize_verb(w)
            processed.append(w)

        if self.mode == 'hybrid' and self.groq_client:
            try:
                gloss_str   = ' '.join(processed)
                gender_note = f'الفاعل ({hints["subject"]}) مؤنث.' if hints['is_feminine'] else ''
                prompt      = f'الكلمات: {gloss_str}\n{gender_note}\nالجملة:'
                resp = self.groq_client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[
                        {'role': 'system', 'content':
                         'أنت محول نصوص للغة العربية. حوّل الكلمات المفتاحية إلى جملة عربية سليمة. '
                         'استخدم جميع الكلمات المعطاة بدون إضافة كلمات محتوى جديدة. '
                         'أعد الجملة النهائية فقط.'},
                        {'role': 'user', 'content': prompt},
                    ],
                    temperature=0.1, max_tokens=80,
                )
                sentence = resp.choices[0].message.content.strip()
                sentence = re.sub(r'^(الجملة|النتيجة)[:\s]+', '', sentence).strip()
                return {'words': selected, 'processed': processed,
                        'sentence': sentence, 'method': 'groq'}
            except Exception as e:
                print(f'[NLP] Groq error: {e}')

        sentence = self._build_sentence(processed, hints)
        return {'words': selected, 'processed': processed,
                'sentence': sentence, 'method': 'rules'}


class LandmarkExtractor:
    def __init__(self, model_complexity=2):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def extract(self, frame_bgr):
        
        rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.holistic.process(rgb)

        if result.pose_landmarks is None:
            return None, result, False

        
        face = (np.array([[lm.x, lm.y, lm.z, 0.0] for lm in result.face_landmarks.landmark], np.float32)
                if result.face_landmarks else np.zeros((468, 4), np.float32))
        pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in result.pose_landmarks.landmark], np.float32)
        lh   = (np.array([[lm.x, lm.y, lm.z, 0.0] for lm in result.left_hand_landmarks.landmark], np.float32)
                if result.left_hand_landmarks else np.zeros((21, 4), np.float32))
        rh   = (np.array([[lm.x, lm.y, lm.z, 0.0] for lm in result.right_hand_landmarks.landmark], np.float32)
                if result.right_hand_landmarks else np.zeros((21, 4), np.float32))

        all_543 = np.vstack([face, pose, lh, rh])   # (543, 4)


        left_hip  = all_543[POSE_START + 23, :3]
        right_hip = all_543[POSE_START + 24, :3]
        mid_hip   = (left_hip + right_hip) / 2.0
        all_543[:, :3] -= mid_hip

    
        joints = np.vstack([
            all_543[:N_FACE_SELECT, :3],          # face[0:33]
            all_543[LH_START:LH_START+21, :3],    # left hand
            all_543[RH_START:RH_START+21, :3],    # right hand
        ]).astype(np.float32)                      # (75, 3)

        has_hands = (result.left_hand_landmarks is not None or
                     result.right_hand_landmarks is not None)
        return joints, result, has_hands

    def close(self):
        self.holistic.close()


# ══════════════════════════════════════════════════════════════════
#  REAL-TIME PREDICTOR
# ══════════════════════════════════════════════════════════════════
class RealTimePredictor:
    def __init__(self, model_j, model_b, model_m, label_map, bone_pairs,
                 temperature, device,
                 window=WINDOW_FRAMES, stride=STRIDE, threshold=THRESHOLD):
        self.mj          = model_j.eval()
        self.mb          = model_b.eval()
        self.mm          = model_m.eval()
        self.label_map   = label_map
        self.bone_pairs  = bone_pairs
        self.temperature = temperature   
        self.device      = device
        self.window      = window
        self.stride      = stride
        self.threshold   = threshold
        self._buf        = collections.deque(maxlen=window)
        self._tick       = 0
        self._last_result       = (None, 0.0, [])
        self._cooldown          = 0
        self._COOLDOWN_FRAMES   = window // 2
        self._no_hand_count     = 0
        self._NO_HAND_RESET     = 12

        # ── Motion-based end-of-sign detection ───────────────────

        self._prev_joints       = None
        self._still_count       = 0
        self._STILL_THRESH      = 6    
        self._MOTION_THRESH     = 0.004  
        self._in_sign           = False  
        self._MIN_SIGN_FRAMES   = 16   

    @classmethod
    def from_checkpoint(cls, checkpoint_path, label_map, device=None, **kw):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        A = build_karsl_adjacency()   # (3, 75, 75)

        ck = torch.load(checkpoint_path, map_location=device, weights_only=False)

    
        bone_pairs = ck.get("bone_pairs", GRAPH_PAIRS)
        temperature = ck.get("temperature", 1.0)

        key_j = "swa_j" if "swa_j" in ck else "model_j"
        key_b = "swa_b" if "swa_b" in ck else "model_b"
        key_m = "swa_m" if "swa_m" in ck else "model_m"

        def load_model(key):
            m   = CTRGCN(A=A).to(device)
            res = m.load_state_dict(ck[key], strict=False)
            if res.missing_keys:
                print(f"  [WARN] {key} missing: {res.missing_keys[:3]}")
            if res.unexpected_keys:
                print(f"  [WARN] {key} unexpected: {res.unexpected_keys[:3]}")
            return m.eval()

        mj = load_model(key_j)
        mb = load_model(key_b)
        mm = load_model(key_m)

        print(f"[Predictor] epoch={ck.get('epoch','?')}  "
              f"best_acc={ck.get('best_acc',0):.2f}%  "
              f"temperature={temperature:.4f}  device={device}")
        print(f"[Predictor] bone_pairs={len(bone_pairs)}  "
              f"window={kw.get('window',WINDOW_FRAMES)}  "
              f"threshold={kw.get('threshold',THRESHOLD)}")
        return cls(mj, mb, mm, label_map, bone_pairs, temperature, device, **kw)

    def reset(self):
        self._buf.clear()
        self._tick          = 0
        self._last_result   = (None, 0.0, [])
        self._cooldown      = 0
        self._no_hand_count = 0

    def _motion_energy(self, joints: np.ndarray) -> float:
       
        if self._prev_joints is None:
            return 1.0
        diff = np.abs(joints[33:] - self._prev_joints[33:])
        return float(diff.mean())

    def step(self, landmarks_75x3: np.ndarray, has_hands: bool = True):
        lm = landmarks_75x3.astype(np.float32)
        motion = self._motion_energy(lm)
        self._prev_joints = lm.copy()
        self._buf.append(lm)
        self._tick += 1

        # ── cooldown  ─────────────────────────
        if self._cooldown > 0:
            self._cooldown -= 1
            return self._last_result

        # ── no-hand reset ─────────────────────────────────────────
        if not has_hands:
            self._no_hand_count += 1
            if self._no_hand_count >= self._NO_HAND_RESET:
                self._buf.clear()
                self._tick          = 0
                self._in_sign       = False
                self._still_count   = 0
                self._last_result   = (None, 0.0, [])
                self._no_hand_count = 0
            return self._last_result
        else:
            self._no_hand_count = 0

        # ── motion segmentation ───────────────────────────────────
        if motion > self._MOTION_THRESH:
           
            self._in_sign     = True
            self._still_count = 0
        else:
           
            if self._in_sign:
                self._still_count += 1


        trigger_by_still = (self._in_sign and
                            self._still_count >= self._STILL_THRESH and
                            len(self._buf) >= self._MIN_SIGN_FRAMES)

        trigger_by_full  = (len(self._buf) >= self.window and
                            self._in_sign and
                            self._tick % self.stride == 0)

        if not (trigger_by_still or trigger_by_full):
            return self._last_result if len(self._buf) >= 4 else (None, 0.0, [])

        buf_list = list(self._buf)
        seq = np.stack(buf_list)   # (T_actual, 75, 3)
        T_actual = seq.shape[0]

        if T_actual < self.window:
            seq_t = (torch.from_numpy(seq)
                     .permute(2,0,1).unsqueeze(0).float())   # (1,3,T,75)
            seq_t = F.interpolate(seq_t,
                                  size=(self.window, NUM_JOINTS),
                                  mode='bilinear', align_corners=False)
            seq = seq_t.squeeze(0).permute(1,2,0).numpy()    # (32,75,3)

        x = (torch.from_numpy(seq)
             .permute(2,0,1).unsqueeze(0).unsqueeze(-1)
             .float().to(self.device))   # (1,3,32,75,1)

        with torch.no_grad():
            xb    = get_bone_stream(x, self.bone_pairs)
            xm    = get_motion_stream(x)
            logits = (WEIGHT_J * self.mj(x)
                    + WEIGHT_B * self.mb(xb)
                    + WEIGHT_M * self.mm(xm))
            probs  = torch.softmax(logits / self.temperature, dim=1)[0]

        conf = probs.max().item()
        top_vals, top_idxs = probs.topk(TOP_K_DISPLAY)
        top_k = [(self.label_map.get(idx.item(), str(idx.item())), val.item())
                 for idx, val in zip(top_idxs, top_vals)]

        # reset state
        self._in_sign     = False
        self._still_count = 0
        self._buf.clear()
        self._tick = 0

        if conf < self.threshold:
            self._last_result = (None, conf, top_k)
        else:
            name = top_k[0][0]
            print(f"[Sign] {name}  conf={conf*100:.1f}%  frames={T_actual}")
            self._last_result = (name, conf, top_k)
            self._cooldown    = self._COOLDOWN_FRAMES

        return self._last_result


# ══════════════════════════════════════════════════════════════════
#  VISUALIZATION
# ══════════════════════════════════════════════════════════════════
def ar(text):
    if not ARABIC_SUPPORT: return text
    return get_display(reshape(text))


def draw_overlay(frame, sign_name, conf, top_k,
                 buf_len, window, collected_signs, generated_sentence):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    bar_w   = int((buf_len / window) * (w - 40))
    bar_col = (0, 200, 0) if buf_len >= window else (0, 140, 255)
    cv2.rectangle(overlay, (20, h-30), (20+bar_w, h-10), bar_col, -1)
    cv2.rectangle(overlay, (20, h-30), (w-20,    h-10), (200,200,200), 1)

    if sign_name:
        cv2.rectangle(overlay, (10,10), (w-10, 70), (0,100,0), -1)
        cv2.rectangle(overlay, (10,10), (w-10, 70), (0,255,0), 2)
    else:
        cv2.rectangle(overlay, (10,10), (w-10, 70), (30,30,30), -1)
    if top_k:
        cv2.rectangle(overlay, (10,80), (340, 80+len(top_k)*28+10), (20,20,20), -1)

    frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(img_pil)
    try:
        font_main  = ImageFont.truetype("arial.ttf", 34)
        font_sub   = ImageFont.truetype("arial.ttf", 21)
        font_small = ImageFont.truetype("arial.ttf", 15)
    except Exception:
        font_main = font_sub = font_small = ImageFont.load_default()

    if sign_name:
        draw.text((20, 18), ar(sign_name), font=font_main, fill=(255,255,255))
        draw.text((w-130, 22), f"{conf*100:.1f}%", font=font_main, fill=(180,255,0))
    else:
        draw.text((20, 18), "Waiting...", font=font_main, fill=(150,150,150))

    for i, (name, p) in enumerate(top_k):
        col = (0,255,120) if i == 0 else (180,180,180)
        draw.text((18,  88+i*28), f"{i+1}. {ar(name)}", font=font_sub, fill=col)
        draw.text((260, 88+i*28), f"{p*100:.1f}%",      font=font_sub, fill=col)

    if collected_signs:
        signs_str = "  ←  ".join([ar(s[0][0]) for s in collected_signs])
        draw.text((15, h-62), f"📝 {signs_str}", font=font_small, fill=(255,180,50))

    if generated_sentence:
        draw.text((15, h-97), ar(generated_sentence), font=font_sub, fill=(100,220,255))

    draw.text((25, h-28), f"Buffer: {buf_len}/{window}", font=font_small, fill=(255,255,255))
    controls = "SPACE=جملة  R=Reset  C=امسح  Q=خروج"
    draw.text((w//2-120, h-28), controls, font=font_small, fill=(160,160,160))

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def draw_skeleton(frame, result):
    mp_drawing  = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    if result is None: return
    if result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, result.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0,180,255), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(0,255,180), thickness=2))
    if result.left_hand_landmarks:
        mp_drawing.draw_landmarks(frame, result.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255,100,0), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(255,180,0), thickness=2))
    if result.right_hand_landmarks:
        mp_drawing.draw_landmarks(frame, result.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0,100,255), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(0,180,255), thickness=2))
    if result.face_landmarks:
        h2, w2 = frame.shape[:2]
        for i, lm in enumerate(result.face_landmarks.landmark):
            if i < N_FACE_SELECT:
                cv2.circle(frame, (int(lm.x*w2), int(lm.y*h2)), 2, (200,200,0), -1)


# ══════════════════════════════════════════════════════════════════
#  MAIN CAMERA LOOP
# ══════════════════════════════════════════════════════════════════
def run_camera(checkpoint_path, labels_path, camera_id=0,
               model_complexity=2, threshold=THRESHOLD):

    print("Loading labels...")
    df = pd.read_excel(labels_path)
    df.columns = df.columns.str.strip()
    name_col = next((c for c in df.columns if c.lower() in
                     ["signname","sign_name","sign-arabic","arabic","label","name"]), df.columns[-2])
    id_col   = next((c for c in df.columns if c.lower() in
                     ["signid","sign_id","classindex","id"]), df.columns[0])
    label_map = {int(row[id_col])-1: str(row[name_col]) for _, row in df.iterrows()}
    VOCABULARY = set(label_map.values())
    print(f"  {len(label_map)} labels loaded")

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = RealTimePredictor.from_checkpoint(
        checkpoint_path, label_map, device=device,
        window=WINDOW_FRAMES, threshold=threshold
    )

    nlp = NLPPipeline(VOCABULARY, mode='rules')
    print(f"[NLP] Pipeline ready — mode={nlp.mode}")

    extractor = LandmarkExtractor(model_complexity=model_complexity)
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera id={camera_id}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("\n" + "="*55)
    print("  CTR-GCN v6 + NLP — KARSL-502")
    print(f"  Window={WINDOW_FRAMES}  Threshold={threshold}")
    print("  SPACE → جملة  |  R → Reset  |  C → امسح  |  Q → خروج")
    print("="*55 + "\n")

    fps_buf            = collections.deque(maxlen=30)
    sign_name          = None
    conf               = 0.0
    top_k              = []
    result             = None
    collected_signs    = []
    last_sign_added    = None
    generated_sentence = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed.")
            break

        t_frame = time.perf_counter()
        joints, result, has_hands = extractor.extract(frame)
        draw_skeleton(frame, result)

        if joints is not None:
            sign_name, conf, top_k = predictor.step(joints, has_hands)
            if sign_name is not None and sign_name != last_sign_added:
                collected_signs.append(top_k)
                last_sign_added = sign_name
                print(f"[Collect] {sign_name} — total: {len(collected_signs)}")

        buf_len = len(predictor._buf)
        fps_buf.append(time.perf_counter() - t_frame)
        fps = 1.0 / (sum(fps_buf)/len(fps_buf)) if fps_buf else 0.0

        frame = draw_overlay(frame, sign_name, conf, top_k,
                             buf_len, WINDOW_FRAMES,
                             collected_signs, generated_sentence)

        cv2.putText(frame, f"FPS:{fps:.0f}  Signs:{len(collected_signs)}",
                    (frame.shape[1]-180, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        if joints is None:
            cv2.putText(frame, "No pose detected",
                        (20, frame.shape[0]-50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 2)

        cv2.imshow("KARSL-502 | CTR-GCN v6 + NLP", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord('r'):
            predictor.reset()
            collected_signs.clear()
            last_sign_added    = None
            generated_sentence = ""
            sign_name, conf, top_k = None, 0.0, []
            print("[Reset] Buffer + signs cleared")
        elif key == ord('c'):
            collected_signs.clear()
            last_sign_added    = None
            generated_sentence = ""
            print("[Clear] Signs cleared")
        elif key == ord(' '):
            if not collected_signs:
                print("[NLP] No signs collected!")
            else:
                print(f"\n[NLP] Generating from {len(collected_signs)} signs...")
                res_nlp = nlp.run(collected_signs)
                generated_sentence = res_nlp['sentence']
                print(f"[NLP] Words   : {' '.join(res_nlp['words'])}")
                print(f"[NLP] Sentence: {generated_sentence}")
                print(f"[NLP] Method  : {res_nlp['method']}\n")

    cap.release()
    extractor.close()
    cv2.destroyAllWindows()
    print("\nDone.")


# ══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default=r"C:\KARSL-502\core\output\checkpoints_v6\best_model.pth")
    parser.add_argument("--labels", type=str,
                        default=r"C:\KARSL-502\core\src\trial_final\KARSL-502_Labels.xlsx")
    parser.add_argument("--camera",     type=int,   default=0)
    parser.add_argument("--complexity", type=int,   default=2, choices=[0,1,2])
    parser.add_argument("--threshold",  type=float, default=THRESHOLD)
    args = parser.parse_args()

    run_camera(
        checkpoint_path  = args.checkpoint,
        labels_path      = args.labels,
        camera_id        = args.camera,
        model_complexity = args.complexity,
        threshold        = args.threshold,
    )