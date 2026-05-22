"""
graph_karsl75_adj.py
====================
Adjacency matrix + Graph class للـ KARSL-75 skeleton
متوافق مع:
  - 2AGCN_input.py   → shape (N, 3, 32, 75, 1)
  - graph_karsl75.py → get_bone_stream(data_batch, Graph().pairs)
  - CTR-GCN          → Graph(layout='karsl75', strategy='spatial')

تعريف الـ 75 joint:
  0–32   → MediaPipe Face-Mesh  (33 نقطة)
  33–53  → Left Hand            (21 نقطة)  [SELECTED_INDICES 501–521]
  54–74  → Right Hand           (21 نقطة)  [SELECTED_INDICES 522–542]
"""

import numpy as np

# ---------------------------------------------------------------------------
# 1. تعريف الحواف (Edges)
# ---------------------------------------------------------------------------

# ── Face: سلسلة خطية تقريبية على حافة الوجه + نقاط مركزية ──────────────
FACE_EDGES = [
    # حافة الوجه (jaw line + forehead approximation)
    (0, 1),  (1, 2),  (2, 3),  (3, 4),  (4, 5),
    (5, 6),  (6, 7),  (7, 8),  (8, 9),  (9, 10),
    (10, 11),(11, 12),(12, 13),(13, 14),(14, 15),
    (15, 16),(16, 17),(17, 18),(18, 19),(19, 20),
    (20, 21),(21, 22),(22, 23),(23, 24),(24, 25),
    (25, 26),(26, 27),(27, 28),(28, 29),(29, 30),
    (30, 31),(31, 32),
    # إغلاق الحلقة (أسفل الذقن → الجانب الآخر)
    (0, 32),
    # وصل النقاط المركزية (أنف، عيون) بأقرب نقاط
    (0, 4),   # nose tip → right cheek
    (0, 28),  # nose tip → left cheek
    (0, 16),  # nose tip → chin center
]

# ── Left Hand (joints 33–53): topology مثل MediaPipe hand ─────────────────
# wrist=33, thumb=34-37, index=38-41, middle=42-45, ring=46-49, pinky=50-53
def _hand_edges(base):
    """يولد حواف اليد بناءً على الـ base index للـ wrist."""
    wrist = base
    edges = []
    finger_bases = [base+1, base+5, base+9, base+13, base+17]  # MCP joints
    # wrist → كل أصبع MCP
    for fb in finger_bases:
        edges.append((wrist, fb))
    # داخل كل أصبع (4 مفاصل)
    for fb in finger_bases:
        for i in range(3):
            edges.append((fb + i, fb + i + 1))
    return edges

LH_EDGES = _hand_edges(33)   # Left Hand  joints 33–53
RH_EDGES = _hand_edges(54)   # Right Hand joints 54–74

# ── وصل اليدين بالوجه (cross-body connections) ───────────────────────────
#
# تعريف النقاط المهمة:
#   LH: wrist=33, thumb_tip=37, index_tip=41, middle_tip=45, ring_tip=49, pinky_tip=53
#   RH: wrist=54, thumb_tip=58, index_tip=62, middle_tip=66, ring_tip=70, pinky_tip=74
#
# Face zones المهمة للـ sign language:
#   nose_tip=0, chin_center=16, mouth_left=4, mouth_right=28
#   left_eye_center=23, right_eye_center=9
#
# المنطق: في sign language الأصابع بتقرب من مناطق معينة في الوجه —
# الـ graph لازم يعكس العلاقات دي عشان الموديل يتعلمها.

CROSS_EDGES = [
    # ── Level 1: Wrist anchors (موجودة قبل، ضرورية) ──────────────────
    (0,  33),   # nose_tip ↔ LH wrist
    (0,  54),   # nose_tip ↔ RH wrist
    (33, 54),   # LH wrist ↔ RH wrist  (inter-hand)

    # ── Level 2: Index fingertip ↔ face zones ─────────────────────────
    # الـ index هو الأكثر استخداماً في الإشارة للوجه
    (41, 0),    # LH index_tip ↔ nose_tip
    (41, 4),    # LH index_tip ↔ mouth_right
    (41, 16),   # LH index_tip ↔ chin_center
    (62, 0),    # RH index_tip ↔ nose_tip
    (62, 28),   # RH index_tip ↔ mouth_left
    (62, 16),   # RH index_tip ↔ chin_center

    # ── Level 3: Thumb tip ↔ face zones ──────────────────────────────
    # الإبهام بيظهر كتير في إشارات الذقن والخد
    (37, 16),   # LH thumb_tip ↔ chin_center
    (37, 4),    # LH thumb_tip ↔ mouth_right
    (58, 16),   # RH thumb_tip ↔ chin_center
    (58, 28),   # RH thumb_tip ↔ mouth_left

    # ── Level 4: Eye zone connections ─────────────────────────────────
    # إشارات العين والجبهة
    (41, 9),    # LH index_tip ↔ right_eye_center
    (62, 23),   # RH index_tip ↔ left_eye_center

    # ── Level 5: Inter-finger connections ─────────────────────────────
    # وصل أطراف الأصابع ببعض (LH ↔ RH fingertips)
    # مهم لإشارات فيها تلاقي اليدين زي "مع" و"ضد"
    (41, 62),   # LH index_tip  ↔ RH index_tip
    (37, 58),   # LH thumb_tip  ↔ RH thumb_tip
    (45, 66),   # LH middle_tip ↔ RH middle_tip
]

# كل الحواف مجمعة
ALL_EDGES = FACE_EDGES + LH_EDGES + RH_EDGES + CROSS_EDGES

# pairs مرتبة للـ Bone Stream (نفس الترتيب اللي بتستخدمه get_bone_stream)
GRAPH_PAIRS = ALL_EDGES  # list of (v1, v2)

NUM_JOINTS = 75

# ---------------------------------------------------------------------------
# 2. Graph Class (CTR-GCN compatible)
# ---------------------------------------------------------------------------

class Graph:
    """
    بناء الـ adjacency matrix للـ KARSL-75 skeleton.

    Parameters
    ----------
    strategy : str
        'uniform'   → A واحد بـ weight موحد
        'distance'  → تقسيم حسب المسافة (distance partitioning)
        'spatial'   → spatial configuration partitioning (الأفضل لـ CTR-GCN)
    max_hop  : int  → أقصى مسافة hop تُحسب
    dilation : int  → step للـ hops

    Attributes
    ----------
    A          : np.ndarray  shape (num_strategies, V, V)
    pairs      : list of (int, int)  → يُستخدم في get_bone_stream
    """

    def __init__(self, strategy='spatial', max_hop=1, dilation=1):
        self.num_joints = NUM_JOINTS
        self.strategy   = strategy
        self.max_hop    = max_hop
        self.dilation   = dilation
        self.pairs      = GRAPH_PAIRS

        self.hop_dis = self._get_hop_distance()
        self.A       = self._get_adjacency()

    # ------------------------------------------------------------------
    def _get_hop_distance(self):
        """BFS لحساب المسافة بين كل جفتين من الـ joints."""
        graph = {i: [] for i in range(self.num_joints)}
        for v1, v2 in ALL_EDGES:
            graph[v1].append(v2)
            graph[v2].append(v1)

        hop_dis = np.full((self.num_joints, self.num_joints), np.inf)
        np.fill_diagonal(hop_dis, 0)

        for root in range(self.num_joints):
            queue = [root]
            visited = {root}
            dist = {root: 0}
            while queue:
                node = queue.pop(0)
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        dist[neighbor] = dist[node] + 1
                        hop_dis[root, neighbor] = dist[neighbor]
                        queue.append(neighbor)
        return hop_dis

    # ------------------------------------------------------------------
    def _normalize_adj(self, A):
        """D^{-1/2} A D^{-1/2} normalization."""
        Dl = np.sum(A, axis=0)
        Dn = np.zeros_like(Dl)
        mask = Dl > 0
        Dn[mask] = Dl[mask] ** (-1.0)
        Dn = np.diag(Dn)
        return Dn @ A @ Dn

    # ------------------------------------------------------------------
    def _get_adjacency(self):
        valid_hops = range(0, self.max_hop + 1, self.dilation)

        if self.strategy == 'uniform':
            A = np.zeros((1, self.num_joints, self.num_joints))
            for hop in valid_hops:
                A[0][self.hop_dis == hop] = 1
            A[0] = self._normalize_adj(A[0])
            return A

        elif self.strategy == 'distance':
            A = np.zeros((len(valid_hops), self.num_joints, self.num_joints))
            for i, hop in enumerate(valid_hops):
                A[i][self.hop_dis == hop] = 1
                A[i] = self._normalize_adj(A[i])
            return A

        elif self.strategy == 'spatial':
            # Spatial Configuration Partitioning (كما في ST-GCN / CTR-GCN)
            # 3 subsets: self, centripetal (قريب من المركز), centrifugal (بعيد عنه)
            #
            # المركز هنا = joint 0 (nose tip) — أقرب نقطة ثابتة نسبياً في الجسم
            # للـ 75-joint layout ده (وجه + يدين بدون جسم)

            A = np.zeros((3, self.num_joints, self.num_joints))

            for hop in valid_hops:
                for i in range(self.num_joints):
                    for j in range(self.num_joints):
                        if self.hop_dis[i, j] != hop:
                            continue
                        if i == j:
                            A[0, i, j] = 1          # self-connection
                        elif self.hop_dis[0, j] <= self.hop_dis[0, i]:
                            A[1, i, j] = 1          # centripetal (j أقرب للمركز)
                        else:
                            A[2, i, j] = 1          # centrifugal (j أبعد عن المركز)

            for k in range(3):
                A[k] = self._normalize_adj(A[k])
            return A

        else:
            raise ValueError(f"Unknown strategy: {self.strategy!r}. "
                             f"Choose from 'uniform', 'distance', 'spatial'.")

    # ------------------------------------------------------------------
    def __repr__(self):
        return (f"Graph(strategy='{self.strategy}', "
                f"num_joints={self.num_joints}, "
                f"A.shape={self.A.shape})")


# ---------------------------------------------------------------------------
# 3. Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 55)
    print("KARSL-75 Graph — Smoke Test")
    print("=" * 55)

    for strat in ('uniform', 'distance', 'spatial'):
        g = Graph(strategy=strat)
        print(f"\nStrategy : {strat}")
        print(f"  A.shape : {g.A.shape}")
        print(f"  A[0] min/max : {g.A[0].min():.4f} / {g.A[0].max():.4f}")
        assert not np.any(np.isnan(g.A)),  "❌ NaN detected in A!"
        assert not np.any(np.isinf(g.A)),  "❌ Inf detected in A!"
        print(f"  ✅ No NaN / Inf")

    g = Graph(strategy='spatial')
    print(f"\nTotal edges defined : {len(g.pairs)}")
    print(f"  Face edges  : {len(FACE_EDGES)}")
    print(f"  LH edges    : {len(LH_EDGES)}")
    print(f"  RH edges    : {len(RH_EDGES)}")
    print(f"  Cross edges : {len(CROSS_EDGES)}")

    # تأكد إن get_bone_stream شغالة
    import torch
    dummy = torch.randn(2, 3, 32, 75, 1)
    from graph_karsl75 import get_bone_stream
    try:
        bone = get_bone_stream(dummy, g.pairs)
        assert bone.shape == dummy.shape, "Shape mismatch in bone stream!"
        print(f"\n✅ get_bone_stream OK — bone.shape: {bone.shape}")
    except ImportError:
        print("\n⚠️  graph_karsl75.py not in path — skipping bone stream test.")

    print("\n✅ All checks passed.")