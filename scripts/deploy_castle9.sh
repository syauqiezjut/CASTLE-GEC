#!/bin/bash
# deploy_castle9.sh
# Hapus checkpoint lama, upload kode terbaru, jalankan training castle9
# Jalankan dari root castle-repo: bash scripts/deploy_castle9.sh

set -e

SSH_PORT=27758
SSH_HOST="root@connect.westd.seetacloud.com"
REMOTE_DIR="/root/autodl-tmp/CASTLE-GEC"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "============================================"
echo "  CASTLE v2 — Deploy & Launch castle9"
echo "============================================"
echo ""

# ── 1. Hapus checkpoint lama ──────────────────
echo "[1/3] Membersihkan checkpoint lama (pertahankan castle7)..."
ssh -p $SSH_PORT $SSH_HOST \
  "rm -rf ${REMOTE_DIR}/checkpoints/castle \
           ${REMOTE_DIR}/checkpoints/castle2 \
           ${REMOTE_DIR}/checkpoints/castle3 \
           ${REMOTE_DIR}/checkpoints/castle4 \
           ${REMOTE_DIR}/checkpoints/castle5 \
           ${REMOTE_DIR}/checkpoints/castle8 \
  && echo 'Sisa checkpoint:' \
  && ls ${REMOTE_DIR}/checkpoints/"
echo ""

# ── 2. Upload kode terbaru ────────────────────
echo "[2/3] Upload kode terbaru ke server..."

# castle_model.py (perubahan terbesar: MLP gate, pre-softmax, beam search)
rsync -avz -e "ssh -p $SSH_PORT" \
  "${LOCAL_DIR}/src/castle_model.py" \
  "${SSH_HOST}:${REMOTE_DIR}/src/"

# train.py (hapus L_KG & L_reg, tambah clip_norm=0.1)
rsync -avz -e "ssh -p $SSH_PORT" \
  "${LOCAL_DIR}/src/train.py" \
  "${SSH_HOST}:${REMOTE_DIR}/src/"

# castle_base.yaml (attention_dropout=0.1, lambda_kg=0, beam_size=5)
rsync -avz -e "ssh -p $SSH_PORT" \
  "${LOCAL_DIR}/configs/castle_base.yaml" \
  "${SSH_HOST}:${REMOTE_DIR}/configs/"

echo "Upload selesai."
echo ""

# ── 3. Launch training castle9 ────────────────
echo "[3/3] Memulai training castle9..."
ssh -p $SSH_PORT $SSH_HOST "
  cd ${REMOTE_DIR}
  mkdir -p checkpoints/castle9 runs/castle9 logs
  nohup python src/train.py \
    --config configs/castle_base.yaml \
    --save_dir checkpoints/castle9 \
    > logs/castle9.log 2>&1 &
  echo \"Training castle9 dimulai — PID: \$!\"
  echo \"Monitor: tail -f ${REMOTE_DIR}/logs/castle9.log\"
"
echo ""
echo "============================================"
echo "  Selesai! Pantau training dengan:"
echo "  ssh -p $SSH_PORT $SSH_HOST 'tail -f ${REMOTE_DIR}/logs/castle9.log'"
echo "============================================"
