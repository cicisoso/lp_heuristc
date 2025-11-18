#!/usr/bin/env bash

# HL-GNN Test Script
# This script runs experiments on various datasets to validate the algorithm
# Exit codes: 0 = success, 1 = failure

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

# Log file setup
LOG_FILE="${LOG_FILE:-test_$(date +%Y%m%d_%H%M%S).log}"

# Base directories
PLANETOID_DIR="scripts/Planetoid"
OGB_DIR="scripts/OGB"

# =============================================================================
# Utility Functions
# =============================================================================

log_status() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2
}

run_experiment() {
    local dataset_name="$1"
    shift
    log_status "Starting experiment: $dataset_name"
    if "$@"; then
        log_status "✓ Completed: $dataset_name"
        return 0
    else
        log_error "✗ Failed: $dataset_name"
        return 1
    fi
}

# =============================================================================
# Main Execution
# =============================================================================

# Redirect all output to log file while also displaying on terminal
exec > >(tee -a "$LOG_FILE") 2>&1

log_status "=========================================="
log_status "HL-GNN Experiments Started"
log_status "Log file: $LOG_FILE"
log_status "=========================================="
echo ""

# =============================================================================
# Planetoid Datasets Configuration
# =============================================================================

log_status "=========================================="
log_status "Planetoid Datasets"
log_status "=========================================="

# Cora configuration
CORA_MLP_LAYERS=3
CORA_HIDDEN=8192
CORA_DROPOUT=0.5
CORA_EPOCHS=100
CORA_K=20
CORA_ALPHA=0.2
CORA_INIT=RWR

log_status ">>> Dataset: Cora"
log_status "    Description: Citation network (node classification)"
log_status "    Configuration: MLP层数=$CORA_MLP_LAYERS, 隐藏层=$CORA_HIDDEN, Dropout=$CORA_DROPOUT"
log_status "    Training: Epochs=$CORA_EPOCHS, K=$CORA_K, Alpha=$CORA_ALPHA, Init=$CORA_INIT"

run_experiment "Cora" \
    python $PLANETOID_DIR/planetoid.py \
    --dataset cora \
    --mlp_num_layers $CORA_MLP_LAYERS \
    --hidden_channels $CORA_HIDDEN \
    --dropout $CORA_DROPOUT \
    --epochs $CORA_EPOCHS \
    --K $CORA_K \
    --alpha $CORA_ALPHA \
    --init $CORA_INIT

# Citeseer configuration
CITESEER_MLP_LAYERS=2
CITESEER_HIDDEN=8192
CITESEER_DROPOUT=0.5
CITESEER_EPOCHS=100
CITESEER_K=20
CITESEER_ALPHA=0.2
CITESEER_INIT=RWR

log_status ">>> Dataset: Citeseer"
log_status "    Description: Citation network (node classification)"
log_status "    Configuration: MLP层数=$CITESEER_MLP_LAYERS, 隐藏层=$CITESEER_HIDDEN, Dropout=$CITESEER_DROPOUT"
log_status "    Training: Epochs=$CITESEER_EPOCHS, K=$CITESEER_K, Alpha=$CITESEER_ALPHA, Init=$CITESEER_INIT"

run_experiment "Citeseer" \
    python $PLANETOID_DIR/planetoid.py \
    --dataset citeseer \
    --mlp_num_layers $CITESEER_MLP_LAYERS \
    --hidden_channels $CITESEER_HIDDEN \
    --dropout $CITESEER_DROPOUT \
    --epochs $CITESEER_EPOCHS \
    --K $CITESEER_K \
    --alpha $CITESEER_ALPHA \
    --init $CITESEER_INIT

# Pubmed configuration
PUBMED_MLP_LAYERS=3
PUBMED_HIDDEN=512
PUBMED_DROPOUT=0.6
PUBMED_EPOCHS=300
PUBMED_K=20
PUBMED_ALPHA=0.2
PUBMED_INIT=KI

log_status ">>> Dataset: Pubmed"
log_status "    Description: Citation network (node classification)"
log_status "    Configuration: MLP层数=$PUBMED_MLP_LAYERS, 隐藏层=$PUBMED_HIDDEN, Dropout=$PUBMED_DROPOUT"
log_status "    Training: Epochs=$PUBMED_EPOCHS, K=$PUBMED_K, Alpha=$PUBMED_ALPHA, Init=$PUBMED_INIT"

run_experiment "Pubmed" \
    python $PLANETOID_DIR/planetoid.py \
    --dataset pubmed \
    --mlp_num_layers $PUBMED_MLP_LAYERS \
    --hidden_channels $PUBMED_HIDDEN \
    --dropout $PUBMED_DROPOUT \
    --epochs $PUBMED_EPOCHS \
    --K $PUBMED_K \
    --alpha $PUBMED_ALPHA \
    --init $PUBMED_INIT

# =============================================================================
# Amazon Datasets Configuration
# =============================================================================

log_status "=========================================="
log_status "Amazon Datasets"
log_status "=========================================="

# Amazon Photo configuration
PHOTO_MLP_LAYERS=3
PHOTO_HIDDEN=512
PHOTO_DROPOUT=0.6
PHOTO_EPOCHS=200
PHOTO_K=20
PHOTO_ALPHA=0.2
PHOTO_INIT=RWR

log_status ">>> Dataset: Amazon Photo"
log_status "    Description: Amazon co-purchase network (node classification)"
log_status "    Configuration: MLP层数=$PHOTO_MLP_LAYERS, 隐藏层=$PHOTO_HIDDEN, Dropout=$PHOTO_DROPOUT"
log_status "    Training: Epochs=$PHOTO_EPOCHS, K=$PHOTO_K, Alpha=$PHOTO_ALPHA, Init=$PHOTO_INIT"

run_experiment "Amazon Photo" \
    python $PLANETOID_DIR/amazon.py \
    --dataset photo \
    --mlp_num_layers $PHOTO_MLP_LAYERS \
    --hidden_channels $PHOTO_HIDDEN \
    --dropout $PHOTO_DROPOUT \
    --epochs $PHOTO_EPOCHS \
    --K $PHOTO_K \
    --alpha $PHOTO_ALPHA \
    --init $PHOTO_INIT

# Amazon Computers configuration
COMPUTERS_MLP_LAYERS=3
COMPUTERS_HIDDEN=512
COMPUTERS_DROPOUT=0.6
COMPUTERS_EPOCHS=200
COMPUTERS_K=20
COMPUTERS_ALPHA=0.2
COMPUTERS_INIT=RWR

log_status ">>> Dataset: Amazon Computers"
log_status "    Description: Amazon co-purchase network (node classification)"
log_status "    Configuration: MLP层数=$COMPUTERS_MLP_LAYERS, 隐藏层=$COMPUTERS_HIDDEN, Dropout=$COMPUTERS_DROPOUT"
log_status "    Training: Epochs=$COMPUTERS_EPOCHS, K=$COMPUTERS_K, Alpha=$COMPUTERS_ALPHA, Init=$COMPUTERS_INIT"

run_experiment "Amazon Computers" \
    python $PLANETOID_DIR/amazon.py \
    --dataset computers \
    --mlp_num_layers $COMPUTERS_MLP_LAYERS \
    --hidden_channels $COMPUTERS_HIDDEN \
    --dropout $COMPUTERS_DROPOUT \
    --epochs $COMPUTERS_EPOCHS \
    --K $COMPUTERS_K \
    --alpha $COMPUTERS_ALPHA \
    --init $COMPUTERS_INIT

# =============================================================================
# OGB Datasets Configuration
# =============================================================================

log_status "=========================================="
log_status "OGB Datasets"
log_status "=========================================="

# ogbl-collab configuration
COLLAB_PREDICTOR=DOT
COLLAB_USE_VALEDGES=True
COLLAB_YEAR=2010
COLLAB_EPOCHS=800
COLLAB_EVAL_LAST_BEST=True
COLLAB_DROPOUT=0.3
COLLAB_USE_NODE_FEAT=True

log_status ">>> Dataset: ogbl-collab"
log_status "    Description: Collaboration network (link prediction)"
log_status "    Configuration: Predictor=$COLLAB_PREDICTOR, Year=$COLLAB_YEAR, Dropout=$COLLAB_DROPOUT"
log_status "    Training: Epochs=$COLLAB_EPOCHS, UseNodeFeat=$COLLAB_USE_NODE_FEAT, EvalLastBest=$COLLAB_EVAL_LAST_BEST"

run_experiment "ogbl-collab" \
    python $OGB_DIR/main.py \
    --data_name ogbl-collab \
    --predictor $COLLAB_PREDICTOR \
    --use_valedges_as_input $COLLAB_USE_VALEDGES \
    --year $COLLAB_YEAR \
    --epochs $COLLAB_EPOCHS \
    --eval_last_best $COLLAB_EVAL_LAST_BEST \
    --dropout $COLLAB_DROPOUT \
    --use_node_feat $COLLAB_USE_NODE_FEAT

# ogbl-ddi configuration
DDI_EMB_HIDDEN=512
DDI_GNN_HIDDEN=512
DDI_MLP_HIDDEN=512
DDI_NUM_NEG=3
DDI_DROPOUT=0.3
DDI_LOSS_FUNC=WeightedHingeAUC

log_status ">>> Dataset: ogbl-ddi"
log_status "    Description: Drug-drug interaction network (link prediction)"
log_status "    Configuration: Emb=$DDI_EMB_HIDDEN, GNN=$DDI_GNN_HIDDEN, MLP=$DDI_MLP_HIDDEN, Dropout=$DDI_DROPOUT"
log_status "    Training: NumNeg=$DDI_NUM_NEG, LossFunc=$DDI_LOSS_FUNC"

run_experiment "ogbl-ddi" \
    python $OGB_DIR/main.py \
    --data_name ogbl-ddi \
    --emb_hidden_channels $DDI_EMB_HIDDEN \
    --gnn_hidden_channels $DDI_GNN_HIDDEN \
    --mlp_hidden_channels $DDI_MLP_HIDDEN \
    --num_neg $DDI_NUM_NEG \
    --dropout $DDI_DROPOUT \
    --loss_func $DDI_LOSS_FUNC

# ogbl-ppa configuration
PPA_EMB_HIDDEN=256
PPA_MLP_HIDDEN=512
PPA_GNN_HIDDEN=512
PPA_GRAD_CLIP=2.0
PPA_EPOCHS=500
PPA_EVAL_STEPS=1
PPA_NUM_NEG=3
PPA_DROPOUT=0.5
PPA_USE_NODE_FEAT=True
PPA_ALPHA=0.5
PPA_LOSS_FUNC=WeightedHingeAUC

log_status ">>> Dataset: ogbl-ppa"
log_status "    Description: Protein-protein association network (link prediction)"
log_status "    Configuration: Emb=$PPA_EMB_HIDDEN, GNN=$PPA_GNN_HIDDEN, MLP=$PPA_MLP_HIDDEN, Dropout=$PPA_DROPOUT"
log_status "    Training: Epochs=$PPA_EPOCHS, NumNeg=$PPA_NUM_NEG, Alpha=$PPA_ALPHA, GradClip=$PPA_GRAD_CLIP"

run_experiment "ogbl-ppa" \
    python $OGB_DIR/main.py \
    --data_name ogbl-ppa \
    --emb_hidden_channels $PPA_EMB_HIDDEN \
    --mlp_hidden_channels $PPA_MLP_HIDDEN \
    --gnn_hidden_channels $PPA_GNN_HIDDEN \
    --grad_clip_norm $PPA_GRAD_CLIP \
    --epochs $PPA_EPOCHS \
    --eval_steps $PPA_EVAL_STEPS \
    --num_neg $PPA_NUM_NEG \
    --dropout $PPA_DROPOUT \
    --use_node_feat $PPA_USE_NODE_FEAT \
    --alpha $PPA_ALPHA \
    --loss_func $PPA_LOSS_FUNC

# ogbl-citation2 configuration
CITATION2_EMB_HIDDEN=64
CITATION2_MLP_HIDDEN=256
CITATION2_GNN_HIDDEN=256
CITATION2_GRAD_CLIP=1.0
CITATION2_EPOCHS=100
CITATION2_EVAL_STEPS=1
CITATION2_NUM_NEG=3
CITATION2_DROPOUT=0.3
CITATION2_EVAL_METRIC=mrr
CITATION2_NEG_SAMPLER=local
CITATION2_USE_NODE_FEAT=True
CITATION2_ALPHA=0.6

log_status ">>> Dataset: ogbl-citation2"
log_status "    Description: Citation network (link prediction)"
log_status "    Configuration: Emb=$CITATION2_EMB_HIDDEN, GNN=$CITATION2_GNN_HIDDEN, MLP=$CITATION2_MLP_HIDDEN, Dropout=$CITATION2_DROPOUT"
log_status "    Training: Epochs=$CITATION2_EPOCHS, NumNeg=$CITATION2_NUM_NEG, Alpha=$CITATION2_ALPHA, Metric=$CITATION2_EVAL_METRIC"

run_experiment "ogbl-citation2" \
    python $OGB_DIR/main.py \
    --data_name ogbl-citation2 \
    --emb_hidden_channels $CITATION2_EMB_HIDDEN \
    --mlp_hidden_channels $CITATION2_MLP_HIDDEN \
    --gnn_hidden_channels $CITATION2_GNN_HIDDEN \
    --grad_clip_norm $CITATION2_GRAD_CLIP \
    --epochs $CITATION2_EPOCHS \
    --eval_steps $CITATION2_EVAL_STEPS \
    --num_neg $CITATION2_NUM_NEG \
    --dropout $CITATION2_DROPOUT \
    --eval_metric $CITATION2_EVAL_METRIC \
    --neg_sampler $CITATION2_NEG_SAMPLER \
    --use_node_feat $CITATION2_USE_NODE_FEAT \
    --alpha $CITATION2_ALPHA

# =============================================================================
# Completion
# =============================================================================

log_status ""
log_status "=========================================="
log_status "All experiments completed successfully!"
log_status "Full log saved to: $LOG_FILE"
log_status "=========================================="

# Exit with success code
exit 0

