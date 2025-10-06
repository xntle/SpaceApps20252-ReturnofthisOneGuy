#!/bin/bash
# Train Pixel CNN on all 5 folds

echo "🎯 Training Pixel CNN on all 5 folds"
echo "======================================"

# Configuration
EPOCHS=40
BASE=16
DROPOUT=0.3
MODEL_TYPE="standard"
BATCH_SIZE=32
LEARNING_RATE=3e-4
WEIGHT_DECAY=5e-4

echo "Configuration:"
echo "  Epochs: $EPOCHS"
echo "  Model: $MODEL_TYPE with $BASE base channels"
echo "  Dropout: $DROPOUT"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo ""

# Python executable
PYTHON_CMD="/home/roshan/Desktop/new-exo-model/.venv/bin/python"

# Train each fold
for fold in {0..4}; do
    echo "🔥 Training fold $fold..."
    $PYTHON_CMD src/train_pixel.py \
        --fold $fold \
        --epochs $EPOCHS \
        --base $BASE \
        --dropout $DROPOUT \
        --model_type $MODEL_TYPE \
        --batch_size $BATCH_SIZE \
        --lr $LEARNING_RATE \
        --weight_decay $WEIGHT_DECAY
    
    if [ $? -eq 0 ]; then
        echo "✅ Fold $fold completed successfully"
    else
        echo "❌ Fold $fold failed"
        break
    fi
    echo ""
done

echo "🎯 Training complete! Running cross-fold evaluation..."
$PYTHON_CMD src/evaluate_pixel.py

echo ""
echo "🎉 All done! Check models/ directory for results"
echo "📊 Best models and evaluation curves saved"