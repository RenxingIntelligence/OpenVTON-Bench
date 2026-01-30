#!/bin/bash
# OpenVTON-Bench Quick Start Script

set -e  # Exit immediately on error

echo "==================================================================="
echo "OpenVTON-Bench - Virtual Try-On Benchmark Evaluation"
echo "==================================================================="
echo ""

# Check configuration file
if [ ! -f "benchmark/config.yaml" ]; then
    echo "Error: config.yaml not found!"
    echo "Please create benchmark/config.yaml first."
    echo "You can copy from benchmark/config.yaml.example"
    exit 1
fi

# Default parameters
CONFIG="benchmark/config.yaml"
EVAL_TYPE=""
MODELS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --eval-type)
            EVAL_TYPE="$2"
            shift 2
            ;;
        --models)
            MODELS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --config PATH        Config file path (default: benchmark/config.yaml)"
            echo "  --eval-type TYPE     Evaluation type: vlm, garment, all, pixel"
            echo "  --models MODELS      Model names to evaluate (space-separated)"
            echo "  --help               Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Run full evaluation"
            echo "  $0 --eval-type vlm                    # Only VLM evaluation"
            echo "  $0 --models \"model_A model_B\"         # Evaluate specific models"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build command
CMD="python benchmark/run_benchmark.py --config $CONFIG"

if [ -n "$EVAL_TYPE" ]; then
    CMD="$CMD --eval_type $EVAL_TYPE"
fi

if [ -n "$MODELS" ]; then
    CMD="$CMD --models $MODELS"
fi

# Display configuration info
echo "Configuration:"
echo "  Config file: $CONFIG"
if [ -n "$EVAL_TYPE" ]; then
    echo "  Evaluation type: $EVAL_TYPE"
else
    echo "  Evaluation type: all (from config)"
fi
if [ -n "$MODELS" ]; then
    echo "  Models: $MODELS"
else
    echo "  Models: all (from config)"
fi
echo ""

# Run evaluation
echo "Starting evaluation..."
echo "Command: $CMD"
echo ""

$CMD

echo ""
echo "==================================================================="
echo "Evaluation completed!"
echo "Results saved to benchmark/results/"
echo "==================================================================="
