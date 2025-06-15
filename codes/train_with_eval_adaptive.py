import os
import subprocess
import json
import re
import sys
import time
import matplotlib.pyplot as plt
from nltk.translate.gleu_score import corpus_gleu
from collections import defaultdict
import argparse

# TOKENIZATION_SCHEMES = ["basic", "bpe", "unigram", "wordpiece"]
TOKENIZATION_SCHEMES = [ "basic", "bpe", "unigram", "wordpiece"]
BASE_DIR = "/ssd-data1/sq2023/belajar"
COMMON_GEN_ARGS = [
    '--task', 'translation',
    '--source-lang', 'source',
    '--target-lang', 'target',
    '--beam', '5',
    '--remove-bpe',
    '--batch-size', '16',
    '--skip-invalid-size-inputs-valid-test',
    # '--fp16'
]

def calculate_gleu(hypothesis_file, reference_file, output_dir=None):
    """Calculate GLEU score for GEC evaluation"""
    if output_dir is None:
        output_dir = os.path.dirname(hypothesis_file)
    
    hyp_texts = read_text_file(hypothesis_file)
    ref_texts = read_text_file(reference_file)
    
    # Tokenize sentences
    hyp_tokens = [sentence.lower().split() for sentence in hyp_texts]
    ref_tokens = [[sentence.lower().split()] for sentence in ref_texts]  # List of lists for multiple references
    
    # Calculate GLEU score
    gleu_score = corpus_gleu(ref_tokens, hyp_tokens)
    
    # Save result
    with open(os.path.join(output_dir, 'gleu_score.txt'), 'w') as f:
        f.write(f"GLEU Score: {gleu_score}")
    
    print(f"GLEU Score: {gleu_score}")
    return gleu_score
def plot_metrics(metrics_file, output_dir=None):
    """Plot training and evaluation metrics"""
    if output_dir is None:
        output_dir = os.path.dirname(metrics_file)
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    # Extract metrics by epoch
    epochs = [entry['epoch'] for entry in metrics['epochs']]
    precision = [entry['precision'] for entry in metrics['epochs']]
    recall = [entry['recall'] for entry in metrics['epochs']]
    f1 = [entry['f1'] for entry in metrics['epochs']]
    
    # Plot precision, recall, and F1
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, precision, 'b-', label='Precision')
    plt.plot(epochs, recall, 'g-', label='Recall')
    plt.plot(epochs, f1, 'r-', label='F1')
    
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Precision, Recall, and F1 Score by Epoch')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(output_dir, 'metrics_plot.png'))
    plt.close()
    
    print(f"Metrics plot saved to {os.path.join(output_dir, 'metrics_plot.png')}")

def train_and_evaluate_gec(data_dir, model_dir, raw_dir, use_linked=False):
    """Train GEC model and track metrics"""
    os.makedirs(model_dir, exist_ok=True)
    
    metrics_dir = os.path.join(model_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)

    # Training command dengan kondisional untuk linked transformer
    if use_linked:
        arch = 'adaptive_linked_transformer'
        # Parameter tambahan untuk linked transformer
        arch_specific = [
            '--user-dir', '/root/autodl-tmp/belajar/fairseq_extensions',  # Sesuaikan dengan path ekstensi Anda
        ]
    else:
        arch = 'transformer'
        arch_specific = []
    
    # Training command
    train_cmd = [
        'fairseq-train',
        data_dir,
        '--arch', arch,  # Or 'linked_transformer'
        *arch_specific,
        '--encoder-layers', '4',
        '--decoder-layers', '4',
        '--encoder-attention-heads', '8',
        '--decoder-attention-heads', '8',
        '--encoder-embed-dim', '256',
        '--decoder-embed-dim', '256',
        '--encoder-ffn-embed-dim', '2048',
        '--decoder-ffn-embed-dim', '2048',
        '--dropout', '0.3',
        '--attention-dropout', '0.1',
        '--share-decoder-input-output-embed',
        '--optimizer', 'adam',
        '--adam-betas', '(0.9, 0.98)',
        '--clip-norm', '0.1',
        '--lr', '5e-4',
        '--lr-scheduler', 'inverse_sqrt',
        '--warmup-updates', '4000',
        '--criterion', 'label_smoothed_cross_entropy',
        '--label-smoothing', '0.1',
        '--max-tokens', '1024',
        '--batch-size', '128',
        '--update-freq', '2',
        '--save-dir', model_dir,
        '--max-epoch', '10',
        '--patience', '5',
        '--keep-best-checkpoints', '2',
        '--save-interval', '1',  # Save every epoch
        '--validate-interval', '1',  # Validate every epoch
        '--no-epoch-checkpoints',  # Enable epoch checkpoints
        '--log-format', 'json',
        '--log-interval', '50',
        '--fp16',
        '--fp16-no-flatten-grads',
        '--memory-efficient-fp16',
        '--seed', '42',
        '--no-save-optimizer-state', 
        '--skip-invalid-size-inputs-valid-test',
        '--best-checkpoint-metric', 'loss',
    ]
    
    # Start training
    print("Starting training...")
    train_process = subprocess.Popen(train_cmd)
    
    # Wait for training to finish
    train_process.wait()
    
    # After your existing evaluation
    best_checkpoint = os.path.join(model_dir, 'checkpoint_best.pt')
    if os.path.exists(best_checkpoint):
        # Tambahkan log
        print(f"============== EVALUASI {os.path.basename(model_dir)} ==============")
        
        print(f"Training completed. Evaluating best checkpoint: {best_checkpoint}")
        try:
            evaluate_checkpoint(data_dir, best_checkpoint, metrics_dir)
            print(f"Standard evaluation completed for {os.path.basename(model_dir)}")
        except Exception as e:
            print(f"Error in standard evaluation for {os.path.basename(model_dir)}: {e}")
        
        print("Performing category-based evaluation...")
        try:
            category_results = evaluate_per_category(model_dir=model_dir, data_dir=data_dir, raw_dir=raw_dir, split='test')
            print(f"Category evaluation completed for {os.path.basename(model_dir)}")
        except Exception as e:
            print(f"Error in category evaluation for {os.path.basename(model_dir)}: {e}")
        
        # Continue with existing code
        log_file = os.path.join(model_dir, 'train.log')
        with open(log_file, 'w') as log:
            train_process = subprocess.Popen(train_cmd, stdout=log, stderr=subprocess.STDOUT)
            train_process.wait()

        if os.path.exists(log_file):
            plot_training_log(log_file, metrics_dir)
        
        # Plot custom metrics if available
        metrics_file = os.path.join(metrics_dir, 'metrics.json')
        if os.path.exists(metrics_file):
            plot_metrics(metrics_file, metrics_dir)
    else:
        print("Training did not produce a best checkpoint.")
    
    print(f"Training and evaluation completed. See results in {metrics_dir}")
    return model_dir, metrics_dir

def evaluate_per_category(model_dir, data_dir, raw_dir, split='test'):
    """Evaluasi model per kategori error"""
    
    # Pastikan best checkpoint ada
    best_checkpoint = f'{model_dir}/checkpoint_best.pt'
    if not os.path.exists(best_checkpoint):
        print(f"Error: Best checkpoint not found at {best_checkpoint}")
        return None
    
    # Load metadata kategori
    category_file = os.path.join(raw_dir, f"{split}.category")
    
    if not os.path.exists(category_file):
        print(f"Error: Category file not found at {category_file}")
        return None
        
    with open(category_file, 'r') as f:
        categories = [line.strip() for line in f]
    
    # Generate predictions
    print(f"Generating predictions for {split} set...")
    gen_cmd = [
        'fairseq-generate', data_dir,
        '--path', best_checkpoint,
        '--gen-subset', split,
        '--results-path', f'{model_dir}/{split}_results',
        *COMMON_GEN_ARGS
    ]
    
    try:
        subprocess.run(gen_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Generation failed with error: {e}")
        return None
    
    # Parse output dan analisis per kategori
    results_by_category = defaultdict(list)
    
    results_file = f'{model_dir}/{split}_results/generate-{split}.txt'
    if not os.path.exists(results_file):
        print(f"Error: Results file not found at {results_file}")
        return None
    
    with open(results_file, 'r') as f:
        lines = f.readlines()
    
    current_id = -1
    for line in lines:
        if line.startswith('S-'):  # Source
            try:
                current_id = int(line.split('\t')[0].split('-')[1])
            except:
                continue
        elif line.startswith('T-'):  # Target
            continue
        elif line.startswith('H-'):  # Hypothesis
            try:
                # Ekstrak score dari line
                parts = line.split('\t')
                if len(parts) >= 2:
                    score = -float(parts[1])  # Ubah dari negative log prob ke prob
                    if 0 <= current_id < len(categories):
                        category = categories[current_id]
                        results_by_category[category].append(score)
            except:
                continue
    
    # Hitung rata-rata per kategori
    category_results = {}
    for category, scores in results_by_category.items():
        if scores:
            avg_score = sum(scores) / len(scores)
            category_results[category] = {
                'average_score': avg_score,
                'count': len(scores)
            }
    
    # Simpan hasil
    output_file = f'{model_dir}/category_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(category_results, f, indent=2)
    
    # Tampilkan hasil
    print("\nAnalisis per kategori:")
    for category, result in category_results.items():
        print(f"{category}: Avg Score = {result['average_score']:.4f} (n={result['count']})")
    
    # Buat visualisasi
    create_category_visualization(category_results, model_dir)
    
    return category_results

def create_category_visualization(category_results, model_dir):
    """Membuat visualisasi hasil per kategori"""
    if not category_results:
        print("No results to visualize")
        return
    
    categories = list(category_results.keys())
    scores = [category_results[cat]['average_score'] for cat in categories]
    counts = [category_results[cat]['count'] for cat in categories]
    
    # Bar plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(categories, scores)
    plt.xlabel('Error Category')
    plt.ylabel('Average Score')
    plt.title('Performance by Error Category')
    plt.xticks(rotation=45, ha='right')
    
    # Tambahkan label count
    for i, (bar, count) in enumerate(zip(bars, counts)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'n={count}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'{model_dir}/category_performance.png', dpi=300)
    plt.close()
    
    print(f"Visualization saved to {model_dir}/category_performance.png")

def evaluate_checkpoint(data_dir, checkpoint, output_dir):
    """Evaluate a checkpoint with various metrics"""
    # Generate outputs
    output_file = os.path.join(output_dir, 'gen_output.txt')
    print(f"Running fairseq-generate and saving output to {output_file}")
    
    cmd = [
        'fairseq-generate', data_dir,
        '--path', checkpoint,
        '--gen-subset', 'test',
        *COMMON_GEN_ARGS
    ]
    
    print(f"Running fairseq-generate with command: {' '.join(cmd)}")
    try:
        with open(output_file, 'w') as f:
            subprocess.run(cmd, stdout=f, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running fairseq-generate: {e}")
        return
    
    # Periksa output file lebih detail
    if not os.path.exists(output_file):
        print(f"ERROR: Output file {output_file} tidak dibuat!")
        return
    
    if os.path.getsize(output_file) < 100:  # Ukuran minimal yang diharapkan
        print(f"WARNING: Output file {output_file} terlalu kecil (size: {os.path.getsize(output_file)} bytes)")
        print("Content:")
        with open(output_file, 'r') as f:
            print(f.read())
        return
    
    
    # Extract texts
    src_file = os.path.join(output_dir, 'src.txt')
    ref_file = os.path.join(output_dir, 'ref.txt')
    hyp_file = os.path.join(output_dir, 'hyp.txt')
    
    extraction_success = extract_texts(output_file, src_file, ref_file, hyp_file)
    
    if not extraction_success:
        print("Failed to extract text properly from fairseq-generate output")
        # Create a debug file with raw content in a more structured format
        debug_file = os.path.join(output_dir, 'debug_output.txt')
        with open(output_file, 'r') as f_in, open(debug_file, 'w') as f_out:
            for i, line in enumerate(f_in):
                f_out.write(f"{i+1}: {line}")
        print(f"Saved debug output to {debug_file}")
        return
    
    # Calculate word-level F1
    precision, recall, f1 = calculate_word_f1(hyp_file, ref_file, output_dir)
    
    # Try to calculate GLEU if NLTK is available
    try:
        from nltk.translate.gleu_score import corpus_gleu
        gleu = calculate_gleu(hyp_file, ref_file, output_dir)
    except ImportError:
        print("NLTK not available for GLEU calculation")
        gleu = None
    
    # Calculate BLEU score using SacreBLEU
    bleu = calculate_sacrebleu(hyp_file, ref_file, output_dir)
    # Save metrics
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'gleu': gleu,
        'bleu' : bleu
    }
    
    with open(os.path.join(output_dir, 'final_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Evaluation results: Precision={precision}, Recall={recall}, F1={f1}, GLEU={gleu}")

def extract_texts(pred_file, src_file, ref_file, hyp_file):
    """Extract source, reference and hypothesis from fairseq-generate output"""
    # First, check if the input file exists and has content
    if not os.path.exists(pred_file) or os.path.getsize(pred_file) == 0:
        print(f"Error: Input file {pred_file} is empty or does not exist")
        return False
        
    with open(pred_file, 'r') as f:
        content = f.read()
        # Debug: Print first few lines to understand the format
        print(f"Debug - First 200 chars of generate output: {content[:200]}")
    
    # Reset file reading
    with open(pred_file, 'r') as f, \
         open(src_file, 'w') as src_out, \
         open(ref_file, 'w') as ref_out, \
         open(hyp_file, 'w') as hyp_out:
        
        # Line counting for debugging
        line_count = 0
        src_count = 0
        ref_count = 0
        hyp_count = 0
        
        for line in f:
            line_count += 1
            # Support both fairseq format options
            if line.startswith('S-') or line.startswith('S\t'):
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    src_out.write(parts[1] + '\n')
                    src_count += 1
            elif line.startswith('T-') or line.startswith('T\t'):
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    ref_out.write(parts[1] + '\n')
                    ref_count += 1
            elif line.startswith('H-') or line.startswith('H\t'):
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    hyp_out.write(parts[2] + '\n')
                    hyp_count += 1
                elif len(parts) == 2:  # Alternative format
                    hyp_out.write(parts[1] + '\n')
                    hyp_count += 1
        
        # Print debug information
        print(f"Processed {line_count} lines from {pred_file}")
        print(f"Extracted: {src_count} source, {ref_count} reference, {hyp_count} hypothesis sentences")
        
        # Verify files are not empty
        return src_count > 0 and ref_count > 0 and hyp_count > 0

def calculate_word_f1(hyp_file, ref_file, output_dir=None):
    """Calculate word-level F1 score"""
    if output_dir is None:
        output_dir = os.path.dirname(hyp_file)
    
    hyp_texts = read_text_file(hyp_file)
    ref_texts = read_text_file(ref_file)
    
    if len(hyp_texts) != len(ref_texts):
        print(f"Error: Number of sentences in hypothesis ({len(hyp_texts)}) and reference ({len(ref_texts)}) do not match")
        return 0, 0, 0
    
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    count = 0
    
    for hyp, ref in zip(hyp_texts, ref_texts):
        hyp_words = set(hyp.lower().split())
        ref_words = set(ref.lower().split())
        
        # True positives
        tp = len(hyp_words.intersection(ref_words))
        
        # False positives
        fp = len(hyp_words - ref_words)
        
        # False negatives
        fn = len(ref_words - hyp_words)
        
        # Calculate precision, recall, and F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        count += 1
    
    # Calculate averages
    avg_precision = total_precision / count if count > 0 else 0
    avg_recall = total_recall / count if count > 0 else 0
    avg_f1 = total_f1 / count if count > 0 else 0
    
    # Save detailed results
    results_file = os.path.join(output_dir, 'f1_results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'average': {
                'precision': avg_precision,
                'recall': avg_recall,
                'f1': avg_f1
            }
        }, f, indent=4)
    
    return avg_precision, avg_recall, avg_f1

def read_text_file(file_path):
    """Read text file and return list of lines"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

def plot_training_log(log_file, output_dir):
    """Plot training metrics from log file"""
    # Parse the log file
    train_losses = []
    val_losses = []
    bleu_scores = []
    
    with open(log_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                
                # Get epoch or update number
                epoch = data.get('epoch', 0)
                
                # Training loss
                if 'train_loss' in data:
                    train_losses.append((epoch, data['train_loss']))
                
                # Validation loss
                if 'valid_loss' in data:
                    val_losses.append((epoch, data['valid_loss']))
                
                # BLEU score
                if 'bleu' in data:
                    bleu_scores.append((epoch, data['bleu']))
            except:
                continue
    
    # Plot training and validation loss
    if train_losses and val_losses:
        plt.figure(figsize=(10, 6))
        
        # Sort by epoch
        train_losses.sort(key=lambda x: x[0])
        val_losses.sort(key=lambda x: x[0])
        
        plt.plot([x[0] for x in train_losses], [x[1] for x in train_losses], 'b-', label='Training Loss')
        plt.plot([x[0] for x in val_losses], [x[1] for x in val_losses], 'r-', label='Validation Loss')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
        plt.close()
    
    # Plot BLEU scores
    if bleu_scores:
        plt.figure(figsize=(10, 6))
        
        # Sort by epoch
        bleu_scores.sort(key=lambda x: x[0])
        
        plt.plot([x[0] for x in bleu_scores], [x[1] for x in bleu_scores], 'g-', marker='o')
        
        plt.xlabel('Epoch')
        plt.ylabel('BLEU Score')
        plt.title('BLEU Score Progress')
        plt.grid(True)
        
        plt.savefig(os.path.join(output_dir, 'bleu_plot.png'))
        plt.close()

def calculate_sacrebleu(hyp_file, ref_file, output_dir=None):
    """Calculate BLEU score using SacreBLEU"""
    if output_dir is None:
        output_dir = os.path.dirname(hyp_file)
    
    try:
        # Import sacrebleu with error handling for different versions
        import sacrebleu
        
        # Read files
        with open(hyp_file, 'r', encoding='utf-8') as f:
            hyps = [line.strip() for line in f]
        
        with open(ref_file, 'r', encoding='utf-8') as f:
            refs = [line.strip() for line in f]
        
        # Handle API differences between sacrebleu versions
        if hasattr(sacrebleu, 'corpus_bleu'):
            # Newer sacrebleu versions
            bleu_score = sacrebleu.corpus_bleu(hyps, [refs]).score
        else:
            # Older sacrebleu versions with compute_bleu function
            refs_transposed = [[r] for r in refs]  # SacreBLEU expects list of lists
            bleu_score = sacrebleu.compute_bleu(hyps, refs_transposed).score
        
        # Save result
        with open(os.path.join(output_dir, 'bleu_score.txt'), 'w') as f:
            f.write(f"BLEU Score: {bleu_score}")
        
        print(f"BLEU Score: {bleu_score}")
        return bleu_score
        
    except ImportError:
        print("SacreBLEU not available for BLEU calculation")
        return None
    except Exception as e:
        print(f"Error calculating BLEU score: {e}")
        return None

def collect_final_scores():
    rows = []
    for scheme in TOKENIZATION_SCHEMES:
        mdir = f"{BASE_DIR}/checkpoints/{scheme}/metrics/final_metrics.json"
        with open(mdir) as f:
            m = json.load(f)
        rows.append([scheme, m['precision'], m['recall'], m['f1'], m['gleu']])
    
    rows.sort(key=lambda x: x[-2], reverse=True)  # urut F1
    print("{:<10} {:>8} {:>8} {:>8} {:>8}".format(
        "Scheme", "Prec", "Recall", "F1", "GLEU"))
    for r in rows:
        print("{:<10} {:8.4f} {:8.4f} {:8.4f} {:8.4f}".format(*r))

# def run_all_experiments(schemes):
#     """Latih & evaluasi hanya skema yang dipilih"""
#     for scheme in schemes:
#         assert scheme in TOKENIZATION_SCHEMES, f"Unknown scheme: {scheme}"
#         data_dir  = f"{BASE_DIR}/data-bin-{scheme}"
#         raw_dir   = f"{BASE_DIR}/raw-{scheme}"
#         model_dir = f"{BASE_DIR}/checkpoints/{scheme}"
#         train_and_evaluate_gec(data_dir=data_dir, model_dir=model_dir, raw_dir=raw_dir)


def run_all_experiments(schemes, use_linked=False):
    """Train & evaluate selected tokenization schemes"""
    model_suffix = "_adaptive_linked" if use_linked else ""

    for scheme in schemes:
        assert scheme in TOKENIZATION_SCHEMES, f"Unknown scheme: {scheme}"
        data_dir = f"{BASE_DIR}/data-bin-{scheme}"
        raw_dir = f"{BASE_DIR}/raw-{scheme}"
        model_dir = f"{BASE_DIR}/checkpoints/{scheme}{model_suffix}"
        
        # Create a specific log file for this scheme
        log_path = os.path.join(model_dir, "log_adaptive_linked.txt")
        os.makedirs(model_dir, exist_ok=True)
        
        # Make sure you have train_with_eval_gatelinked.py in the same directory
        cmd = f"python -c \"import sys; sys.path.append('{os.path.dirname(__file__)}'); from train_with_eval_adaptive import train_and_evaluate_gec; train_and_evaluate_gec('{data_dir}', '{model_dir}', '{raw_dir}', {str(use_linked)})\" 2>&1 | tee {log_path}"
        
        print(f"\n==== STARTING EXPERIMENT FOR {scheme.upper()} {model_suffix.upper()} ====\n")
        subprocess.run(cmd, shell=True)
        print(f"\n==== COMPLETED EXPERIMENT FOR {scheme.upper()} {model_suffix.upper()} ====\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--schemes",
        nargs="+",
        # default=TOKENIZATION_SCHEMES,
        default=["wordpiece"],
        help="Daftar tokenization scheme yang ingin dijalankan "
             "(pilih: basic bpe unigram wordpiece)",
    )
    parser.add_argument(
        "--linked",
        action="store_true",
        help="Gunakan arsitektur Linked Transformer"
    )
    args = parser.parse_args()

    # Opsional: tampilkan GPU yg dipakai—diatur dari shell lewat CUDA_VISIBLE_DEVICES
    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES", "(not set)"))

    run_all_experiments(args.schemes, args.linked)


    # CUDA_VISIBLE_DEVICES=6 python /ssd-data1/sq2023/belajar/scripts/train_with_eval.py > logfile_indogec_IGED.txt

# CUDA_VISIBLE_DEVICES=5 python /ssd-data1/sq2023/belajar/scripts/train_with_eval.py --schemes basic bpe > logfile_indogec_IGED_basicBPE.txt
# CUDA_VISIBLE_DEVICES=8 python /ssd-data1/sq2023/belajar/scripts/train_with_eval.py --schemes unigram wordpiece > logfile_indogec_IGED_UniWord.txt
# CUDA_VISIBLE_DEVICES=0 python /root/autodl-tmp/belajar/train_with_eval.py --schemes bpe unigram wordpiece > logfile_indogec_IGED.txt

# CUDA_VISIBLE_DEVICES=0 fairseq-generate /root/autodl-tmp/belajar/data-bin-wordpiece \
#   --path /root/autodl-tmp/belajar/checkpoints/wordpiece_adaptive_linked/checkpoint_best.pt \
#   --gen-subset test \
#   --task translation \
#   --source-lang source \
#   --target-lang target \
#   --beam 5 \
#   --remove-bpe \
#   --batch-size 32 \
#   --fp16 \
#   --skip-invalid-size-inputs-valid-test \
#   --user-dir /root/autodl-tmp/belajar/fairseq_extensions \
#   > /root/autodl-tmp/belajar/checkpoints/wordpiece_adaptive_linked/metrics/gen_output.txt

# 
# cat > /root/autodl-tmp/belajar/evaluate_adaptive_linked_manual.py << 'EOF'
# import sys
# import os
# sys.path.append('/root/autodl-tmp/belajar')

# # Import fungsi-fungsi yang diperlukan dari train_with_eval.py
# from train_with_eval_adaptive import extract_texts, calculate_word_f1, calculate_gleu, calculate_sacrebleu

# # Definisikan jalur file
# output_dir = '/root/autodl-tmp/belajar/checkpoints/wordpiece_adaptive_linked/metrics'
# gen_output = os.path.join(output_dir, 'gen_output.txt')
# src_file = os.path.join(output_dir, 'src.txt')
# ref_file = os.path.join(output_dir, 'ref.txt')
# hyp_file = os.path.join(output_dir, 'hyp.txt')

# # Langkah 1: Ekstrak teks sumber, referensi dan prediksi
# print("Extracting text...")
# extraction_success = extract_texts(gen_output, src_file, ref_file, hyp_file)
# if not extraction_success:
#     print("Failed to extract texts from generation output")
#     sys.exit(1)

# # Langkah 2: Hitung metrik
# print("Calculating metrics...")
# precision, recall, f1 = calculate_word_f1(hyp_file, ref_file, output_dir)

# try:
#     from nltk.translate.gleu_score import corpus_gleu
#     gleu = calculate_gleu(hyp_file, ref_file, output_dir)
# except ImportError:
#     print("NLTK not available for GLEU calculation")
#     gleu = None

# bleu = calculate_sacrebleu(hyp_file, ref_file, output_dir)

# # Langkah 3: Simpan metrik ke JSON
# import json
# metrics = {
#     'precision': precision,
#     'recall': recall,
#     'f1': f1,
#     'gleu': gleu,
#     'bleu': bleu
# }

# with open(os.path.join(output_dir, 'final_metrics.json'), 'w') as f:
#     json.dump(metrics, f, indent=4)

# print(f"Evaluation completed. Results saved to {os.path.join(output_dir, 'final_metrics.json')}")
# print(f"Evaluation results: Precision={precision}, Recall={recall}, F1={f1}, GLEU={gleu}, BLEU={bleu}")
# EOF

# # Jalankan script evaluasi manual
# python /root/autodl-tmp/belajar/evaluate_adaptive_linked_manual.py


# 3.
# cat > /root/evaluate_baseline_category.py << 'EOF'
# import sys
# import os
# sys.path.append('/ssd-data1/sq2023/belajar')

# # Import fungsi evaluasi per kategori
# from train_with_eval import evaluate_per_category

# # Evaluasi per kategori
# print("Performing category-based evaluation...")
# try:
#     category_results = evaluate_per_category(
#         model_dir='/ssd-data1/sq2023/belajar/checkpoints/baseline',
#         data_dir='/ssd-data1/sq2023/belajar/data-bin-basic',
#         raw_dir='/ssd-data1/sq2023/belajar/raw-basic',
#         split='test'
#     )
#     print("Category evaluation completed.")
# except Exception as e:
#     print(f"Error in category evaluation: {e}")
# EOF

# Jalankan script evaluasi kategori
# python /root/evaluate_bpe_category.py