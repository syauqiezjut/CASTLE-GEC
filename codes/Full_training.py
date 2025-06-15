# Full training dengan parameter optimal


CUDA_VISIBLE_DEVICES=9 fairseq-train data-bin-wordpiece \
    --arch castle_transformer \
    --user-dir fairseq_extensions \
    --kg-path semantic_kg_deepseek.json \
    --semantic-weight 0.7 \
    --encoder-layers 4 --decoder-layers 4 \
    --encoder-attention-heads 8 --decoder-attention-heads 8 \
    --encoder-embed-dim 256 --decoder-embed-dim 256 \
    --encoder-ffn-embed-dim 2048 --decoder-ffn-embed-dim 2048 \
    --dropout 0.3 --attention-dropout 0.1 \
    --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.1 --lr 5e-4 \
    --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 1024 --batch-size 64 --update-freq 4 \
    --save-dir hyperparameter_test_castle_simple \
    --max-epoch 10 --patience 5 \
    --keep-best-checkpoints 1 --save-interval 1 --validate-interval 1 \
    --no-epoch-checkpoints --log-format json --log-interval 50 \
    --seed 42 --no-save-optimizer-state \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric loss



CUDA_VISIBLE_DEVICES=9 fairseq-generate data-bin-wordpiece \
        --path castle/checkpoint_best.pt \
        --user-dir fairseq_extensions \
        --gen-subset test \
        --beam 5 --batch-size 32 --max-tokens 2048 \
        --task translation --source-lang source --target-lang target \
        --skip-invalid-size-inputs-valid-test --remove-bpe \
        > castle/gen_output.txt 2>&1

python -c "
import sys
sys.path.append('/ssd-data1/sq2023/belajar')
from train_with_eval_adaptive import extract_texts, calculate_word_f1, calculate_sacrebleu
import os

output_dir = 'hyperparameter_test_castle_simple'
gen_output = f'{output_dir}/gen_output.txt'
src_file = f'{output_dir}/src.txt'  
ref_file = f'{output_dir}/ref.txt'
hyp_file = f'{output_dir}/hyp.txt'

print('Extracting texts from fairseq-generate output...')
success = extract_texts(gen_output, src_file, ref_file, hyp_file)

if success:
    print('Calculating metrics...')
    precision, recall, f1 = calculate_word_f1(hyp_file, ref_file, output_dir)
    bleu = calculate_sacrebleu(hyp_file, ref_file, output_dir)
    
    print('\\n' + '='*50)
    print('CASTLE ENHANCED RESULTS (1 Epoch):')
    print('='*50)
    print(f'Precision: {precision:.4f}')
    print(f'Recall:    {recall:.4f}')
    print(f'F1 Score:  {f1:.4f}')
    print(f'BLEU:      {bleu:.2f}')
    print()
    
    # Compare dengan paper baseline results
    print('COMPARISON WITH PAPER BASELINES:')
    print('-'*50)
    baselines = {
        'Baseline + WordPiece': {'f1': 0.9579, 'bleu': 92.30},
        'WP + Linked Attention': {'f1': 0.9570, 'bleu': 92.21},
        'CASTLE (Paper)': {'f1': 0.9579, 'bleu': 92.41}
    }
    
    for name, scores in baselines.items():
        f1_diff = f1 - scores['f1']
        bleu_diff = bleu - scores['bleu']
        print(f'{name:25} | F1: {scores[\"f1\"]:.4f} | BLEU: {scores[\"bleu\"]:5.2f} | Δ F1: {f1_diff:+.4f} | Δ BLEU: {bleu_diff:+.2f}')
    
    print()
    if f1 > 0.9579:
        print('SUCCESS: F1 IMPROVED over baseline!')
    elif f1 > 0.9570:
        print(' F1 better than linked attention!')
    else:
        print(' F1 needs improvement, but architecture is working')
 
    
else:
    print('Failed to extract texts - checking generation output...')
    
    # Debug generation output
    if os.path.exists(gen_output):
        print(f'Generation file exists, size: {os.path.getsize(gen_output)} bytes')
        print('First 10 lines:')
        with open(gen_output, 'r') as f:
            for i, line in enumerate(f):
                if i < 10:
                    print(f'{i+1}: {line.strip()}')
                else:
                    break
    else:
        print('Generation output file not found!')
"



python -c "
import sys
sys.path.append('/ssd-data1/sq2023/belajar')

# Import fungsi yang sudah ada
from train_with_eval_adaptive import evaluate_per_category

print('Running category evaluation...')
category_results = evaluate_per_category(
    model_dir='hyperparameter_test_castle_simple',
    data_dir='data-bin-wordpiece',
    raw_dir='raw-wordpiece', 
    split='test'
)

if category_results:
    print('')
    print('CATEGORY PERFORMANCE RESULTS:')
    print('='*50)
    
    
    for category, result in category_results.items():
        score = result['average_score']
        
"



Detailed results saved to: castle/category_analysis_detailed.json
