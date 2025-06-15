# /ssd-data1/sq2023/belajar/fairseq_extensions/semantic_aware_criterion.py
import math
import torch
import torch.nn.functional as F
from fairseq import utils, metrics
from fairseq.criterions import FairseqCriterion, register_criterion

@register_criterion('semantic_aware_cross_entropy')
class SemanticAwareCrossEntropyCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg, label_smoothing=0.1, semantic_loss_weight=0.7):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.semantic_loss_weight = semantic_loss_weight
        self.padding_idx = task.target_dictionary.pad()
    
    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--label-smoothing', default=0.1, type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--semantic-loss-weight', default=0.7, type=float, metavar='D',
                            help='weight for semantic error categories in the loss function')
    
    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample."""
        net_output = model(**sample['net_input'])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output
    
    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        
        # Apply loss
        nll_loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction='sum' if reduce else 'none',
        )
        
        # Apply label smoothing
        if self.eps > 0.0:
            smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
            non_pad_mask = target.ne(self.padding_idx)
            smooth_loss = smooth_loss[non_pad_mask]
            smooth_loss = smooth_loss.sum()
            eps_i = self.eps / (lprobs.size(-1) - 1)
            loss = (1. - self.eps - eps_i) * nll_loss + eps_i * smooth_loss
        else:
            loss = nll_loss
        
        # Apply semantic weight
        loss = loss * self.semantic_loss_weight
        
        return loss, nll_loss
    
    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        
        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))
    
    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return True
    
    
    # CUDA_VISIBLE_DEVICES=2 fairseq-train /ssd-data1/sq2023/belajar/data-bin-wordpiece \
    # CUDA_VISIBLE_DEVICES=0 fairseq-train /ssd-data1/sq2023/belajar/data-bin-wordpiece-semantic \
    # --arch castle_transformer \
    # --user-dir /ssd-data1/sq2023/belajar/fairseq_extensions \
    # --kg-path /ssd-data1/sq2023/belajar/semantic_kg_deepseek.json \
    # --semantic-weight 0.9 \
    # --criterion semantic_aware_cross_entropy \
    # --encoder-layers 4 \
    # --decoder-layers 4 \
    # --encoder-attention-heads 8 \
    # --decoder-attention-heads 8 \
    # --encoder-embed-dim 256 \
    # --decoder-embed-dim 256 \
    # --encoder-ffn-embed-dim 2048 \
    # --decoder-ffn-embed-dim 2048 \
    # --dropout 0.2 \
    # --attention-dropout 0.1 \
    # --share-decoder-input-output-embed \
    # --optimizer adam \
    # --adam-betas '(0.9, 0.98)' \
    # --clip-norm 0.1 \
    # --lr 1e-4 \
    # --lr-scheduler inverse_sqrt \
    # --warmup-updates 4000 \
    # --criterion label_smoothed_cross_entropy \
    # --label-smoothing 0.1 \
    # --max-tokens 1024 \
    # --batch-size 128 \
    # --update-freq 2 \
    # --save-dir /ssd-data1/sq2023/belajar/checkpoints/castle_improved \
    # --max-epoch 10 \
    # --patience 7 \
    # --keep-best-checkpoints 2 \
    # --save-interval 1 \
    # --validate-interval 1 \
    # --no-epoch-checkpoints \
    # --log-format json \
    # --log-interval 50 \
    # --seed 42 \
    # --no-save-optimizer-state \
    # --skip-invalid-size-inputs-valid-test \
    # --best-checkpoint-metric loss \
    # > /ssd-data1/sq2023/belajar/checkpoints/castle_improved/trainingnew.log 2>&1


    # CUDA_VISIBLE_DEVICES=1 fairseq-train /ssd-data1/sq2023/belajar/data-bin-wordpiece \
    # --arch castle_transformer \
    # --user-dir /ssd-data1/sq2023/belajar/fairseq_extensions \
    # --kg-path /ssd-data1/sq2023/belajar/semantic_kg_deepseek.json \
    # --semantic-weight 0.9 \
    # --criterion label_smoothed_cross_entropy \
    # --encoder-layers 6 \
    # --decoder-layers 6 \
    # --encoder-attention-heads 8 \
    # --decoder-attention-heads 8 \
    # --encoder-embed-dim 256 \
    # --decoder-embed-dim 256 \
    # --encoder-ffn-embed-dim 2048 \
    # --decoder-ffn-embed-dim 2048 \
    # --dropout 0.2 \
    # --attention-dropout 0.1 \
    # --share-decoder-input-output-embed \
    # --optimizer adam \
    # --adam-betas '(0.9, 0.98)' \
    # --clip-norm 0.1 \
    # --lr 5e-4 \
    # --lr-scheduler inverse_sqrt \
    # --warmup-updates 4000 \
    # --label-smoothing 0.1 \
    # --max-tokens 1024 \
    # --batch-size 128 \
    # --update-freq 2 \
    # --save-dir /ssd-data1/sq2023/belajar/checkpoints/castle_improved_orig \
    # --max-epoch 15 \
    # --patience 7 \
    # --keep-best-checkpoints 2 \
    # --save-interval 1 \
    # --validate-interval 1 \
    # --no-epoch-checkpoints \
    # --log-format json \
    # --log-interval 50 \
    # --seed 42 \
    # --no-save-optimizer-state \
    # --skip-invalid-size-inputs-valid-test \
    # --best-checkpoint-metric loss \
    # > /ssd-data1/sq2023/belajar/checkpoints/castle_improved_orig/training.log 2>&1