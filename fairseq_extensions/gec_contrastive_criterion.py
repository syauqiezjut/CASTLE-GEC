import math
import torch
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion


@register_criterion('gec_contrastive')
class GECContrastiveCriterion(LabelSmoothedCrossEntropyCriterion):
    """
    Specialized contrastive learning criterion for Grammatical Error Correction.
    
    This criterion implements a combination of:
    1. Label-smoothed cross-entropy loss for token prediction
    2. Sentence-level contrastive loss between correct and incorrect sentence representations
    """

    def __init__(self, task, sentence_avg, label_smoothing, contrastive_weight=0.1, 
                 temperature=0.1):
        super().__init__(task, sentence_avg, label_smoothing)
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        LabelSmoothedCrossEntropyCriterion.add_args(parser)
        parser.add_argument('--contrastive-weight', default=0.1, type=float, metavar='D',
                            help='weight for the contrastive loss component')
        parser.add_argument('--temperature', default=0.1, type=float, metavar='D',
                            help='temperature parameter for contrastive loss')
        parser.add_argument('--contrastive-mode', default='sentence', type=str,
                            choices=['sentence', 'both'],
                            help='level at which to apply contrastive loss')

    def compute_sentence_level_contrastive_loss(self, src_encodings, tgt_encodings):
        """
        Compute sentence-level contrastive loss between source and target sentence representations.
        
        This brings representations of corrected sentences closer to their correct versions
        and pushes away from incorrect versions.
        """
        # Get average representation per sentence by mean pooling over sequence length
        # Result shape: [batch_size, embed_dim]
        src_sent_repr = src_encodings.mean(dim=1)
        tgt_sent_repr = tgt_encodings.mean(dim=1)
        
        # Normalize the representations
        src_sent_repr = F.normalize(src_sent_repr, p=2, dim=1)
        tgt_sent_repr = F.normalize(tgt_sent_repr, p=2, dim=1)
        
        # Compute similarity matrix for all possible pairs
        # For each source, all targets except its own are negatives
        src_tgt_similarity = torch.matmul(src_sent_repr, tgt_sent_repr.transpose(0, 1))
        
        # Apply temperature scaling
        src_tgt_similarity = src_tgt_similarity / self.temperature
        
        # For each source, the positive is its corresponding target
        labels = torch.arange(src_tgt_similarity.size(0), device=src_tgt_similarity.device)
        
        # Compute contrastive loss (InfoNCE / NTXent loss)
        contrastive_loss = F.cross_entropy(src_tgt_similarity, labels)
        
        return contrastive_loss
        
    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # Standard cross-entropy loss computation
        net_output = model(**sample['net_input'])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        
        # Initialize contrastive loss
        sentence_contrastive_loss = torch.tensor(0.0, device=loss.device)
        
        # Compute encoder representations
        encoder_out = model.encoder(
            src_tokens=sample['net_input']['src_tokens'],
            src_lengths=sample['net_input']['src_lengths'],
        )
        
        # Get source encodings - shape: [batch_size, src_len, embed_dim]
        src_encodings = encoder_out.encoder_out.transpose(0, 1)
        
        # Compute target encodings by running a forward pass with teacher forcing
        # Just to get representations, not for loss computation
        with torch.no_grad():
            # Use target as input to encoder to get "correct" sentence encodings
            target_encoder_out = model.encoder(
                src_tokens=sample['target'],
                src_lengths=(sample['target'] != self.padding_idx).sum(1),
            )
            # Get target encodings - shape: [batch_size, tgt_len, embed_dim]
            tgt_encodings = target_encoder_out.encoder_out.transpose(0, 1)
        
        # Compute sentence-level contrastive loss
        sentence_contrastive_loss = self.compute_sentence_level_contrastive_loss(
            src_encodings, tgt_encodings
        )
        
        # Combine with CE loss
        total_loss = loss + self.contrastive_weight * sentence_contrastive_loss
        
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(total_loss.data) if reduce else total_loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'sentence_contrastive_loss': utils.item(sentence_contrastive_loss.data) if reduce else sentence_contrastive_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        
        return total_loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        LabelSmoothedCrossEntropyCriterion.reduce_metrics(logging_outputs)
        
        # Aggregate contrastive losses
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        sentence_loss_sum = sum(log.get('sentence_contrastive_loss', 0) for log in logging_outputs)
        
        metrics.log_scalar(
            'sentence_contrastive_loss', sentence_loss_sum / sample_size / math.log(2), sample_size, round=3
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True