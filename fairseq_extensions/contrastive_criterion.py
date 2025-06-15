import math
import torch
import torch.nn.functional as F

from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion


@register_criterion('contrastive_label_smoothed_cross_entropy')
class ContrastiveLabelSmoothedCrossEntropyCriterion(LabelSmoothedCrossEntropyCriterion):
    """
    Implementation of label-smoothed cross-entropy loss with contrastive learning.
    """

    def __init__(self, task, sentence_avg, label_smoothing, contrastive_weight=0.1, temperature=0.1):
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

    def compute_contrastive_loss(self, encoder_out, sample):
        """
        Compute contrastive loss between encoder representations of source sentences
        and target sentences.
        
        Args:
            encoder_out: encoder output from the model
            sample: the sample batch
            
        Returns:
            contrastive_loss: the contrastive loss
        """
        # Get encoder output - shape: [src_len, batch_size, embed_dim]
        src_encodings = encoder_out.encoder_out
        
        # Get average representation per sentence by mean pooling over sequence length
        # Result shape: [batch_size, embed_dim]
        src_encodings = src_encodings.transpose(0, 1).mean(dim=1)
        
        # Normalize the encodings
        src_encodings = F.normalize(src_encodings, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(src_encodings, src_encodings.transpose(0, 1))
        
        # Apply temperature scaling
        similarity_matrix = similarity_matrix / self.temperature
        
        # Create labels: each example forms a positive pair with itself
        labels = torch.arange(similarity_matrix.size(0), device=similarity_matrix.device)
        
        # Compute contrastive loss (cross entropy loss with positive pairs as targets)
        contrastive_loss = F.cross_entropy(similarity_matrix, labels)
        
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
        
        # Get encoder output for contrastive loss
        encoder_out = model.encoder(
            src_tokens=sample['net_input']['src_tokens'],
            src_lengths=sample['net_input']['src_lengths'],
        )
        
        # Compute contrastive loss
        contrastive_loss = self.compute_contrastive_loss(encoder_out, sample)
        
        # Combine losses
        total_loss = loss + self.contrastive_weight * contrastive_loss
        
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(total_loss.data) if reduce else total_loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'contrastive_loss': utils.item(contrastive_loss.data) if reduce else contrastive_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        
        return total_loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        LabelSmoothedCrossEntropyCriterion.reduce_metrics(logging_outputs)
        
        # Aggregate contrastive loss
        contrastive_loss_sum = sum(log.get('contrastive_loss', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        
        metrics.log_scalar(
            'contrastive_loss', contrastive_loss_sum / sample_size / math.log(2), sample_size, round=3
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True