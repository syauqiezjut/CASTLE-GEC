import torch
import os

def analyze_model_architecture(checkpoint_path):
    """Analyze model architecture in detail"""
    print(f"\n{'='*80}")
    print(f"DETAILED ANALYSIS: {os.path.basename(os.path.dirname(checkpoint_path))}")
    print(f"{'='*80}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Find model dict
        model_keys = ['model', 'model_state_dict', 'state_dict', 'net']
        model_dict = None
        
        for key in model_keys:
            if key in checkpoint:
                model_dict = checkpoint[key]
                break
        
        if model_dict is None and any(k.endswith('.weight') or k.endswith('.bias') for k in checkpoint.keys()):
            model_dict = checkpoint
        
        if model_dict is None:
            print("Could not find model state dict")
            return
        
        # Analyze embedding layers
        embedding_params = 0
        encoder_params = 0
        decoder_params = 0
        other_params = 0
        
        print(f"{'Layer Type':40} | {'Parameters':>15} | {'Shape':>20}")
        print(f"{'-'*40} | {'-'*15} | {'-'*20}")
        
        for name, param in model_dict.items():
            if isinstance(param, torch.Tensor):
                param_count = param.numel()
                shape_str = str(list(param.shape))
                
                # Categorize parameters
                if 'embed' in name.lower():
                    embedding_params += param_count
                    category = "EMBEDDING"
                elif 'encoder' in name.lower():
                    encoder_params += param_count
                    category = "ENCODER"
                elif 'decoder' in name.lower():
                    decoder_params += param_count
                    category = "DECODER"
                else:
                    other_params += param_count
                    category = "OTHER"
                
                # Print first few layers of each type
                if (embedding_params <= param_count * 5 and 'embed' in name.lower()) or \
                   (encoder_params <= param_count * 5 and 'encoder' in name.lower()) or \
                   (decoder_params <= param_count * 5 and 'decoder' in name.lower()):
                    print(f"{name[:40]:40} | {param_count:>15,} | {shape_str:>20}")
        
        total_params = embedding_params + encoder_params + decoder_params + other_params
        
        print(f"\n{'='*80}")
        print("PARAMETER BREAKDOWN:")
        print(f"Embedding Parameters: {embedding_params:>15,} ({embedding_params/total_params*100:.1f}%)")
        print(f"Encoder Parameters:   {encoder_params:>15,} ({encoder_params/total_params*100:.1f}%)")
        print(f"Decoder Parameters:   {decoder_params:>15,} ({decoder_params/total_params*100:.1f}%)")
        print(f"Other Parameters:     {other_params:>15,} ({other_params/total_params*100:.1f}%)")
        print(f"{'='*40}")
        print(f"TOTAL PARAMETERS:     {total_params:>15,}")
        
        # Check for vocabulary size hints
        for name, param in model_dict.items():
            if 'embed' in name.lower() and param.dim() == 2:
                vocab_size = param.shape[0]
                embed_dim = param.shape[1]
                print(f"\nVocabulary Size: {vocab_size:,}")
                print(f"Embedding Dimension: {embed_dim:,}")
                break
        
        # Check other checkpoint contents
        print(f"\nCheckpoint Contents: {list(checkpoint.keys())}")
        
        # Check if optimizer state is included
        if 'optimizer' in checkpoint:
            print("Optimizer state: INCLUDED (this adds significant size)")
        
        if 'lr_scheduler' in checkpoint:
            print("LR Scheduler state: INCLUDED")
            
    except Exception as e:
        print(f"Error: {str(e)}")

# Analyze problematic models
models_to_check = [
    "/ssd-data1/sq2023/belajar/checkpoints/baseline/checkpoint_best.pt",
    "/ssd-data1/sq2023/belajar/checkpoints/bpe/checkpoint_best.pt",
    "/ssd-data1/sq2023/belajar/checkpoints/unigram/checkpoint_best.pt",
    "/ssd-data1/sq2023/belajar/checkpoints/wordpiece/checkpoint_best.pt",
    "/ssd-data1/sq2023/belajar/checkpoints/wordpiece_linked/checkpoint_best.pt",
    "/ssd-data1/sq2023/belajar/checkpoints/wordpiece_kg_gated_linked/checkpoint_best.pt"
]

for model_path in models_to_check:
    if os.path.exists(model_path):
        analyze_model_architecture(model_path)
    else:
        print(f"File not found: {model_path}")