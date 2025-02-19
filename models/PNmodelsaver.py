import torch
from pathlib import Path

class ModelSaver:
    @staticmethod
    def save_checkpoint(framework, epoch, loss, output_dir: Path, is_best: bool = False, save_full: bool = False):
        """
        Save model checkpoints with options to save full model or separate encoder/decoder
        
        Args:
            framework: FeatureLearningFramework instance
            epoch: Current epoch number
            loss: Current loss value
            output_dir: Directory to save checkpoints
            is_best: Whether this is the best model so far
            save_full: Whether to save the full model or just encoder/decoder separately
        """
        # Prepare common checkpoint data
        checkpoint_base = {
            'epoch': epoch,
            'optimizer_state_dict': framework.optimizer.state_dict(),
            'loss': loss,
            'feature_dim': framework.model.encoder.feature_dim,
            'num_joints': framework.model.encoder.mlp[0].in_features - 6  # Subtract position and offset dims
        }
        
        # Save encoder
        encoder_checkpoint = {
            **checkpoint_base,
            'encoder_state_dict': framework.model.encoder.state_dict()
        }
        
        # Save decoder
        decoder_checkpoint = {
            **checkpoint_base,
            'decoder_state_dict': framework.model.decoder.state_dict()
        }
        
        # Determine file names
        suffix = '_best' if is_best else f'_epoch_{epoch}'
        encoder_path = output_dir / f'encoder{suffix}.pth'
        decoder_path = output_dir / f'decoder{suffix}.pth'
        full_model_path = output_dir / f'full_model{suffix}.pth'
        
        # Save separate components
        torch.save(encoder_checkpoint, encoder_path)
        torch.save(decoder_checkpoint, decoder_path)
        
        # Optionally save full model
        if save_full:
            full_checkpoint = {
                **checkpoint_base,
                'model_state_dict': framework.model.state_dict()
            }
            torch.save(full_checkpoint, full_model_path)
    
    @staticmethod
    def load_encoder(checkpoint_path: str, device: torch.device = None):
        """
        Load only the encoder part of the model
        
        Args:
            checkpoint_path: Path to the encoder checkpoint
            device: Device to load the model to
        Returns:
            Loaded encoder model and checkpoint data
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Create new encoder with saved parameters
        from PointNet import VertexEncoder
        encoder = VertexEncoder(
            num_joints=checkpoint['num_joints'],
            feature_dim=checkpoint['feature_dim']
        )
        
        # Load state dict
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        encoder.to(device)
        
        return encoder, checkpoint
