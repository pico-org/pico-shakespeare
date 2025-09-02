# pico-shakespeare
A minimal Transformer based language model inspired by Andrej Karpathy's nanoGPT, trained from scratch on Shakespeare's works. Built to be simple, educational, and capable of generating text in Shakespearean style.

## Features
- Clean, modular architecture with separate dataset, config, and model files
- Proper transformer implementation with multi-head attention
- Layer normalization and dropout for better training stability
- Residual connections and feed-forward layers
- Dynamic vocabulary size detection
- Easy-to-use training and generation functions

## Usage

### Quick Start
```bash
python train.py
```

### Manual Usage
```python
import model

# Train the model
trained_model = model.train_model()

# Generate text
generated_text = model.generate_text(trained_model, model.tok, max_new_tokens=500)
print(generated_text)
```

## Architecture
- **Embedding Layer**: Token + positional embeddings
- **Transformer Blocks**: Multi-head self-attention with residual connections
- **Feed-Forward**: Expanded hidden dimension with ReLU activation
- **Output Layer**: Linear projection to vocabulary size

## Files
- `config.py`: Model configuration and hyperparameters
- `dataset.py`: Data loading, tokenization, and batching utilities  
- `model.py`: Main transformer model implementation and training loop
- `train.py`: Simple training script
- `GPT_Karypathy.ipynb`: Jupyter notebook with step-by-step implementation
