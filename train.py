import torch
import model

def main():
    print(f"Vocabulary size: {model.config['vocab_size']}")
    print(f"Model parameters: {sum(p.numel() for p in model.Model(model.config).parameters()):,}")
    print()
    trained_model = model.train_model()
    generated = model.generate_text(trained_model, model.ptt, max_new_tokens=500)
    print(generated)
    
    # Save the model
    torch.save(trained_model.state_dict(), 'shakespeare_model.pth')
    print(f"\nModel saved to shakespeare_model.pth")

if __name__ == "__main__":
    main()

