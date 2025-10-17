A lightweight and educational implementation of diffusion models for generative AI, built with PyTorch.

## 🚀 Features

- **Simple Diffusion Implementation**: Clean and understandable codebase for learning diffusion models
- **Training Pipeline**: Complete training workflow with customizable parameters
- **Generation Module**: Generate new samples from trained diffusion models
- **Dataset Support**: Compatible with various image datasets (including cat/dog 64x64 pixel dataset)
- **Modular Design**: Separated components for model, dataset, training, and generation

## 📁 Project Structure

```

my-simple-diffuser/
├──model.py          # Diffusion model architecture
├──train.py          # Training script and loop
├──generate.py       # Sample generation utilities
├──dataset.py        # Data loading and preprocessing
├──data/             # Dataset directory
├──requirements.txt  # Project dependencies
└──README.md         # Project documentation

```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/korob874-coder/my-simple-diffuser.git
cd my-simple-diffuser
```

1. Install dependencies:

```bash
pip install -r requirements.txt
```

🏃‍♂️ Quick Start

Training

```bash
python train.py --dataset path/to/your/dataset --epochs 100
```

Generation

```bash
python generate.py --checkpoint path/to/checkpoint.pth --samples 10
```

📊 Results

The model learns to generate coherent images through the diffusion process:

· Forward Process: Gradually add noise to data
· Reverse Process: Learn to denoise and generate new samples

🎯 Use Cases

· Educational purposes for understanding diffusion models
· Research experiments with different architectures
· Starting point for more complex generative AI projects

🤝 Contributing

Feel free to fork this project and submit pull requests for any improvements!

📝 License

This project is open source and available under the MIT License.



Built with ❤️ using PyTorch and Google Colab
