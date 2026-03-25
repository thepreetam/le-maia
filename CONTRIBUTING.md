# Contributing to LeWM-VC

Thank you for your interest in contributing to LeWM-VC!

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/lewm-vc/lewm-vc.git
   cd lewm-vc
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Verify installation**
   ```bash
   python -c "from lewm_vc import LeWMEncoder, LeWMDecoder; print('OK')"
   ```

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) with 100 character line length
- Use type hints for function signatures
- Use `ruff` for linting:
  ```bash
  ruff check src/ tests/
  ```

## Testing Guidelines

- All new code should include tests
- Run the full test suite before submitting:
  ```bash
  pytest tests/ -v
  ```
- Run specific test files:
  ```bash
  pytest tests/test_encoder.py -v
  ```

## Project Structure

```
lewm-vc/
├── src/
│   └── lewm_vc/
│       ├── encoder.py        # ViT encoder
│       ├── decoder.py        # Decoder network
│       ├── predictor.py      # Motion prediction
│       ├── quant.py         # Quantization
│       ├── entropy.py       # Entropy coding
│       ├── bitstream/       # NAL serialization
│       └── utils/          # Utilities
├── tests/                   # Test suite
├── configs/                # Configuration files
├── ffmpeg/                # FFmpeg plugin
└── checkpoints/           # Model checkpoints
```

## Submitting Changes

1. Create a branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and add tests

3. Run the test suite:
   ```bash
   pytest tests/ -v
   ruff check src/ tests/
   ```

4. Commit your changes:
   ```bash
   git add .
   git commit -m "Add your feature"
   ```

5. Push and create a pull request

## Code Review Process

- All submissions require review
- Address feedback promptly
- Ensure tests pass before requesting review

## Reporting Issues

- Use GitHub Issues for bug reports
- Include reproduction steps
- Include environment details (Python version, PyTorch version, etc.)
