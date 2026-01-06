# Contributing to Soynade Research

Thank you for helping us illuminate the digital horizon! We welcome contributions of all kinds: code, data, documentation, bug reports, and feedback.

## Pull Request Workflow

1. **Fork** the repository
2. **Create a branch** for your feature:
```bash
   git checkout -b feature/your-feature-name
```
3. **Make your changes** with clear commit messages
4. **Test your changes** locally (run `pytest` if applicable)
5. **Push** to your fork
6. **Open a Pull Request** targeting the `main` branch

### PR Requirements
- At least **one review** from a core maintainer
- All CI/CD checks must pass (linting, tests)
- Clear description of what changed and why

## Branch Protection Rules
- Direct commits to `main` are blocked
- All changes must come via reviewed pull requests
- No force pushes to protected branches

## Code Style

### Python
- Use **ruff** for formatting and linting 
- Include docstrings for all functions
- Add type hints where helpful

### Documentation
- Keep language simple and clear
- Include examples
- Update README if you add features

## Testing
- Add tests for new features
- Ensure existing tests still pass
- Run full test suite before submitting: `pytest`

## Commit Messages
Good:

- Implements custom tokenizer for Bambara
- Fixing circular import issue in run_pipeline.py
  
Bad:
- fixing errors

## Types of Contributions

### Code
- Bug fixes, new features, performance improvements

### Data
- New datasets, corrections to existing data, evaluation sets

### Documentation
- Tutorials, examples, translations, clarifications

### Testing
- Bug reports, test cases, validation

### Community
- Answering questions, helping newcomers, organizing events

## Recognition
All contributors are acknowledged in:
- Repository README
- Release notes
- Annual community reports
- Research papers (for significant contributions)

## Code of Conduct
Please read our [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). We are committed to providing a welcoming and inclusive environment.

---
*The moon shines for all. Your contribution matters.* 