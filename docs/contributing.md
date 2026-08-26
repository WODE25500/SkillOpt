# Contributing to SkillOpt

Thank you for your interest in contributing to SkillOpt! This guide covers how to get started.

## Development Setup

```bash
git clone https://github.com/microsoft/SkillOpt.git
cd SkillOpt
pip install -e ".[dev]"
```

## Ways to Contribute

### 🐛 Bug Reports

Open an issue with:

- Steps to reproduce
- Expected vs actual behavior
- Config file used (sanitize API keys)
- Python version and OS

### 🔧 New Benchmark

See [Add a New Benchmark](guide/new-benchmark.md) for the implementation guide.

**Checklist:**

- [ ] Data loader in `skillopt/envs/<benchmark>/dataloader.py`
- [ ] Scored rollout implementation in `skillopt/envs/<benchmark>/rollout.py`
- [ ] Per-item `predictions/<id>/conversation.json` artifacts for shared reflection
- [ ] Environment adapter in `skillopt/envs/<benchmark>/adapter.py`
- [ ] Config file in `configs/<benchmark>/default.yaml`
- [ ] Lazy registration in `scripts/train.py` and `scripts/eval_only.py`
- [ ] Focused tests and an optional seed skill referenced by `env.skill_init`
- [ ] Documentation update

### 🤖 New Model Backend

See [Add a New Model Backend](guide/new-backend.md) for the implementation guide.

**Checklist:**

- [ ] Function-based backend module in `skillopt/model/<name>_backend.py`
- [ ] Alias and default model in `skillopt/model/common.py`
- [ ] Optimizer/target whitelist entries in `skillopt/model/backend_config.py`
- [ ] Dispatch, token tracking, and setter forwarding in `skillopt/model/__init__.py`
- [ ] YAML/CLI wiring when the backend exposes structured config fields
- [ ] Focused routing, configuration, tool-call, and token-accounting tests
- [ ] `.env.example` and backend/configuration reference updates

### 📝 Documentation

Documentation is built with MkDocs Material:

```bash
pip install -e ".[docs]"
mkdocs serve  # Preview at http://localhost:8000
```

## Code Style

- Follow existing patterns in the codebase
- Use type hints for function signatures
- Keep docstrings concise

## Pull Request Process

1. Fork the repository
2. Create a feature branch: `git switch -c feature/my-benchmark`
3. Make your changes
4. Run focused tests, the full test suite, and `mkdocs build --strict` when docs change
5. Submit a PR with a clear description

## Code Review Guidelines

We maintain a [code review & contribution guide](review_guidelines.md) distilled
from maintainer review feedback. Reviewers and contributors should use it as a
shared checklist before merging.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
