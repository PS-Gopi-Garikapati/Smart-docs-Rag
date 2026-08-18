# Deployment Documentation

This document explains deployment guidelines for local, staging and production environments.

## 1. Running locally
- Verify Python 3.10+ is installed on the host.
- Activate the virtual environment:
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
- Install target dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- Configure variables in `.env` (API key, Host, Port).
- Launch the server:
  ```bash
  python main.py
  ```

