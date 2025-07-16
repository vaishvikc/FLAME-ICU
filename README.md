# FLAME-ICU
Federated Learning Adaptable Mortality Estimator for the ICU

## Environment Setup

This project uses a Python virtual environment named `flameICU`. Follow the instructions below to set up your environment.

### Prerequisites

- Python 3.8 or higher

### Setting up the Virtual Environment

#### macOS/Linux

1. Open Terminal
2. Navigate to the project directory:
   ```bash
   cd /path/to/FLAME-ICU
   ```
3. Create the virtual environment:
   ```bash
   python3 -m venv flameICU
   ```
4. Activate the virtual environment:
   ```bash
   source flameICU/bin/activate
   ```
5. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
6. To deactivate the environment when done:
   ```bash
   deactivate
   ```

#### Windows

1. Open Command Prompt or PowerShell
2. Navigate to the project directory:
   ```cmd
   cd C:\path\to\FLAME-ICU
   ```
3. Create the virtual environment:
   ```cmd
   python -m venv flameICU
   ```
   Note: If `python` doesn't work, try `py` or `python3`
4. Activate the virtual environment:
   - In Command Prompt:
     ```cmd
     flameICU\Scripts\activate.bat
     ```
   - In PowerShell:
     ```powershell
     .\flameICU\Scripts\Activate.ps1
     ```
5. Install required packages:
   ```cmd
   pip install -r requirements.txt
   ```
6. To deactivate the environment when done:
   ```cmd
   deactivate
   ```

### Important Notes

- Always activate the virtual environment before working on the project
- The `requirements.txt` file contains all necessary dependencies
- If adding new packages, update the requirements file using:
  ```bash
  pip freeze > requirements.txt
  ```
