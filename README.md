# Quantifying Job Description Complexity and Its Association to Freelance Earnings Using NLP
This repository hosts the code, data, and documentation for a thesis project examining the relationship between professional skills, country economic classification, and salary outcomes. The study extracts and quantifies skills from job-related data, conducts regression analysis, and evaluates the impact of skills and country categories (e.g., first-world, second-world) on salary levels.

# Project Overview
The project investigates whether skills and a country’s economic classification influence salaries. Using natural language processing (NLP) to extract skills from datasets like job postings or resumes, it categorizes countries by economic development (e.g., GDP-based tiers) and applies regression models to quantify their effects on earnings.

# Get Started

### Clone Repository
1. Open cmd then run:
```
git clone https://github.com/luvellieee/Thesis-2.git
```

### Setting up Virtual Environment
1. Open cmd
2. Install python version 3.11
3. Open the cloned folder in vscode, then create a virtual environment. In search bar, click on
```
Show and run commands
Python: Create Environment
Venv
Python 3.11
```
4. Once you have successfully set up the folder for your virtual environment, open Git Bash then run (per line):
```
cd [path to the thesis folder]
source .venv/Scripts/activate
```

### Login to huggingface
1. Install huggingface (git bash)
```
pip install huggingface_hub
```
2. To check if you have successfuly installed huggingface, run (git bash):
```
pip show huggingface_hub
```
3. In the huggingface website, sign up
4. After you have already signed up, click your profile in the upper-right corner
5. Click on 'Access Token'
6. Click on '+ Create new token'
7. Name it however u want
8. Choose all under 'Repositories'
9. Click 'create token'
10. Copy and paste the token somewhere safe (you're gonna use it in the future)
11. In Git Bash, run:
```
from huggingface_hub import login
login() [tap enter]
[paste the api key]
```
12. To check if the login was successful, run (git bash):
```
hf auth whoami
```

### Wandb
1. Sign up in wandb website
2. In dashboard, click profile
3. Click API key
4. Copy api key
5. Install wandb (in bash)
```
pip install wandb
```
6. To check if you have successfully installed wandb, run (in bash)
```
pip show wandb
```
7.Login in wandb (in bash)
```
wandb login
```
8. Paste API key
9. To check if login was successful, run (in bash):
```
wandb login
```

### Data Preparation (skip this bc i finished it already)

### Training
run (in bash)
```
Skill-Extraction/src/training.sh
```