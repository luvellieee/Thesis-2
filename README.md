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
1. In the huggingface website, sign up
2. After you have already signed up, click your profile in the upper-right corner
3. Click on 'Access Token'
4. Click on '+ Create new token'
5. Name it however u want
6. Choose all under 'Repositories'
7. Click 'create token'
8. Copy and paste the token somewhere safe (you're gonna use it in the future)
9. In Git Bash, run:
```
from huggingface_hub import login
login() [tap enter]
[paste the api key]
```
