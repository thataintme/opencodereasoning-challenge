# OpenCodeReasoning Submission

This repository contains three main folders:

- `/leapla-ocr-frontend` – React UI application
- `/kaggle_notebooks` – Working Kaggle `.ipynb` notebooks used for model development
- `/early_experiments_python` – Experimental Python scripts (early-stage / testing code)

## Model outputs:
- `https://huggingface.co/santhosh-m/ocr2-sft-lora-merged-v2` - Stage 1 output (SFT)
- `https://huggingface.co/santhosh-m/ocr2-grpo-lora-merged-v2` - Stage 2 output (RL)
Refer to `/kaggle_notebooks/dataset-curation-ocr2-challenge.ipynb` for detailed description and pipeline workflow

## Running the Frontend

Navigate to the frontend folder:

```bash
cd leapla-ocr-frontend
```

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm start
```

The app will run locally at http://localhost:3000

## Note
The UI is not connected to AI at the moment due to lack of free tier deployment services.
That said, the repository includes working Kaggle notebooks that showcase the model experimentation, OCR pipeline development, and reasoning workflows behind the project.
