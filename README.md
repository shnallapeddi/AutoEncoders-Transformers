## Autoencoders & Transformers — Experiments and Applied NLP/CV
A notebook suite that spans self-supervised reconstruction (autoencoders), from-scratch Transformer implementations, and applied Transformer/LLM workflows for classification and summarization, plus a Vision Transformer (ViT) for image classification. The goal is to show end-to-end modeling patterns across modalities with clear training/evaluation scaffolding.

### Contents
1. Autoencoders for Anamoly Detection.ipynb — reconstruction-based anomaly detection with autoencoders (AE). Anomaly score = reconstruction error; includes thresholding and curve-based calibration. 
2. Building Transformer with Pytorch.ipynb — encoder/decoder blocks from scratch: scaled dot-product attention, multi-head attention, sinusoidal positional encodings, FFN, residuals + LayerNorm, padding/causal masks, and a minimal train/eval loop. 
3. NLP Transformer.ipynb — tokenization → batching/masking → Transformer for sequence tasks with accuracy/F1 style evaluation. 
4. Classification using pre-trained LLMs.ipynb — Hugging Face workflow for sequence classification with a pretrained encoder(-decoder) model; fine-tuning/evaluation pipeline. 
5. Summarization using LLMs-Billsum dataset.ipynb — abstractive summarization on BillSum with a pretrained seq2seq model; ROUGE scoring & length control. 
6. Summarization using LLMs-Multi-news dataset.ipynb — multi-document summarization on Multi-News with a pretrained seq2seq model; ROUGE evaluation. 
7. Vision Transformer (ViT) for Image Classification.ipynb — patch embedding → Transformer encoder stack → MLP head; training & evaluation on a small image dataset. 
8. Autoencoder and Transformer Architectures : Datasets.rtf — dataset notes/references used across notebooks. 
9. Lohit-Devanagari-Dataset for NLP Transformer.ttf — font asset for rendering/tokenization experiments in the NLP notebooks.

### Technical highlights
### Autoencoder anomaly detection

1. Model: convolutional/dense encoder → bottleneck → decoder with non-linearities; reconstruction loss (MSE/BCE depending on data).
2. Anomaly scoring: per-sample reconstruction error; optional feature-space error; threshold via ROC-AUC / PR-AUC sweep or percentile heuristic.
3. Regularization: dropout/weight-decay; early stopping on val loss; optional denoising (add noise at input) to improve robustness.
4. Interpretability: visualize reconstructions and error heatmaps to localize anomalies.

### Transformer (from scratch)
1. Attention core: scaled dot-product <img width="142" height="25" alt="image" src="https://github.com/user-attachments/assets/5ecb1b68-328e-4ebf-8937-ee3f2df689c5" /> with multi-head projections; sinusoidal positional encodings added to token embeddings.
2. Blocks: Encoder/Decoder stacks with LayerNorm → MHA → residual and LayerNorm → positionwise FFN → residual; padding masks (encoder) and causal masks (decoder).
3. Optimization: Adam/AdamW; linear-warmup + cosine/step decay; label smoothing option for classification/seq2seq heads.
4. Data path: tokenization, padding, attention masks; teacher forcing for decoder targets when applicable.

### Pretrained LLM workflows
1. Classification: Hugging Face AutoTokenizer + AutoModelForSequenceClassification; stratified split; metric loop (accuracy/F1/precision/recall). Checkpointing best-val model and saving config/tokenizer for inference.
2. Summarization (BillSum & Multi-News): encoder-decoder models (e.g., BART/T5-family) with ROUGE-1/2/L evaluation; generation params num_beams, length_penalty, min_length, no_repeat_ngram_size, num_beams, length_penalty, min_length, no_repeat_ngram_size to balance faithfulness and concision.
3. Throughput: gradient accumulation for small VRAM; mixed precision (fp16) when available.

### Vision Transformer (ViT)
1. Patchify images → linear embed → add learnable or sinusoidal position encodings → Transformer encoder layers → CLS token head.
2. Training: cross-entropy loss, AdamW with weight decay, cosine schedule; light augmentations (resize/normalize, flips).
3. Evaluation: top-1 accuracy; confusion matrix; per-class precision/recall.

### Start
##### clone
git clone https://github.com/shnallapeddi/AutoEncoders-Transformers
cd AutoEncoders-Transformers

##### environment
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install jupyter numpy pandas scikit-learn matplotlib

##### deep learning stacks (install what your notebook uses)
pip install torch torchvision torchaudio          # PyTorch
pip install transformers datasets evaluate rouge-score  # HF Transformers + metrics

### Suggested runs
1. AE anomaly detection: train AE on “normal” subset → tune threshold on validation errors → report ROC-AUC/PR-AUC on mixed test set; visualize error maps.
2. Transformer from scratch: overfit a tiny batch first (sanity check), then enable schedule + label smoothing; inspect attention masks and gradient flow.
3. LLM classification: fine-tune, log F1/accuracy, save the best checkpoint + tokenizer, and test an inference cell on raw text.
4. BillSum/Multi-News: run ROUGE evaluation; sweep num_beams and length_penalty for summary quality vs speed; verify truncation/padding lengths.
5. ViT: compare ViT to a small CNN baseline on the same split; analyze misclassifications via confusion matrix.

