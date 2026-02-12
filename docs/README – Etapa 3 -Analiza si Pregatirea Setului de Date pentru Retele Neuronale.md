# 📘 README – Etapa 3: Analiza și Pregătirea Setului de Date pentru Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Cretu Alexandru Mihai 
**Data:** 20/11/2025

---

## Introducere

Acest document descrie activitățile realizate în **Etapa 3**, în care se analizează și se preprocesează setul de date necesar proiectului „Rețele Neuronale". Scopul etapei este pregătirea corectă a datelor pentru instruirea modelului RN, respectând bunele practici privind calitatea, consistența și reproductibilitatea datelor.



---

##  2. Analiza Exploratorie a Datelor (EDA) – Sintetic

### 2.1 Statistici descriptive aplicate

* **Medie, mediană, deviație standard**
* **Min–max și quartile**
* **Distribuții pe caracteristici**
* **Identificarea outlierilor**

### 3.2 Analiza calității datelor

* **Detectarea valorilor lipsă** (% pe coloană)
* **Detectarea valorilor inconsistente sau eronate**
* **Identificarea caracteristicilor redundante sau puternic corelate**

### 3.3 Probleme identificate

* Mici greseli de indentificare a glitchurilor de programul automat, necesita review manual

---

##  4. Preprocesarea Datelor

### 4.1 Curățarea datelor

* Video Feedul ramane acelasi
* Logurile YOLOului sunt trecute printr-un labeler automat, necesita review manual pentru cele nesigure

### 4.2 Transformarea caracteristicilor

* Transformam detectiile YOLOului in Loguri. Luam in considerare "varsta" detectiilor si increderea YOLOului 

### 4.3 Structurarea seturilor de date

* 70% Train
* 15% Validation
* 15% Test

### 4.4 Salvarea rezultatelor preprocesării

* Date preprocesate în `data/processed/`
* Seturi train/val/test în foldere dedicate

---

##  5. Fișiere Generate în Această Etapă

* `data/raw/` – date brute
* `data/processed/` – date curățate & transformate
* `data/train/`, `data/validation/`, `data/test/` – seturi finale
* `src/preprocessing/` – codul de preprocesare
* `data/README.md` – descrierea dataset-ului

---

##  6. Stare Etapă (de completat de student)

- [X] Structură repository configurată
- [ ] Dataset analizat (EDA realizată)
- [X] Date preprocesate
- [X] Seturi train/val/test generate
- [X] Documentație actualizată în README + `data/README.md`

---
