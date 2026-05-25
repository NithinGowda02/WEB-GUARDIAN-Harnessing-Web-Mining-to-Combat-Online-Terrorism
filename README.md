# WEB GUARDIAN – Harnessing Web Mining to Combat Online Terrorism

## Overview

WEB GUARDIAN is a web mining and machine learning based application designed to detect and analyze terrorism-related content on websites.

With the rapid growth of the internet, extremist organizations increasingly use websites and online platforms to spread propaganda, recruit members, and distribute harmful content. WEB GUARDIAN addresses this problem by automatically scanning webpage content, extracting textual data, and identifying suspicious patterns using Natural Language Processing (NLP) and Machine Learning techniques.

The system helps classify websites based on their content and generates a risk score to indicate the likelihood of terrorism-related material.

This project is inspired by research on using web mining and machine learning to detect online terrorist activity. :contentReference[oaicite:0]{index=0}

---

## Problem Statement

Terrorist organizations often misuse online platforms to spread extremist ideologies and propaganda.

Traditional manual monitoring is time-consuming and difficult because of the massive amount of web content generated every day.

WEB GUARDIAN solves this by providing an automated system that:

- scans website content
- extracts meaningful text from webpages
- analyzes suspicious keywords and patterns
- predicts whether the webpage may contain terrorism-related content
- provides a risk score for review

---

## Objectives

- Detect online terrorism-related content from websites
- Apply web mining techniques to extract webpage text
- Use machine learning models for classification
- Generate risk-based analysis of webpages
- Support safer web monitoring through automated detection

---

## Features

### URL Scanning
Users can enter any webpage URL into the application.

The system fetches the webpage and extracts readable content for analysis.

---

### Web Scraping
Extracts text from webpage HTML using web parsing libraries.

Removes:
- scripts
- navigation elements
- unwanted HTML tags
- page noise

---

### NLP-Based Text Processing
Processes extracted text using Natural Language Processing techniques:

- Tokenization
- Stopword Removal
- Stemming / Lemmatization
- Keyword Frequency Analysis

---

### Machine Learning Classification
Classifies webpages into categories such as:

- Safe
- Suspicious
- Potentially Terrorism Related

Possible algorithms evaluated:

- Logistic Regression
- Naive Bayes
- Decision Tree
- K-Nearest Neighbors
- Random Forest

Research in this area often reports Random Forest as highly effective for classification tasks. :contentReference[oaicite:1]{index=1}

---

### Risk Score Generation
Each scanned webpage receives a risk score based on:

- keyword frequency
- content relevance
- model prediction confidence

Example:

0–30 → Safe

31–60 → Needs Review

61–100 → High Risk

---

### History Tracking
Stores previous scans for future review.

Includes:

- scanned URLs
- results
- scores
- timestamps

---

## Tech Stack

### Frontend
- HTML
- CSS
- JavaScript

### Backend
- Python

### Libraries
- BeautifulSoup
- Pandas
- NumPy
- Scikit-learn
- NLTK

### Machine Learning
- Random Forest
- Naive Bayes
- Logistic Regression
- KNN

---

## System Workflow

```text
User enters URL
      ↓
Fetch webpage content
      ↓
Extract visible text from HTML
      ↓
Preprocess text
      ↓
Feature extraction
      ↓
Machine learning prediction
      ↓
Risk score generation
      ↓
Display result
