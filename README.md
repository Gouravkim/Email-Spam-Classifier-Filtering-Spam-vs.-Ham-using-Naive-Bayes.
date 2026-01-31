#  Email Spam Classifier - Indian Email Context

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-FF6B6B?style=for-the-badge&logo=gradio&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

An intelligent **Spam Email Detector** specifically trained on **Indian email patterns** using **Multinomial Naive Bayes** algorithm. This project achieves **99.70% accuracy** and includes a professional **Gradio web interface** for real-time email classification.

---

## 📊 Project Highlights

<p align="center">
  <img src="https://img.shields.io/badge/Accuracy-99.70%25-brightgreen?style=flat-square&logo=target" alt="Accuracy"/>
  <img src="https://img.shields.io/badge/Dataset-100K%20Emails-blue?style=flat-square&logo=database" alt="Dataset"/>
  <img src="https://img.shields.io/badge/Model-Naive%20Bayes-orange?style=flat-square&logo=python" alt="Model"/>
  <img src="https://img.shields.io/badge/Interface-Gradio-red?style=flat-square&logo=gradio" alt="Interface"/>
</p>

### 🎯 Key Features

- ✅ **99.70% Classification Accuracy** - Highly accurate spam detection
- 🇮🇳 **Indian Context Dataset** - Trained on 100,000 Indian emails with local patterns (Paytm, BSNL, Flipkart, etc.)
- 🌐 **Interactive Web Interface** - User-friendly Gradio UI with confidence scores
- ⚡ **Real-time Predictions** - Instant spam/ham classification
- 📊 **Confidence Scoring** - Shows prediction confidence percentage
- 🎨 **Professional Design** - Clean, dark-themed interface with example emails
- 🔍 **Detailed Explanations** - Provides reasoning for each prediction

---

## 📸 Live Demo Screenshots

### Spam Detection Example

<img width="1866" height="973" alt="Screenshot 2026-01-31 145100" src="https://github.com/user-attachments/assets/d6315a87-2eb3-485c-b007-33eac66bec90" />

**Example Spam Email:**
```
T-Mobile customer you may now claim your FREE CAMERA PHONE upgrade & a 
pay & go sim card for your loyalty. Call on 0845 021 3680.Offer ends 
28thFeb.T&C's apply
```
**Result:** 🚫 SPAM DETECTED! (Confidence: 99.99%)

---

### Ham (Legitimate) Detection Example

<img width="1857" height="959" alt="Screenshot 2026-01-31 145309" src="https://github.com/user-attachments/assets/2c1638be-2bb9-44c1-bdfc-5883e9b329b1" />

**Example Ham Email:**
```
Congratulations on your achievement!
```
**Result:** ✅ LEGITIMATE EMAIL (HAM) (Confidence: 99.95%)

---

## 📖 Table of Contents

- [How It Works](#-how-it-works)
- [Dataset Information](#-dataset-information)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [Web Interface Features](#-web-interface-features)
- [Code Explanation](#-code-explanation)
- [Technologies Used](#-technologies-used)
- [Results & Analysis](#-results--analysis)
- [Future Enhancements](#-future-enhancements)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔍 How It Works

The spam classifier uses **Multinomial Naive Bayes** algorithm based on **Bayes' Theorem**:

$$P(Spam|Email) = \frac{P(Email|Spam) \times P(Spam)}{P(Email)}$$

### 📋 Classification Pipeline

```
Input Email Text
        ↓
┌─────────────────────────┐
│  Text Vectorization     │ → CountVectorizer converts text to numbers
└─────────────────────────┘
        ↓
┌─────────────────────────┐
│  Feature Extraction     │ → Creates word frequency matrix
└─────────────────────────┘
        ↓
┌─────────────────────────┐
│  Naive Bayes Model      │ → Calculates probabilities
└─────────────────────────┘
        ↓
┌─────────────────────────┐
│  Classification Result  │ → Spam (1) or Ham (0)
└─────────────────────────┘
        ↓
   Confidence Score (%)
```

### 🧮 Algorithm Steps

1. **Training Phase:**
   - Load 100,000 labeled emails (spam/ham)
   - Convert text to numerical features using CountVectorizer
   - Train Multinomial Naive Bayes model
   - Learn probability distributions for spam and ham words

2. **Prediction Phase:**
   - Input new email text
   - Transform text using trained vectorizer
   - Calculate P(Spam|Email) and P(Ham|Email)
   - Choose class with higher probability
   - Return prediction with confidence score

---

## 📊 Dataset Information

### Dataset Overview

| Attribute | Details |
|-----------|---------|
| **Name** | Indian Email Dataset |
| **Total Emails** | 100,000 |
| **File Format** | CSV |
| **Columns** | `msg` (email text), `label` (spam/ham) |
| **Context** | Indian emails with local references |

### Data Distribution

```
Ham (Legitimate):  ████████████████████████  49,936 emails (49.94%)
Spam:              ████████████████████████  50,064 emails (50.06%)
```

**Perfectly Balanced Dataset** - Almost 50-50 distribution prevents model bias!

### Sample Data Examples

| Label | Email Message | Category |
|-------|--------------|----------|
| spam | "Congratulations! You won iPhone 15 from Flipkart. Claim at 2 lakhs" | Promotional Scam |
| ham | "Can we reschedule to 3:30 instead?" | Personal Message |
| ham | "Reminder: Doctor appointment tomorrow at 10:30 AM" | Reminder |
| spam | "BSNL bill overdue Rs 25000. Pay at 8912345678 immediately" | Payment Scam |
| spam | "FREE! Get instant loan upto Rs 5 lakhs without documents" | Financial Scam |
| ham | "Meeting rescheduled to 3:30 PM tomorrow" | Work Communication |
| spam | "Paytm KYC pending. Complete at 50000 or account blocked" | KYC Fraud |
| ham | "Happy birthday! Have a wonderful day!" | Personal Greeting |

### Indian Context Features

The dataset includes **Indian-specific patterns**:
- 💳 **Digital Payment Platforms:** Paytm, PhonePe, Google Pay
- 🛒 **E-commerce Sites:** Flipkart, Amazon India
- 📱 **Telecom Operators:** BSNL, Jio, Airtel, Vodafone
- 💰 **Currency:** Indian Rupees (Rs, lakhs, crores)
- 🏦 **Banking Terms:** KYC, NEFT, IFSC, UPI
- 📞 **Phone Numbers:** 10-digit Indian mobile numbers
- 🗓️ **Indian English:** Local phrases and expressions

---

## 📁 Project Structure

```
Email-Spam-Classifier/
│
├── 📊 indian_email_dataset_100k.csv          # Dataset with 100K Indian emails
├── 📓 Spam_Ham_Email_Detection.ipynb         # Main Jupyter Notebook with complete code
├── 🖼️ Screenshot_2026-01-31_125830.png       # Web UI - Spam detection result
├── 🖼️ Screenshot_2026-01-31_130002.png       # Web UI - Ham detection result
├── 📄 README.md                               # Project documentation (this file)
└── 🔒 .gitignore                              # Git ignore file

Project Components:
├── Data Loading & Exploration
├── Data Preprocessing (Label Encoding)
├── Train-Test Split (80-20)
├── Text Vectorization (CountVectorizer)
├── Model Training (Multinomial Naive Bayes)
├── Model Evaluation (99.70% Accuracy)
├── Manual Testing Examples
└── Gradio Web Interface
```

---

## 🚀 Installation

### Prerequisites

- **Python 3.7+** (Python 3.13 recommended)
- **Jupyter Notebook** or **JupyterLab**
- **pip** package manager

### Step-by-Step Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Gouravkim/Email-Spam-Classifier-Filtering-Spam-vs.-Ham-using-Naive-Bayes.git
   cd Email-Spam-Classifier-Filtering-Spam-vs.-Ham-using-Naive-Bayes
   ```

2. **Create Virtual Environment** (Recommended)
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Required Packages**
   ```bash
   pip install pandas numpy scikit-learn gradio jupyter
   ```

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

5. **Open the Notebook**
   - Open `Spam_Ham_Email_Detection_using_Naive_Bayes.ipynb`
   - Run all cells sequentially

### Package Dependencies

```txt
pandas>=1.5.0           # Data manipulation
numpy>=1.21.0           # Numerical operations
scikit-learn>=1.0.0     # Machine learning library
gradio>=4.0.0           # Web interface
jupyter>=1.0.0          # Notebook environment
```

---

## 💻 Usage

### Method 1: Run Complete Notebook

```bash
# Open Jupyter Notebook
jupyter notebook

# Run all cells in sequence:
# 1. Import libraries
# 2. Load dataset
# 3. Preprocess data
# 4. Train model
# 5. Evaluate accuracy
# 6. Launch Gradio interface
```

### Method 2: Python Script Usage

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
spam_df = pd.read_csv("indian_email_dataset_100k.csv")

# Prepare data
spam_df['spam'] = spam_df['label'].map({'spam': 1, 'ham': 0})
x = spam_df['msg']
y = spam_df['spam']

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Vectorize text
cv = CountVectorizer()
x_train_count = cv.fit_transform(x_train)

# Train model
model = MultinomialNB()
model.fit(x_train_count, y_train)

# Test accuracy
x_test_count = cv.transform(x_test)
accuracy = model.score(x_test_count, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Make prediction
email_test = ["FREE! Get instant loan upto Rs 1 lakh"]
email_count = cv.transform(email_test)
prediction = model.predict(email_count)[0]
print(f"Prediction: {'SPAM' if prediction == 1 else 'HAM'}")
```

### Method 3: Gradio Web Interface

```python
import gradio as gr

def predict_spam(email_text):
    if not email_text.strip():
        return "Please enter an email!", 0.0, ""
    
    email_count = cv.transform([email_text])
    prediction = model.predict(email_count)[0]
    probability = model.predict_proba(email_count)[0]
    
    if prediction == 1:
        result = "🚫 SPAM DETECTED!"
        confidence = probability[1] * 100
        explanation = "This email appears to be spam. Be cautious!"
    else:
        result = "✅ LEGITIMATE EMAIL (HAM)"
        confidence = probability[0] * 100
        explanation = "This email appears to be legitimate and safe."
    
    return result, confidence, explanation

# Launch interface
demo = gr.Interface(
    fn=predict_spam,
    inputs=gr.Textbox(lines=5, label="Email Message"),
    outputs=[
        gr.Textbox(label="Prediction Result"),
        gr.Number(label="Confidence (%)"),
        gr.Textbox(label="Explanation")
    ],
    title="Spam Email Detector"
)

demo.launch()
```

---

## 📈 Model Performance

### Classification Metrics

| Metric | Value |
|--------|-------|
| **Model Accuracy** | **99.70%** |
| **Training Time** | < 5 seconds |
| **Prediction Time** | < 0.01 seconds per email |
| **Training Samples** | 80,000 emails (80%) |
| **Testing Samples** | 20,000 emails (20%) |

### Performance Details

```python
Model Accuracy: 99.70%

Training Set: 80,000 emails
Testing Set:  20,000 emails
Random State: 42 (reproducible results)
```

### Why 99.70% Accuracy?

1. **Large Dataset** - 100,000 diverse emails for robust training
2. **Balanced Classes** - Equal spam/ham distribution prevents bias
3. **Indian Context** - Specialized patterns for Indian emails
4. **Naive Bayes** - Excellent for text classification tasks
5. **CountVectorizer** - Effective word frequency features

### Model Predictions

**Prediction Encoding:**
- `1` = **SPAM** ⚠️ (Dangerous/Fraudulent email)
- `0` = **HAM** ✅ (Legitimate/Safe email)

---

## 🌐 Web Interface Features

### Gradio Interface Components

1. **Input Section**
   - Multi-line text area for email input
   - Placeholder text guidance
   - Clear/Submit buttons

2. **Output Section**
   - **Prediction Result:** SPAM or HAM with emoji indicator
   - **Confidence Score:** Percentage confidence (0-100%)
   - **Explanation:** User-friendly message about the prediction

3. **Pre-loaded Examples**
   - 4 example emails (2 spam, 2 ham)
   - One-click testing
   - Demonstrates functionality

4. **Design Features**
   - Professional dark theme
   - Responsive layout
   - Clean typography
   - Intuitive user flow

### Interface Access

```python
# Local URL (after running notebook)
http://127.0.0.1:7862

# For public sharing (optional)
demo.launch(share=True)  # Creates temporary public link
```

---

## 📝 Code Explanation

### Step 1: Import Libraries

```python
import pandas as pd           # Data handling
import numpy as np           # Numerical operations
from sklearn.model_selection import train_test_split    # Data splitting
from sklearn.feature_extraction.text import CountVectorizer  # Text to numbers
from sklearn.naive_bayes import MultinomialNB           # ML model
```

### Step 2: Load Dataset

```python
spam_df = pd.read_csv("indian_email_dataset_100k.csv")
# Columns: 'msg' (email text), 'label' (spam/ham)
```

### Step 3: Label Encoding

```python
# Convert labels: 'spam' → 1, 'ham' → 0
spam_df['spam'] = spam_df['label'].map({'spam': 1, 'ham': 0})
```

### Step 4: Prepare Features

```python
x = spam_df['msg']    # Input: Email text
y = spam_df['spam']   # Output: 1 (spam) or 0 (ham)
```

### Step 5: Train-Test Split

```python
x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    test_size=0.2,      # 20% for testing
    random_state=42     # Reproducible results
)
# Training: 80,000 emails
# Testing: 20,000 emails
```

### Step 6: Text Vectorization

```python
cv = CountVectorizer()
x_train_count = cv.fit_transform(x_train)
# Converts text to word frequency matrix
# Example: "FREE loan" → [1, 1, 0, 0, ...]
```

### Step 7: Train Model

```python
model = MultinomialNB()
model.fit(x_train_count, y_train)
# Learns probability of each word in spam vs ham
```

### Step 8: Evaluate Accuracy

```python
x_test_count = cv.transform(x_test)
accuracy = model.score(x_test_count, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
# Output: Model Accuracy: 99.70%
```

### Step 9: Make Predictions

```python
email_test = ["Your property tax Rs 2 lakhs overdue"]
email_count = cv.transform(email_test)
prediction = model.predict(email_count)[0]
print(f"Prediction: {prediction}")  # Output: 1 (SPAM)
```

---

## 🛠️ Technologies Used

### Core Technologies

| Technology | Purpose | Version |
|------------|---------|---------|
| **Python** | Programming Language | 3.13.0 |
| **Pandas** | Data Manipulation | Latest |
| **NumPy** | Numerical Computing | Latest |
| **scikit-learn** | Machine Learning | Latest |
| **Gradio** | Web Interface | 4.0+ |
| **Jupyter** | Interactive Development | Latest |

### ML Components

- **CountVectorizer** - Converts text to numerical features (word counts)
- **MultinomialNB** - Naive Bayes classifier for discrete features
- **train_test_split** - Divides data into training/testing sets

### Why These Technologies?

1. **scikit-learn** - Industry-standard ML library, easy to use
2. **Naive Bayes** - Fast, efficient, perfect for text classification
3. **Gradio** - Creates web interfaces with minimal code
4. **Pandas** - Powerful data manipulation and CSV handling
5. **Jupyter** - Interactive development, easy visualization

---

## 📊 Results & Analysis

### Real-World Testing

#### Test Case 1: Spam Detection ⚠️

**Input Email:**
```
T-Mobile customer you may now claim your FREE CAMERA PHONE upgrade 
& a pay & go sim card for your loyalty. Call on 0845 021 3680.
Offer ends 28thFeb.T&C's apply
```

**Results:**
- **Prediction:** 🚫 SPAM DETECTED!
- **Confidence:** 99.9999999997868%
- **Explanation:** This email appears to be spam. Be cautious and don't click any links or provide personal information.

**Why It's Spam:**
- Keywords: "FREE", "claim", "Call", "Offer ends"
- Urgency tactics: Limited time offer
- Phone number included
- Too-good-to-be-true offer

---

#### Test Case 2: Ham Detection ✅

**Input Email:**
```
Congratulations on your achievement!
```

**Results:**
- **Prediction:** ✅ LEGITIMATE EMAIL (HAM)
- **Confidence:** 99.950933153200051%
- **Explanation:** This email appears to be legitimate and safe.

**Why It's Ham:**
- Simple, genuine message
- No suspicious keywords
- No urgency or scam patterns
- Natural language

---

### Additional Test Results

```python
# Test 1: Indian Spam
"Your property tax Rs 2 lakhs overdue. Pay at 8765432109"
→ Prediction: 1 (SPAM) ⚠️

# Test 2: Personal Message
"I HAVE A DATE ON SUNDAY WITH WILL!!"
→ Prediction: 0 (HAM) ✅

# Test 3: Financial Scam
"FREE! Get instant loan upto Rs 1 lakh without documents"
→ Prediction: 1 (SPAM) ⚠️

# Test 4: Reminder
"Reminder: Doctor appointment tomorrow at 10:30 AM"
→ Prediction: 0 (HAM) ✅
```

### Common Spam Indicators Detected

🚫 **Spam Patterns Identified:**
- "FREE", "WIN", "CONGRATULATIONS"
- Phone numbers with suspicious context
- "Rs X lakhs", urgent payment requests
- "KYC pending", "Account blocked"
- "Claim now", "Limited time"
- "Click here", "Call immediately"

✅ **Ham Patterns Identified:**
- Personal pronouns (I, we, you)
- Normal conversation language
- Specific times and dates
- Genuine reminders
- Professional communication

---

## 🔮 Future Enhancements

### Planned Features

- [ ] **Deep Learning Models** - Implement LSTM/BERT for improved accuracy
- [ ] **Email Header Analysis** - Check sender, subject, attachments
- [ ] **URL Detection** - Analyze links for phishing
- [ ] **Multi-language Support** - Support Hindi, Tamil, Telugu emails
- [ ] **API Development** - REST API for integration with email clients
- [ ] **Browser Extension** - Chrome/Firefox extension for Gmail
- [ ] **Mobile App** - Android/iOS app for email verification
- [ ] **Sender Reputation** - Track and score sender history
- [ ] **Real-time Learning** - Continuous model updates with user feedback
- [ ] **Attachment Scanning** - Malware detection in attachments

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repository and submit pull requests.

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 👨‍💻 Author

**Gourav**

- 🌐 GitHub: [@Gouravkim](https://github.com/Gouravkim)
- 🔗 Project Link: [Email Spam Classifier](https://github.com/Gouravkim/Email-Spam-Classifier-Filtering-Spam-vs.-Ham-using-Naive-Bayes)

---

<p align="center">
  <b>🎯 Built with ❤️ for Indian Email Users</b>
</p>

<p align="center">
  <sub>⚡ Powered by Multinomial Naive Bayes | 🐍 Built with Python | 🌐 Interface by Gradio</sub>
</p>

---

**Last Updated:** January 31, 2026  
**Status:** ✅ Active Development
