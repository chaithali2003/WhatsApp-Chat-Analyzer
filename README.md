# WhatsApp Chat Analyzer

The **WhatsApp Chat Analyzer** is a Machine Learning based application that analyzes exported WhatsApp chat `.txt` files and extracts meaningful insights such as message statistics, emoji usage, late-night activity, quick replies, and outlier detection using the **DBSCAN clustering algorithm**.

This project focuses on analyzing real chat behavior patterns while applying strict filtering rules to ensure accurate and meaningful results.

---

## 🚀 Features

- Upload and analyze WhatsApp `.txt` chat files  
- Total message count (per user and overall)  
- Emoji analysis (counts emojis only inside text messages)  
- Late-night message detection (12 AM – 4 AM)  
- Quick reply detection (≤ 5 minutes)  
- Outlier detection using **DBSCAN**  
- Clean and simple web-based interface  

---

## 🛠️ Technology Stack

**Frontend**
- HTML
- CSS
- JavaScript

**Backend**
- Python
- Flask

**Machine Learning**
- DBSCAN (from scikit-learn)

**Libraries**
- pandas
- numpy
- regex
- scikit-learn

---

## 📁 Project Structure

```
WhatsApp-Chat-Analyzer/
│
├── app.py                # Main application file
├── requirements.txt      # Python dependencies
├── templates/            # HTML templates
├── static/               # CSS and JS files
└── README.md             # Project documentation
```

---

## ⚙️ Installation & Setup

1. Clone the repository
```bash
git clone https://github.com/chaithali2003/WhatsApp-Chat-Analyzer.git
cd WhatsApp-Chat-Analyzer
```

2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate    # Windows
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Run the application
```bash
python app.py
```

Open your browser and go to:
```
http://127.0.0.1:5000
```

---

## 📊 Output Template

```
Chat Timeline
Start: DD-MM-YYYY
End: DD-MM-YYYY

Total Messages
Total: sum
User 1: count
User 2: count

Total Emoji
Total: sum
User 1: count
User 2: count

Late-night Messages (12AM - 4AM)
Total: count

Quick Replies (≤5m)
Total: count

Outliers
Total: count
```

---

## 📌 DBSCAN Usage

DBSCAN is used to identify:
- Irregular communication patterns  
- Sparse or noisy message events  
- Deleted messages as outliers  

The algorithm does not require a predefined number of clusters and works efficiently with real-world chat data.

---

## 🎯 Use Case

- Academic mini-project  
- Machine Learning project  
- Chat behavior analysis  
- Data preprocessing demonstration  

---

## 👩‍💻 Author

Made with ❤️ by **Chaithali S**  

**Chaithali S**  
GitHub: https://github.com/chaithali2003

---

⭐ If you like this project, don’t forget to star the repository!
