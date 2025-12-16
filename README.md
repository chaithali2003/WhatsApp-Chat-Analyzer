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

## 🧠 Message Filtering Rules

### Message Count Includes
- Normal text messages  
- Messages containing **text + emoji**  
- Messages containing links  

### Message Count Excludes
- Emoji-only messages  
- "This message was deleted"  
- System-generated messages  

### Emoji Count Rules
- Emojis inside text messages are counted  
- Each emoji is counted individually  
- Repeated emojis are counted every time  
- Emoji-only messages are ignored  

### Outliers
- Messages with the text **"This message was deleted"** are treated as outliers and noise points for DBSCAN

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

## 📊 Sample Output

```
Total Messages
Total: 139
User 1: 83
User 2: 56

Total Emoji
Total: 8
User 1: 6
User 2: 2

Late-night Messages (12AM - 4AM)
Total: 0

Quick Replies (≤5m)
Total: 48

Total Chat Days: 37

Outliers
Total: 2
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

## 📜 License

This project is open-source and available under the **MIT License**.

---

## 👩‍💻 Author

**Chaithali S**  
GitHub: https://github.com/chaithali2003

---

⭐ If you like this project, don’t forget to star the repository!
