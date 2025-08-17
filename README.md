# Predicting User Churn in Apple Music

This project uses machine learning to predict **user churn in Apple Music** based on user behavior data.  
Three models were implemented and compared:  
- Logistic Regression  
- Decision Tree  
- Random Forest  

The project evaluates model performance using **confusion matrices, ROC and Precision-Recall curves, feature importance analysis, and learning curves**.  
It also demonstrates how predicted churn probabilities can be translated into **actionable business interventions** (e.g., sending discounts or recommendations).

---

## 📂 Project Structure
📁 apple-music-churn
- 📄 apple_music_churn.py # Main script with models and evaluation
- 📄 AppleMusic_Churn_Converted.csv # Dataset (if sharable)
- 📄 requirements.txt # Python dependencies
- 📄 README.md # Project documentation

## ⚙️ Requirements

- Packages listed in `requirements.txt`

Install dependencies with:  
```bash
pip install -r requirements.txt

▶️ How to use:

Clone the repository:

git clone https://github.com/42Duff/apple-music-churn.git
cd apple-music-churn


Make sure the dataset file AppleMusic_Churn_Converted.csv is in the project folder.

Run the script:

python apple_music_churn.py
