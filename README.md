# SheMaps-Internship-Safety-For-Women
# 🗺️ SheMaps: Internship Safety Recommender for Women

**Where Safety Meets Success ✨**  
SheMaps is a smart internship city recommender system designed specially for women. It helps students evaluate and compare cities for internships based on **crime rate**, **PG availability**, **affordability**, and **distance from home**. 🌆

---

##  Features

-  **City Recommendation Engine**  
  Suggests the best city out of 4 chosen cities using a personalized scoring model & ML logic.

-  **Auto Distance Calculator**  
  Computes distance from your home city to each internship city using coordinates (Haversine formula).

-  **Safety-First Scoring**  
  Incorporates crime data, rent data & distance — all weighted by your custom preferences.

-  **Visual Insights**  
  Generates radar charts and comparison tables to help visualize safety, cost & distance.

- 📝 **Smart Summary & Justification**  
  Shows a chat-style summary explaining why a city was recommended.

- 💡 **Safety Tips Section**  
  Recommends real-life safety tips post prediction for added awareness.

---

## 📂 Project Structure
SheMaps/
├── All_cities_df.csv # Latitude and longitude data
├── merged_dataset.csv # Final merged dataset (crime + rent + coords)
├── shemaps_app.py # Streamlit app code
├── city_recommender_model.pkl # Trained ML model (Decision Tree)
├── sheMaps_logo.png # Logo for the app
└── README.md # This file

## Tech stacks

- **Frontend/UI:** Streamlit  
- **Backend/Logic:** Python, Pandas, NumPy  
- **ML Model:** Decision Tree Classifier (via scikit-learn)  
- **Data Visualization:** Plotly  
- **Deployment Ready:** Streamlit or local hosting


##  Feedback
Have ideas to improve SheMaps?
Open an issue or drop us a message — we’d love to hear from you!

  

