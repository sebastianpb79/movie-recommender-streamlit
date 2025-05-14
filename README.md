# 🎬 Movie Recommender Streamlit App

A lightweight Streamlit app demonstrating a hybrid movie recommendation system combining content‑based, collaborative, and hybrid filtering techniques.

---

## 📖 Overview

This project showcases:

- **Content‑Based Filtering** using TF‑IDF on movie genres
- **Collaborative Filtering** (User‑ and Item‑based) using the Surprise library
- **Hybrid Model** averaging SVD and KNN predictions
- Interactive Streamlit interface with data exploration, visualizations, model details, and live recommendations

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Git
- (Optional) [Anaconda](https://www.anaconda.com/) to manage environments

### Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/sebastianpb79/movie-recommender-streamlit.git
   cd movie-recommender-streamlit
   ```

movie-recommender-streamlit/
│
├── data/
│ ├── movies.csv
│ ├── ratings.csv
│ ├── _.pkl ← precomputed model + similarity matrices
│ └── _.png ← static visualizations used in the app
│
├── app.py ← main Streamlit application
├── requirements.txt ← Python packages
├── .gitignore
└── README.md
