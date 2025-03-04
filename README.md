# **FoundYOU: A Foundation Shade Recommendation System Based on Skin Tone Identification**

### **Overview**
FoundYOU is a web app that helps users find the perfect foundation shade based on their skin tone. Using machine learning, the system analyzes uploaded face images and recommends the most suitable foundation. This project aims to enhance the online shopping experience and reduce the risk of purchasing mismatched products.

### **How It Works**
1. Upload or Capture a clear photo of your face.
2. The system performs face detection using Cascade Classifier.
3. Skin tone classification is done using Gaussian Mixture Model and Normalization.
4. The system identifies the dominant hex color of the skin.
5. K-Means Clustering is used to find the closest matching foundation shade.
6. The best foundation recommendations are displayed, including brand, description, hex color, and product URL.

### **Features**
- Automatic Skin Tone Detection
- Personalized Foundation Shade Recommendation
- Easy-to-Use Web App Interface
- Machine Learning-Based Prediction

### **Dataset**
The model is trained using the Cosmetic Foundation Shades dataset from Kaggle, which contains 6,816 foundation products from online cosmetic stores like Ulta and Sephora.
