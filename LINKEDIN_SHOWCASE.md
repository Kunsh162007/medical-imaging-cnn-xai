# 🏷️ LinkedIn Showcase

> Copy-paste this entire document into your LinkedIn post or article. You can upload the referenced images directly from your `results/` folder into the post in the order specified.

---

## 🧠 Alzheimer's Disease MRI Classification: A Deep Learning Journey

I'm incredibly proud to share the culmination of a massive 5-week deep learning project I've been working on, focused on a critical problem in the medical field: **Classifying Alzheimer's Disease severity from MRI brain scans using Deep Learning.** 

### 🚨 The Problem
Alzheimer's disease is a progressive neurologic disorder that causes the brain to shrink and brain cells to die. Early diagnosis is crucial, but analyzing MRI scans for subtle structural changes is incredibly complex and time-consuming for radiologists. My goal was to build an automated, highly accurate tool to classify MRI scans into four stages:
1. Non-Demented  
2. Very Mild Demented  
3. Mild Demented  
4. Moderate Demented  

### 💡 The Solution & Pipeline
I built a complete, production-grade Deep Learning pipeline from scratch, covering everything from fundamental perceptrons to state-of-the-art CNNs and Explainable AI (XAI). The entire project is **framework-agnostic**, with parallel implementations in both **TensorFlow** and **PyTorch**.

**Here's what I built over 5 modules:**
📐 **Module 1 & 2:** Hand-coded basic MLPs, Backpropagation, and regularization to fully understand the math behind the magic.  
🖼️ **Module 3:** Implemented CNN fundamentals and custom data loaders with advanced image augmentation for medical data.  
🏗️ **Module 4:** Built and trained state-of-the-art architectures: **AlexNet, VGG16, GoogLeNet/Inception, and ResNet50**, culminating in a powerful **Ensemble model**.  
🔬 **Module 5:** Integrated a massive Explainable AI (XAI) suite because in medical tech, *trust is everything*. I implemented Grad-CAM, SHAP, LIME, Guided Backprop, and Occlusion Sensitivity.

### 🏆 The Results
The Ensemble Model achieved an outstanding **95.4% Test Accuracy (0.953 F1, 0.993 AUC)**, significantly outperforming individual architectures while maintaining high precision across all four dementia classes.

---

## 📸 Visual Showcase (Upload these images in order)

**(Image 1: Model Accuracy Comparison)**
Path: `results/04_advanced_architectures/ensemble_model_comparison.png`
*Description:* A clear look at how the Custom CNN, ResNet50, and the final Ensemble model compare in accuracy and loss. The Ensemble model clearly pushes the boundaries of performance.

**(Image 2: Ensemble Confusion Matrix)**
Path: `results/04_advanced_architectures/ensemble_pt_cm.png`
*Description:* The confusion matrix visualization showing high predictive accuracy across all four classes, minimizing false negatives for severe stages.

**(Image 3: Grad-CAM Explainability)**
Path: `results/05_explainability_xai/gradcam/gradcam_resnet50_pt.png`
*Description:* Gradient-weighted Class Activation Mapping (Grad-CAM). This medical-grade heatmap shows *exactly* which regions of the brain the ResNet50 model is looking at (like ventricular enlargement) to make its diagnosis.

**(Image 4: Guided Backpropagation)**
Path: `results/05_explainability_xai/guided_backprop/guided_bp_resnet50_pt.png`
*Description:* Guided Backpropagation revealing the fine-grained, high-resolution structural edges the model identifies as critical features.

**(Image 5: LIME / SHAP Explanations)**
Paths: `results/05_explainability_xai/lime/` or `results/05_explainability_xai/shap/` (Pick your favorite plot here)
*Description:* Perturbation-based and game-theoretic approaches proving the model isn't just "guessing" but using sound structural indicators.

---

I'd love to hear feedback from engineers and researchers working in MedTech or Computer Vision! 

🔗 **Full Code & Architecture on GitHub:** https://github.com/Kunsh162007/medical-imaging-cnn-xai

#DeepLearning #MachineLearning #ComputerVision #MedicalImaging #TensorFlow #PyTorch #XAI #DataScience #AI #Alzheimers #NeuralNetworks
