# 🏷️ LinkedIn Showcase

> Copy-paste this entire document into your LinkedIn post or article. Upload the referenced images directly from your `results/` folder into the post in the exact order specified below to create a comprehensive project gallery.

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
I built a complete, production-grade Deep Learning pipeline from scratch, covering everything from fundamental perceptrons to state-of-the-art CNNs and Explainable AI (XAI). The entire project is **framework-agnostic** (it fully supports both TensorFlow and PyTorch), but for this showcase, I executed the entire training and inference pipeline using **PyTorch**.

**Here's what I built over 5 modules:**
📐 **Module 1 & 2:** Hand-coded basic MLPs, Backpropagation, and regularization to fully understand the math behind the magic.  
🖼️ **Module 3:** Implemented CNN fundamentals and custom data loaders with advanced image augmentation for medical data.  
🏗️ **Module 4:** Built and trained state-of-the-art architectures: **AlexNet, VGG16, GoogLeNet/Inception, and ResNet50**, culminating in a powerful **Ensemble model**.  
🔬 **Module 5:** Integrated a massive Explainable AI (XAI) suite because in medical tech, *trust is everything*. I implemented Grad-CAM, SHAP, LIME, Guided Backprop, Score-CAM, and Occlusion Sensitivity.

### 🏆 The Results
The Ensemble Model achieved an outstanding **95.4% Test Accuracy**, significantly outperforming individual architectures while maintaining high precision across all four dementia classes.

---

## 📸 Comprehensive Visual Showcase (Upload these images in order)

*(Tip for LinkedIn: Upload these images in the exact order below. LinkedIn will create a nice clean gallery swipe for viewers.)*

### Part 1: Architecture Performance
**(Image 1: Overall Model Comparison)**
Path: `results/04_advanced_architectures/ensemble_model_comparison.png`
*Description:* A clear look at how all architectures (Custom CNN, AlexNet, VGGNet, GoogLeNet, ResNet50, and Ensemble) compare in accuracy and loss. 

**(Image 2: AlexNet Performance)**
Path: `results/04_advanced_architectures/alexnet_pt_cm.png`
*Description:* Confusion matrix detailing how the baseline AlexNet architecture performed on the 4-class dataset.

**(Image 3: VGGNet Performance)**
Path: `results/04_advanced_architectures/vggnet_pt_cm.png`
*Description:* Performance evaluation using the much deeper VGG16 architecture.

**(Image 4: GoogLeNet Performance)**
Path: `results/04_advanced_architectures/googlenet_pt_cm.png`
*Description:* Results from the Inception-based GoogLeNet model, showing improved feature extraction.

**(Image 5: ResNet50 Performance)**
Path: `results/04_advanced_architectures/resnet50_pt_cm.png`
*Description:* Highly accurate ResNet50 classification results leveraging residual connections to avoid vanishing gradients.

**(Image 6: The Ultimate Ensemble Matrix)**
Path: `results/04_advanced_architectures/ensemble_pt_cm.png`
*Description:* The final Ensemble confusion matrix. By combining all models, we minimized false negatives across the critical dementia stages to achieve >95% accuracy.

### Part 2: Opening the Black Box (Explainable AI)
**(Image 7: Grad-CAM Explainability)**
Path: `results/05_explainability_xai/gradcam/gradcam_resnet50_pt.png`
*Description:* Gradient-weighted Class Activation Mapping (Grad-CAM). This heatmap shows *exactly* which regions of the brain the ResNet50 model is looking at (like ventricular enlargement) to make its diagnosis.

**(Image 8: Guided Backpropagation)**
Path: `results/05_explainability_xai/guided_backprop/guided_bp_resnet50_pt.png`
*Description:* Guided Backpropagation revealing the fine-grained, high-resolution structural edges the model identifies as critical features.

**(Image 9: LIME Superpixel Explanations)**
Path: `results/05_explainability_xai/lime/lime_resnet50_pt.png`
*Description:* Local Interpretable Model-agnostic Explanations (LIME). This perturbation-based approach highlights the specific "superpixels" that contributed most heavily to the Alzheimer's prediction.

**(Image 10: SHAP Feature Attributions)**
Path: `results/05_explainability_xai/shap/shap_resnet50_pt.png`
*Description:* Game-theoretic SHAP (SHapley Additive exPlanations) values proving the model isn't just "guessing" but using sound structural indicators across the scan.

**(Image 11: Occlusion Sensitivity Analysis)**
Path: `results/05_explainability_xai/occlusion/occlusion_resnet50_pt.png`
*Description:* Analyzing how the model's confidence drops when critical areas of the MRI scan are artificially blocked out.

---

I'd love to hear feedback from engineers and researchers working in MedTech or Computer Vision! 

🔗 **Full Code & Architecture on GitHub:** https://github.com/Kunsh162007/medical-imaging-cnn-xai

#DeepLearning #MachineLearning #ComputerVision #MedicalImaging #PyTorch #TensorFlow #XAI #DataScience #AI #Alzheimers #NeuralNetworks
