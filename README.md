# **Understanding Disagreement Between Humans and Machines in XAI: Robustness, Fidelity, and Region-Based Explanations in Automatic Neonatal Pain Assessment**

*This repository contains all main code needed to reproduce our research.*

## **Abstract**

Artificial Intelligence (AI) offers a promising approach to automating neonatal pain assessment, improving consistency and objectivity in clinical decision-making. However, differences between how humans and AI models perceive and explain pain-related features present challenges for adoption. This study systematically examines four key aspects of disagreement in neonatal pain classification: (1) the interpretability and stability of region-based versus pixel-based explanations, (2) differences in agreement between ''pain'' and ''no-pain'' classifications, (3) the influence of robustness on explainer agreement, and (4) the impact of faithfulness (measured by infidelity) on agreement between explainers.

Our findings show that region-based explanations are more intuitive and stable than pixel-based methods, enhancing interpretability and clarity in understanding disagreement. Humans primarily focus on central facial features, such as the nose, mouth, and the area between the eyebrows, whereas AI-generated explanations depend on the model and explainer. Agreement is higher in ''pain'' classifications than in ''no-pain'' cases, suggesting that pain-related features are more distinct and consistently identified by both humans and AI. Additionally, more robust explanations exhibit greater consistency across explainers, while higher faithfulness (lower infidelity) tends to reduce pixel-level agreement, as different explainers capture distinct but equally valid aspects of the model’s reasoning. These findings underscore the limitations of using high agreement as a sole indicator of explanation quality, emphasizing the need for multi-faceted XAI evaluation. By introducing region-level agreement metrics and assessing robustness and faithfulness, we offer a framework to improve trust and transparency in AI-driven neonatal pain assessment.

# Requirements

Coded and tested on Python 3.10 with torch==1.13.1.

# How to Use It

- ``GradCAM.py`` and ``IntegratedGradients.py`` were implemented for PyTorch models.  

- ``agreement.py`` contains all the necessary code to calculate pixel-level and region-level agreement. It runs exclusively with NumPy. 
 
- ``sensitivity.py`` and ``infidelity.py`` can be easily adapted to any XAI explainer output and primarily run with NumPy.  





# Citation

``` latex
\cite{

}
```