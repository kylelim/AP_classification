
# How AutoPrompt works

This document outlines the optimization process flows of AutoPrompt. The framework is designed with modularity and adaptability in mind, allowing for easy extension of the prompt calibration process from classification tasks to generative tasks. 


##   Classification Pipeline Overview 

The classification pipeline executes a calibration process involving the following steps:

1. **User Input:**
   - The user provides an initial prompt and task description to kickstart the calibration process.

2. **Prediction:**
   - The annotated samples are evaluated using the current prompt to assess model performance.

3. **Prompt Analysis:**
   - The pipeline analyzes the prompt scores and identifies instances of large errors.

4. **Prompt Refinement:**
   - A new prompt is suggested based on the evaluation results, aiming to improve model accuracy.

5. **Iteration:**
   - Steps 2-4 are iteratively repeated until convergence, refining the prompt and enhancing the model's performance throughout the process. 
