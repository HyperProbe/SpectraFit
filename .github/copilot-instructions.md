<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# HSI-Biopsy Project Instructions

This is a scientific/research project focused on hyperspectral imaging (HSI) analysis of biopsy samples.

## Project Context
- This project uses Python for data processing and analysis of hyperspectral imaging data
- The codebase likely includes functions for image processing, feature extraction, and machine learning
- The project may include visualization tools for medical imaging data
- Consider best practices for medical/scientific data processing when suggesting code

## Coding Guidelines
- Follow PEP 8 style guidelines for Python code
- Use NumPy docstring format for function and class documentation
- Write unit tests for core functionality only if prompted explicitly
- Prefer established scientific libraries (NumPy, SciPy, scikit-learn, scikit-image, etc.) when possible

## Data Handling
- Assume HSI data is stored in standard formats (HDF5, NumPy arrays, etc.)
- Ensure data processing maintains provenance and traceability
- Follow best practices for medical data privacy

## System Prompts
- You are an agent - please keep going until the user’s query is completely resolved, before ending your turn and yielding back to the user. Only terminate your turn when you are sure that the problem is solved.
- If you are not sure about file content or codebase structure pertaining to the user’s request, use your tools to read files and gather the relevant information: do NOT guess or make up an answer.
- You MUST plan extensively before each function call, and reflect extensively on the outcomes of the previous function calls. DO NOT do this entire process by making function calls only, as this can impair your ability to solve the problem and think insightfully.
- First, think carefully step by step about what documents are needed to answer the query.