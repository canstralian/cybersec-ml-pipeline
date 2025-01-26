# ML Pipeline for Cybersecurity Purple Teaming 🛡️

![Build Status](https://img.shields.io/github/actions/workflow/status/canstralian/cybersec-ml-pipeline/ci.yml)
![License](https://img.shields.io/github/license/canstralian/cybersec-ml-pipeline)
![Coverage](https://img.shields.io/codecov/c/github/canstralian/cybersec-ml-pipeline)
![Contributors](https://img.shields.io/github/contributors/canstralian/cybersec-ml-pipeline)
![GitHub issues](https://img.shields.io/github/issues/canstralian/cybersec-ml-pipeline)

A scalable Streamlit-based machine learning pipeline platform specialized for cybersecurity purple-teaming, enabling advanced data processing and model training.

## Features 🚀

- **Distributed Data Processing**: Leverage Dask for handling large-scale datasets
- **Interactive ML Pipeline**: Build and customize machine learning workflows
- **Real-time Visualization**: Monitor model performance and data insights
- **Cybersecurity Focus**: Tailored for purple team operations and security analytics

## Tech Stack 💻

- **Dask**: Distributed data processing
- **Scikit-learn**: ML model training and evaluation
- **Streamlit**: Interactive web interface
- **Pandas/NumPy**: Data manipulation and analysis
- **Matplotlib/Seaborn**: Data visualization

## Getting Started 🏁

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/cybersec-ml-pipeline.git
cd cybersec-ml-pipeline
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run app.py
```

## Usage Guide 📖

1. **Data Upload**
   - Support for CSV and JSON formats
   - Automatic handling of large datasets using Dask

2. **Pipeline Configuration**
   - Choose preprocessing steps
   - Configure model parameters
   - Select features for training

3. **Model Training**
   - Interactive parameter tuning
   - Real-time performance metrics
   - Visual model evaluation

## GitHub Actions for Hugging Face Hub

To set up GitHub Actions for pushing to Hugging Face Hub, follow these steps:

1. **Create a GitHub Actions workflow file**: The workflow file should be located at `.github/workflows/hf-push.yml`.

2. **Trigger the workflow**: The workflow should be triggered on a push to the `main` branch.

3. **Set up Python environment**: Ensure the Python version is set to 3.11.

4. **Install dependencies**: Install the necessary dependencies including `requests`, `pandas`, `numpy`, `plotly`, `scikit-learn`, `statsmodels`, `streamlit`, `nltk`, and `huggingface_hub`.

5. **Retrieve Hugging Face token**: The Hugging Face token (`HF_TOKEN`) should be retrieved from the GitHub secrets and set as an environment variable.

6. **Push to Hugging Face Hub**: Use the `huggingface_hub` library to push the repository contents to the Hugging Face Hub.

Make sure you have the `HF_TOKEN` secret set up in your GitHub repository settings to authenticate with Hugging Face Hub.

## Contributing 🤝

Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## Security 🔒

For security concerns, please review our [Security Policy](.github/SECURITY.md).

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments 👏

- Streamlit community for the amazing framework
- Scikit-learn team for the ML tools
- All contributors who help improve this project
