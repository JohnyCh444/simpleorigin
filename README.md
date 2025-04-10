# Scientific Data Visualization App

A comprehensive tool for scientific data visualization, curve fitting, and code generation.

![App Screenshot](https://github.com/yourusername/scientific-data-visualization/raw/main/screenshot.png)

## Features

- **Interactive Data Plotting**: Upload and visualize scientific data with customizable plots
- **Curve Fitting**: Apply various curve fitting methods:
  - Linear regression
  - Polynomial regression (customizable degree)
  - Logarithmic regression
  - Exponential regression
- **Error Analysis**: Add error bars to your data points with full customization
- **Statistical Analysis**: Compute R² values and parameter uncertainties
- **Function Drawing**: Draw mathematical functions with adjustable parameters:
  - Linear functions
  - Quadratic functions
  - Polynomial functions
  - Trigonometric functions
  - Custom expressions
- **Derivative Calculation**: Visualize derivatives of fitted curves and functions
- **Code Generation**: Export ready-to-use Python code for reproducible analysis

## Installation

1. Clone this repository or download the ZIP file
2. Navigate to the project directory
3. Install the required dependencies:

```bash
pip install streamlit numpy pandas matplotlib scipy
```

4. Run the application:

```bash
streamlit run app.py
```

## Usage

### Data Entry & Plotting

1. Enter your X and Y values in the text areas
2. Customize plot parameters (colors, labels, title, etc.)
3. Add error bars if needed
4. Apply curve fitting with the desired method

### Function Drawing

1. Select a function type from the dropdown
2. Adjust parameters to customize the function
3. Add the function to the main plot

### Code Generation

1. After creating your visualization, navigate to the "Code Generation" tab
2. Review the auto-generated Python code
3. Download the code for use in your own projects

## Dependencies

- Python 3.7+
- Streamlit
- NumPy
- Pandas
- Matplotlib
- SciPy

## License

MIT License

## Author

Your Name

## Acknowledgements

This tool was created to help scientists and researchers easily create publication-quality visualizations and perform data analysis without extensive programming knowledge.