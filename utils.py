import pandas as pd
import numpy as np
import io
import base64
import plotly.io as pio
from datetime import datetime, timedelta

def load_example_data(dataset_type):
    """
    Load example datasets for demonstration
    
    Parameters:
    -----------
    dataset_type : str
        Type of dataset to load
    
    Returns:
    --------
    pd.DataFrame
        Example dataset
    """
    if dataset_type == "Time Series":
        # Generate time series data
        dates = pd.date_range(start='2023-01-01', periods=100)
        np.random.seed(42)
        df = pd.DataFrame({
            'Date': dates,
            'Sales': np.random.normal(loc=100, scale=20, size=100).cumsum(),
            'Revenue': np.random.normal(loc=200, scale=30, size=100).cumsum(),
            'Expenses': np.random.normal(loc=50, scale=10, size=100).cumsum()
        })
        return df
    
    elif dataset_type == "Categorical Data":
        # Generate categorical data
        categories = ['Category A', 'Category B', 'Category C', 'Category D', 'Category E']
        df = pd.DataFrame({
            'Category': categories,
            'Value 1': np.random.randint(10, 100, size=5),
            'Value 2': np.random.randint(20, 80, size=5),
            'Value 3': np.random.randint(30, 90, size=5)
        })
        return df
    
    elif dataset_type == "Scatter Plot Data":
        # Generate scatter plot data
        np.random.seed(42)
        n = 100
        x = np.random.normal(loc=50, scale=15, size=n)
        y = x + np.random.normal(loc=0, scale=10, size=n)
        
        # Add some categorical variables
        categories = np.random.choice(['Group A', 'Group B', 'Group C'], size=n)
        sizes = np.random.randint(10, 100, size=n)
        
        df = pd.DataFrame({
            'X_Value': x,
            'Y_Value': y,
            'Category': categories,
            'Size': sizes
        })
        return df
    
    else:
        # Default simple dataset
        return pd.DataFrame({
            'x': list(range(10)),
            'y': np.random.randint(0, 100, size=10)
        })

def download_plot(fig, format_type, title):
    """
    Create a download link for the plot
    
    Parameters:
    -----------
    fig : plotly.graph_objs._figure.Figure
        The plotly figure to export
    format_type : str
        File format (png, jpg, svg, html, pdf)
    title : str
        Plot title used for filename
    
    Returns:
    --------
    str
        HTML download link
    """
    # Sanitize filename
    filename = title.replace(' ', '_').lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename}_{timestamp}.{format_type}"
    
    # Check the format and generate appropriate file
    if format_type == 'html':
        buffer = io.StringIO()
        pio.write_html(fig, file=buffer)
        html_bytes = buffer.getvalue().encode()
        encoded = base64.b64encode(html_bytes).decode()
        mime_type = 'text/html'
    else:
        buffer = io.BytesIO()
        if format_type == 'png':
            pio.write_image(fig, buffer, format='png')
            mime_type = 'image/png'
        elif format_type == 'jpg' or format_type == 'jpeg':
            pio.write_image(fig, buffer, format='jpg')
            mime_type = 'image/jpeg'
        elif format_type == 'svg':
            pio.write_image(fig, buffer, format='svg')
            mime_type = 'image/svg+xml'
        elif format_type == 'pdf':
            pio.write_image(fig, buffer, format='pdf')
            mime_type = 'application/pdf'
        else:
            # Default to PNG
            pio.write_image(fig, buffer, format='png')
            mime_type = 'image/png'
        
        encoded = base64.b64encode(buffer.getvalue()).decode()
    
    # Create download link
    href = f'<a href="data:{mime_type};base64,{encoded}" download="{filename}">Download {format_type.upper()} file</a>'
    return href
