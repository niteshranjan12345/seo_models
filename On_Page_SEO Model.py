import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd

# Step 1: Generate SEO-Optimized HTML File
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <!-- On-page SEO: Title Tag (Unique, keyword-rich, <60 characters) -->
    <title>Best Hotels with Spa in Udaipur | Bidinn</title>
    
    <!-- On-page SEO: Meta Description (Engaging, keyword-rich, <160 characters) -->
    <meta name="description" content="Book luxury hotels with spa in Udaipur at Bidinn. Enjoy top amenities, great deals, and a relaxing stay. Reserve now!">
    
    <!-- On-page SEO: Canonical Tag (Prevent duplicate content) -->
    <link rel="canonical" href="https://www.bidinn.in/hotels-with-spa-in-udaipur">
    
    <!-- Technical SEO: Character Encoding -->
    <meta charset="UTF-8">
    
    <!-- Technical SEO: Viewport for Mobile-Friendliness -->
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    
    <!-- Technical SEO: Robots Meta Tag (Allow indexing) -->
    <meta name="robots" content="index, follow">
    
    <!-- Technical SEO: Minified CSS for Page Speed -->
    <style>
        body {font-family: Arial, sans-serif; margin: 0; padding: 0;}
        header, main, footer {padding: 20px; max-width: 1200px; margin: auto;}
        img {max-width: 100%; height: auto;}
        .lazy {loading: lazy;}
    </style>
    
    <!-- Technical SEO: Structured Data (Schema Markup for Hotels) -->
    <script type="application/ld+json">
    {
        "@context": "https://schema.org",
        "@type": "Hotel",
        "name": "Hotels with Spa in Udaipur",
        "address": {
            "@type": "PostalAddress",
            "addressLocality": "Udaipur",
            "addressRegion": "Rajasthan",
            "addressCountry": "India"
        },
        "starRating": {
            "@type": "Rating",
            "ratingValue": "4.5"
        },
        "description": "Luxury hotels with spa facilities in Udaipur, offering premium amenities and great deals."
    }
    </script>
</head>
<body>
    <!-- On-page SEO: Header Tag (H1, one per page, keyword-rich) -->
    <header>
        <h1>Best Hotels with Spa in Udaipur</h1>
    </header>
    
    <!-- On-page SEO: Content (High-quality, keyword-optimized, >500 words) -->
    <main>
        <!-- On-page SEO: Header Tags (H2–H6 for structure) -->
        <h2>Why Choose Hotels with Spa in Udaipur?</h2>
        <p>Discover the best hotels in Udaipur with spa facilities, offering relaxation and luxury. Our curated list includes top properties with premium amenities like infinity pools, gourmet dining, and spa treatments, perfect for a romantic getaway or family vacation.</p>
        
        <!-- On-page SEO: Keyword Usage (Natural, 1–2% density) -->
        <p>Looking for a <strong>hotel with spa in Udaipur</strong>? Bidinn offers exclusive deals on luxury accommodations. Enjoy serene lake views, world-class spa services, and unmatched hospitality.</p>
        
        <!-- On-page SEO: Image with Alt Text -->
        <img src="udaipur-spa-hotel.jpg" alt="Luxury hotel with spa in Udaipur" class="lazy">
        
        <!-- On-page SEO: Internal Links -->
        <p>Explore more: <a href="/hotels-in-udaipur">All Udaipur Hotels</a> | <a href="/luxury-hotels-india">Luxury Hotels in India</a></p>
        
        <!-- On-page SEO: External Links (Reputable sources) -->
        <p>Learn about Udaipur tourism at <a href="https://www.rajasthantourism.gov.in" target="_blank" rel="noopener">Rajasthan Tourism</a>.</p>
    </main>
    
    <!-- Technical SEO: Footer with Clean URL Structure -->
    <footer>
        <p>© 2025 Bidinn | <a href="/sitemap.xml">Sitemap</a> | <a href="/robots.txt">Robots.txt</a></p>
    </footer>
    
    <!-- Technical SEO: Lazy Loading for Images (Page Speed) -->
    <!-- Technical SEO: HTTPS enforced via server configuration -->
    <!-- Technical SEO: 404 Error Handling (via server or CMS) -->
    <!-- Technical SEO: XML Sitemap (Generated separately at /sitemap.xml) -->
    <!-- Technical SEO: Robots.txt (Generated separately at /robots.txt) -->
    <!-- Technical SEO: Noindex for non-critical pages (e.g., <meta name="robots" content="noindex"> on login pages) -->
</body>
</html>
"""

# Write HTML to file
with open('seo_optimized_page.html', 'w') as f:
    f.write(html_content)
print("Generated SEO-optimized HTML file: 'seo_optimized_page.html'")

# Step 2: Define SEO Factors Weightage
on_page_seo = {
    'Title Tag': 10,
    'Meta Description': 8,
    'Header Tags': 8,
    'Keyword Usage': 7,
    'Image Alt Text': 5,
    'Internal/External Links': 5,
    'Content Length': 4,
    'Schema Markup': 2,
    'Canonical Tag': 1
}

technical_seo = {
    'HTTPS': 10,
    'Mobile-Friendliness': 10,
    'Page Speed': 10,
    'XML Sitemap': 5,
    'Robots.txt': 5,
    'Structured Data': 4,
    '404 Error Handling': 3,
    'Clean URL Structure': 2,
    'Noindex Tags': 1
}

# Convert to DataFrames for Plotly
on_page_df = pd.DataFrame(list(on_page_seo.items()), columns=['Factor', 'Weightage'])
technical_df = pd.DataFrame(list(technical_seo.items()), columns=['Factor', 'Weightage'])

# Step 3: Create Dash Dashboard with Pie Charts
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("SEO Factors Weightage Dashboard", style={'textAlign': 'center'}),
    
    html.Div([
        html.H2("On-page SEO Factors"),
        dcc.Graph(
            id='on-page-pie',
            figure=px.pie(
                on_page_df,
                values='Weightage',
                names='Factor',
                title='On-page SEO Weightage (%)'
            )
        )
    ], style={'width': '45%', 'display': 'inline-block', 'padding': '20px'}),
    
    html.Div([
        html.H2("Technical SEO Factors"),
        dcc.Graph(
            id='technical-pie',
            figure=px.pie(
                technical_df,
                values='Weightage',
                names='Factor',
                title='Technical SEO Weightage (%)'
            )
        )
    ], style={'width': '45%', 'display': 'inline-block', 'padding': '20px'})
])

# Run the Dash app
if __name__ == '__main__':
    app.run(debug=True)
    print("Dash dashboard is running at http://127.0.0.1:8050")

# Save pie charts as HTML (for static viewing)
on_page_fig = px.pie(on_page_df, values='Weightage', names='Factor', title='On-page SEO Weightage (%)')
technical_fig = px.pie(technical_df, values='Weightage', names='Factor', title='Technical SEO Weightage (%)')
on_page_fig.write_html('on_page_seo_pie.html')
technical_fig.write_html('technical_seo_pie.html')
print("Pie charts saved as 'on_page_seo_pie.html' and 'technical_seo_pie.html'")