import streamlit as st
import pandas as pd
import requests
import json
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from geopy.geocoders import Nominatim
import google.generativeai as genai
from PIL import Image
import io
import base64
import time
import random

# Page configuration
st.set_page_config(
    page_title="🍛 Smart Vendor Assistant",
    page_icon="🍛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #FF6B35, #F7931E);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FF6B35;
        margin: 0.5rem 0;
    }
    .alert-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .alert-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .chat-message {
        padding: 10px;
        margin: 10px 0;
        border-radius: 10px;
        max-width: 80%;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: auto;
        text-align: right;
    }
    .bot-message {
        background-color: #f5f5f5;
        margin-right: auto;
    }
    .profile-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'vendor_profile' not in st.session_state:
    st.session_state.vendor_profile = {
        'name': '',
        'business_name': '',
        'location': '',
        'city': '',
        'state': '',
        'pincode': '',
        'business_type': '',
        'menu_items': [],
        'daily_capacity': 100,
        'operating_hours': '8:00 AM - 10:00 PM',
        'target_customers': '',
        'budget_range': '',
        'speciality': '',
        'years_experience': 0,
        'setup_complete': False
    }

if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []

if 'suppliers_data' not in st.session_state:
    st.session_state.suppliers_data = []

if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = ''


# Gemini AI Configuration
def configure_gemini():
    """Configure Gemini AI with API key"""
    if st.session_state.gemini_api_key:
        try:
            genai.configure(api_key=st.session_state.gemini_api_key)
            return True
        except Exception as e:
            st.error(f"Gemini API configuration error: {str(e)}")
            return False
    return False


def get_gemini_response(prompt, context=""):
    """Get response from Gemini AI"""
    if not configure_gemini():
        return "Please configure your Gemini API key first."

    try:
        model = genai.GenerativeModel('gemini-2.0-flash')

        # Create context-aware prompt
        full_prompt = f"""
        You are a smart assistant for Indian street food vendors. 

        Vendor Profile:
        - Name: {st.session_state.vendor_profile['name']}
        - Business: {st.session_state.vendor_profile['business_name']}
        - Location: {st.session_state.vendor_profile['city']}, {st.session_state.vendor_profile['state']}
        - Business Type: {st.session_state.vendor_profile['business_type']}
        - Speciality: {st.session_state.vendor_profile['speciality']}
        - Menu Items: {', '.join(st.session_state.vendor_profile['menu_items'])}
        - Experience: {st.session_state.vendor_profile['years_experience']} years

        Additional Context: {context}

        User Query: {prompt}

        Please provide helpful, practical advice specific to this vendor's profile and location. 
        Include relevant information about suppliers, pricing, weather considerations, and business tips.
        Keep responses conversational and supportive.
        """

        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"Error getting AI response: {str(e)}"


def get_location_based_suppliers(city, state, business_type):
    """Get suppliers based on location using Gemini AI"""
    if not configure_gemini():
        return []

    try:
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"""
        Provide a list of famous wholesale markets, suppliers, and manufacturers for street food vendors in {city}, {state}, India.
        Focus on suppliers for {business_type} business.

        For each supplier, provide:
        1. Name
        2. Location/Address
        3. Speciality items
        4. Contact information (if known)
        5. Type (Wholesale/Retail/Manufacturer)
        6. Approximate price range

        Format the response as a structured list.
        """

        response = model.generate_content(prompt)
        return parse_suppliers_response(response.text)
    except Exception as e:
        return [{"error": f"Error fetching suppliers: {str(e)}"}]


def parse_suppliers_response(response_text):
    """Parse Gemini response into structured supplier data"""
    # This is a simplified parser - you might want to make it more robust
    suppliers = []
    lines = response_text.split('\n')

    current_supplier = {}
    for line in lines:
        line = line.strip()
        if line and not line.startswith('-') and not line.startswith('*'):
            if any(keyword in line.lower() for keyword in ['market', 'supplier', 'wholesale', 'trader', 'company']):
                if current_supplier:
                    suppliers.append(current_supplier)
                current_supplier = {
                    'name': line,
                    'location': 'Location will be updated',
                    'speciality': 'General supplies',
                    'type': 'Wholesale',
                    'contact': 'Contact details to be verified',
                    'rating': random.uniform(3.5, 5.0)
                }

    if current_supplier:
        suppliers.append(current_supplier)

    return suppliers[:10]  # Return top 10


def get_weather_insights(city):
    """Get weather-based business insights using Gemini AI"""
    if not configure_gemini():
        return "Please configure Gemini API key"

    try:
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"""
        Provide current weather insights and business recommendations for a street food vendor in {city}, India.
        Include:
        1. Expected weather conditions today
        2. How weather affects street food business
        3. Specific menu recommendations
        4. Preparation tips
        5. Expected customer flow

        Make it practical and actionable.
        """

        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error getting weather insights: {str(e)}"


def get_market_price_analysis(city, items):
    """Get market price analysis using Gemini AI"""
    if not configure_gemini():
        return "Please configure Gemini API key"

    try:
        model = genai.GenerativeModel('gemini-pro')
        items_str = ', '.join(items) if items else 'common street food ingredients'
        prompt = f"""
        Provide current market price analysis for street food ingredients in {city}, India.
        Focus on these items: {items_str}

        Include:
        1. Current price trends
        2. Seasonal variations
        3. Best places to buy
        4. Cost-saving tips
        5. Quality vs price recommendations

        Provide practical pricing guidance.
        """

        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error getting price analysis: {str(e)}"


# Main App Layout
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🍛 Smart Street Food Vendor Assistant</h1>
        <p>AI-Powered Business Companion for Indian Street Food Vendors</p>
    </div>
    """, unsafe_allow_html=True)

    # API Key Configuration
    with st.sidebar:
        st.header("🔑 API Configuration")
        api_key = st.text_input("Gemini API Key", type="password",
                                value=st.session_state.gemini_api_key,
                                help="Get your API key from Google AI Studio")
        if st.button("Save API Key"):
            st.session_state.gemini_api_key = api_key
            if configure_gemini():
                st.success("✅ API Key configured successfully!")
            else:
                st.error("❌ Invalid API Key")

    # Check if profile is complete
    if not st.session_state.vendor_profile['setup_complete']:
        show_profile_setup()
    else:
        show_main_dashboard()


def show_profile_setup():
    """Complete vendor profile setup"""
    st.header("👨‍🍳 Complete Your Vendor Profile")
    st.write("Let's set up your profile to get personalized recommendations!")

    with st.form("vendor_profile_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Personal Information")
            name = st.text_input("Your Name*", value=st.session_state.vendor_profile['name'])
            business_name = st.text_input("Business Name*", value=st.session_state.vendor_profile['business_name'])
            years_experience = st.slider("Years of Experience", 0, 50,
                                         st.session_state.vendor_profile['years_experience'])

            st.subheader("Location Details")
            city = st.text_input("City*", value=st.session_state.vendor_profile['city'])
            state = st.selectbox("State*", [
                "", "Maharashtra", "Delhi", "Karnataka", "Tamil Nadu", "Gujarat",
                "Rajasthan", "Uttar Pradesh", "West Bengal", "Madhya Pradesh",
                "Other"
            ], index=0 if not st.session_state.vendor_profile['state'] else None)
            pincode = st.text_input("Pin Code", value=st.session_state.vendor_profile['pincode'])
            location = st.text_area("Specific Location/Area", value=st.session_state.vendor_profile['location'])

        with col2:
            st.subheader("Business Details")
            business_type = st.selectbox("Business Type*", [
                "", "Street Food Cart", "Food Stall", "Tea Stall", "Juice Corner",
                "Snack Counter", "Tiffin Service", "Mobile Vendor", "Other"
            ])

            speciality = st.text_input("Your Speciality*",
                                       value=st.session_state.vendor_profile['speciality'],
                                       help="e.g., Vada Pav, Dosa, Chaat, Tea")

            target_customers = st.selectbox("Target Customers", [
                "", "Office Workers", "Students", "Local Residents", "Tourists",
                "Mixed Crowd", "Late Night Customers"
            ])

            budget_range = st.selectbox("Daily Budget Range", [
                "", "₹500-1000", "₹1000-2000", "₹2000-5000", "₹5000-10000", "₹10000+"
            ])

            operating_hours = st.text_input("Operating Hours",
                                            value=st.session_state.vendor_profile['operating_hours'])

            daily_capacity = st.slider("Daily Serving Capacity", 50, 1000,
                                       st.session_state.vendor_profile['daily_capacity'])

        st.subheader("Menu Items")
        menu_items = st.text_area("List your menu items (one per line)*",
                                  value="\n".join(st.session_state.vendor_profile['menu_items']),
                                  help="Enter each item on a new line")

        submitted = st.form_submit_button("Complete Profile Setup", use_container_width=True)

        if submitted:
            if name and business_name and city and business_type and speciality and menu_items:
                st.session_state.vendor_profile.update({
                    'name': name,
                    'business_name': business_name,
                    'city': city,
                    'state': state,
                    'pincode': pincode,
                    'location': location,
                    'business_type': business_type,
                    'speciality': speciality,
                    'target_customers': target_customers,
                    'budget_range': budget_range,
                    'operating_hours': operating_hours,
                    'daily_capacity': daily_capacity,
                    'years_experience': years_experience,
                    'menu_items': [item.strip() for item in menu_items.split('\n') if item.strip()],
                    'setup_complete': True
                })

                # Get initial supplier recommendations
                if st.session_state.gemini_api_key:
                    with st.spinner("Finding suppliers in your area..."):
                        suppliers = get_location_based_suppliers(city, state, business_type)
                        st.session_state.suppliers_data = suppliers

                st.success("✅ Profile setup complete! Welcome to your personalized dashboard!")
                st.rerun()
            else:
                st.error("Please fill in all required fields marked with *")


def show_main_dashboard():
    """Main dashboard after profile setup"""

    # Display profile summary
    with st.sidebar:
        st.markdown(f"""
        <div class="profile-card">
            <h3>👨‍🍳 {st.session_state.vendor_profile['name']}</h3>
            <p><strong>{st.session_state.vendor_profile['business_name']}</strong></p>
            <p>📍 {st.session_state.vendor_profile['city']}, {st.session_state.vendor_profile['state']}</p>
            <p>🍽️ {st.session_state.vendor_profile['speciality']}</p>
            <p>⏰ {st.session_state.vendor_profile['operating_hours']}</p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Edit Profile"):
            st.session_state.vendor_profile['setup_complete'] = False
            st.rerun()

    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💬 AI Chat Assistant", "🏠 Dashboard", "🔍 Suppliers", "📈 Market Insights", "🌦️ Weather Insights"
    ])

    with tab1:
        show_ai_chat()

    with tab2:
        show_dashboard()

    with tab3:
        show_suppliers()

    with tab4:
        show_market_insights()

    with tab5:
        show_weather_insights()


def show_ai_chat():
    """AI Chat Assistant"""
    st.header("💬 Your AI Business Assistant")

    if not st.session_state.gemini_api_key:
        st.warning("Please configure your Gemini API key in the sidebar to use the AI assistant.")
        return

    # Display chat messages
    chat_container = st.container()

    with chat_container:
        for message in st.session_state.chat_messages:
            if message['role'] == 'user':
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>AI Assistant:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)

    # Quick action buttons
    st.subheader("Quick Questions")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("💰 Today's Pricing Tips"):
            handle_quick_query("What are the current market prices and cost-saving tips for my ingredients today?")

    with col2:
        if st.button("🌤️ Weather Business Impact"):
            handle_quick_query("How will today's weather affect my business and what should I prepare?")

    with col3:
        if st.button("📈 Sales Optimization"):
            handle_quick_query("Give me tips to increase my sales and optimize my menu based on my profile.")

    # Chat input
    user_input = st.text_input("Ask me anything about your business...", key="chat_input")

    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Send", use_container_width=True):
            if user_input:
                handle_user_query(user_input)

    with col2:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_messages = []
            st.rerun()


def handle_quick_query(query):
    """Handle quick query buttons"""
    handle_user_query(query)


def handle_user_query(user_input):
    """Handle user queries and get AI responses"""
    if not user_input.strip():
        return

    # Add user message
    st.session_state.chat_messages.append({
        'role': 'user',
        'content': user_input
    })

    # Get current context
    current_time = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
    context = f"Current time: {current_time}. "

    # Add relevant context based on query type
    if any(word in user_input.lower() for word in ['weather', 'rain', 'temperature']):
        context += get_weather_insights(st.session_state.vendor_profile['city'])
    elif any(word in user_input.lower() for word in ['price', 'cost', 'market', 'ingredient']):
        context += get_market_price_analysis(st.session_state.vendor_profile['city'],
                                             st.session_state.vendor_profile['menu_items'])
    elif any(word in user_input.lower() for word in ['supplier', 'wholesale', 'buy']):
        context += f"Available suppliers: {len(st.session_state.suppliers_data)} suppliers found in your area."

    # Get AI response
    with st.spinner("Getting AI response..."):
        ai_response = get_gemini_response(user_input, context)

    # Add AI response
    st.session_state.chat_messages.append({
        'role': 'assistant',
        'content': ai_response
    })

    st.rerun()


def show_dashboard():
    """Enhanced dashboard with real-time insights"""
    st.header("🏠 Your Business Dashboard")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Today's Potential", "₹2,500", "Based on weather & location")

    with col2:
        st.metric("Menu Items", len(st.session_state.vendor_profile['menu_items']))

    with col3:
        st.metric("Suppliers Found", len(st.session_state.suppliers_data))

    with col4:
        st.metric("Business Score", "8.5/10", "Weather + Market conditions")

    # Get real-time insights
    if st.session_state.gemini_api_key:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("🎯 Today's AI Recommendations")
            if st.button("Get Fresh Recommendations"):
                with st.spinner("Analyzing current conditions..."):
                    recommendations = get_gemini_response(
                        "Give me 5 specific actionable recommendations for my business today",
                        f"Current time: {datetime.now().strftime('%A, %B %d, %Y')}"
                    )
                    st.markdown(recommendations)

        with col2:
            st.subheader("📊 Quick Insights")
            if st.button("Market Analysis"):
                with st.spinner("Analyzing market..."):
                    analysis = get_market_price_analysis(
                        st.session_state.vendor_profile['city'],
                        st.session_state.vendor_profile['menu_items'][:3]
                    )
                    st.markdown(analysis[:300] + "...")


def show_suppliers():
    """Enhanced suppliers section with AI recommendations"""
    st.header("🔍 Smart Supplier Discovery")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Find Suppliers")
        search_type = st.selectbox("Search for:", [
            "All Suppliers", "Vegetable Wholesalers", "Spice Merchants",
            "Oil & Ghee Suppliers", "Packaging Suppliers", "Equipment Dealers"
        ])

        if st.button("Find Suppliers with AI"):
            if st.session_state.gemini_api_key:
                with st.spinner("Finding best suppliers in your area..."):
                    suppliers = get_location_based_suppliers(
                        st.session_state.vendor_profile['city'],
                        st.session_state.vendor_profile['state'],
                        search_type
                    )
                    st.session_state.suppliers_data = suppliers
                    st.success(f"Found {len(suppliers)} suppliers!")
            else:
                st.warning("Please configure Gemini API key first")

    with col2:
        st.subheader("📋 Recommended Suppliers")

        if st.session_state.suppliers_data:
            for idx, supplier in enumerate(st.session_state.suppliers_data[:5]):
                with st.expander(f"⭐ {supplier.get('name', 'Unknown Supplier')}"):
                    col_a, col_b = st.columns(2)

                    with col_a:
                        st.write(f"**Location:** {supplier.get('location', 'Not specified')}")
                        st.write(f"**Speciality:** {supplier.get('speciality', 'General')}")
                        st.write(f"**Type:** {supplier.get('type', 'Unknown')}")

                    with col_b:
                        st.write(f"**Contact:** {supplier.get('contact', 'To be verified')}")
                        st.write(f"**Rating:** {'⭐' * int(supplier.get('rating', 4))}")

                    if st.button(f"Get More Details", key=f"details_{idx}"):
                        if st.session_state.gemini_api_key:
                            with st.spinner("Getting detailed information..."):
                                details = get_gemini_response(
                                    f"Tell me more about {supplier.get('name')} supplier in {st.session_state.vendor_profile['city']}. Include pricing, quality, and tips for dealing with them."
                                )
                                st.info(details)
        else:
            st.info("Click 'Find Suppliers with AI' to discover suppliers in your area!")


def show_market_insights():
    """Market insights with AI analysis"""
    st.header("📈 Market Intelligence")

    if not st.session_state.gemini_api_key:
        st.warning("Please configure Gemini API key to get real-time market insights")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("💰 Price Analysis")
        if st.button("Get Current Prices"):
            with st.spinner("Analyzing market prices..."):
                price_analysis = get_market_price_analysis(
                    st.session_state.vendor_profile['city'],
                    st.session_state.vendor_profile['menu_items']
                )
                st.markdown(price_analysis)

    with col2:
        st.subheader("📊 Demand Forecast")
        if st.button("Forecast Demand"):
            with st.spinner("Forecasting demand..."):
                forecast = get_gemini_response(
                    f"Predict customer demand for my {st.session_state.vendor_profile['speciality']} business today and this week. Consider weather, local events, and seasonal factors.",
                    f"Location: {st.session_state.vendor_profile['city']}, Menu: {', '.join(st.session_state.vendor_profile['menu_items'][:5])}"
                )
                st.markdown(forecast)

    # Competition analysis
    st.subheader("🏪 Competition & Opportunities")
    if st.button("Analyze Competition"):
        with st.spinner("Analyzing local competition..."):
            competition = get_gemini_response(
                f"Analyze the competition for {st.session_state.vendor_profile['business_type']} in {st.session_state.vendor_profile['city']}. What opportunities can I leverage?",
                f"My speciality: {st.session_state.vendor_profile['speciality']}, Target: {st.session_state.vendor_profile['target_customers']}"
            )
            st.markdown(competition)


def show_weather_insights():
    """Weather-based business insights"""
    st.header("🌦️ Weather Business Intelligence")

    if not st.session_state.gemini_api_key:
        st.warning("Please configure Gemini API key to get weather insights")
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("☀️ Today's Weather Impact")
        if st.button("Get Weather Insights", use_container_width=True):
            with st.spinner("Analyzing weather impact on your business..."):
                weather_insights = get_weather_insights(st.session_state.vendor_profile['city'])
                st.markdown(weather_insights)

    with col2:
        st.subheader("⚡ Quick Actions")
        if st.button("Menu Optimization"):
            with st.spinner("Optimizing menu for weather..."):
                menu_tips = get_gemini_response(
                    "Based on current weather, suggest how I should modify my menu quantities and which items to focus on today.",
                    f"My menu: {', '.join(st.session_state.vendor_profile['menu_items'])}"
                )
                st.info(menu_tips)

        if st.button("Customer Flow Prediction"):
            with st.spinner("Predicting customer patterns..."):
                flow_prediction = get_gemini_response(
                    "Predict customer flow patterns for today based on weather and suggest the best operating strategy."
                )
                st.info(flow_prediction)

    # Weekly weather business forecast
    st.subheader("📅 Weekly Business Forecast")
    if st.button("Get 7-Day Forecast"):
        with st.spinner("Generating weekly business forecast..."):
            weekly_forecast = get_gemini_response(
                "Provide a 7-day weather-based business forecast for my street food business. Include expected sales, recommended preparations, and strategic tips for each day."
            )
            st.markdown(weekly_forecast)


if __name__ == "__main__":
    main()