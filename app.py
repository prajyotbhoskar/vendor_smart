import streamlit as st
import json
import os
from datetime import datetime, timedelta
import pandas as pd
import uuid
import time
import requests  # For general API calls like Gemini
import requests_cache  # For Open-Meteo Weather API caching
from retry_requests import retry  # For Open-Meteo Weather API retries
import openmeteo_requests  # Open-Meteo Weather API client
import google.generativeai as genai  # Gemini API client

# --- Configuration ---
st.set_page_config(
    page_title="RasoiLink - Smart Supply Chain for Street Food Vendors",
    page_icon="🍛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Initialize Session State ---
if 'user_type' not in st.session_state:
    st.session_state.user_type = None
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_data' not in st.session_state:
    st.session_state.user_data = {}
if 'cart' not in st.session_state:
    st.session_state.cart = []
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'market_data' not in st.session_state:
    st.session_state.market_data = pd.DataFrame()  # Store fetched market data
if 'weather_data' not in st.session_state:
    st.session_state.weather_data = {}  # Store fetched weather data

# --- API Key Configuration ---
try:
    # Attempt to load from Streamlit secrets
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    DATA_GOV_IN_API_KEY = st.secrets["DATA_GOV_IN_API_KEY"]
except KeyError:
    st.error("API keys not found in .streamlit/secrets.toml. Please set GEMINI_API_KEY and DATA_GOV_IN_API_KEY.")
    st.stop()  # Stop the app if keys are not set

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)


# --- Data Storage Functions (using JSON files for demo) ---
def load_data(filename):
    """Load data from JSON file"""
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                # Handle cases where the JSON file is empty or corrupted
                st.warning(f"Warning: {filename} is empty or corrupted. Initializing with empty data.")
                return {}  # Return empty if file is corrupt
    return {}


def save_data(data, filename):
    """Save data to JSON file"""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)


# --- Weather API Function ---
# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Default location for weather (e.g., Mumbai)
DEFAULT_LATITUDE = 19.0760
DEFAULT_LONGITUDE = 72.8777


@st.cache_data(ttl=3600)  # Cache weather data for 1 hour
def get_current_weather(latitude=DEFAULT_LATITUDE, longitude=DEFAULT_LONGITUDE):
    """Fetches current and daily weather data from Open-Meteo."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": ["temperature_2m", "relative_humidity_2m", "apparent_temperature", "is_day", "precipitation", "rain",
                    "showers", "snowfall", "weather_code", "cloud_cover", "wind_speed_10m", "wind_direction_10m"],
        "daily": ["weather_code", "temperature_2m_max", "temperature_2m_min", "precipitation_sum",
                  "precipitation_hours"],
        "timezone": "auto",
        "forecast_days": 3  # Get forecast for 3 days
    }
    try:
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]

        # Current weather data
        current = response.Current()
        current_data = {
            "time": datetime.fromtimestamp(current.Time() + response.UtcOffsetSeconds(),
                                           tz=datetime.now().astimezone().tzinfo).strftime('%Y-%m-%d %H:%M'),
            "temperature_2m": current.Variables(0).Value(),
            "relative_humidity_2m": current.Variables(1).Value(),
            "apparent_temperature": current.Variables(2).Value(),
            "is_day": current.Variables(3).Value(),
            "precipitation": current.Variables(4).Value(),
            "rain": current.Variables(5).Value(),
            "showers": current.Variables(6).Value(),
            "snowfall": current.Variables(7).Value(),
            "weather_code": current.Variables(8).Value(),
            "cloud_cover": current.Variables(9).Value(),
            "wind_speed_10m": current.Variables(10).Value(),
            "wind_direction_10m": current.Variables(11).Value()
        }

        # Daily weather data
        daily = response.Daily()
        daily_data = {
            "time": pd.to_datetime(daily.Time(), unit="s", utc=True).tz_convert(datetime.now().astimezone().tzinfo),
            "weather_code": daily.Variables(0).ValuesAsNumpy(),
            "temperature_2m_max": daily.Variables(1).ValuesAsNumpy(),
            "temperature_2m_min": daily.Variables(2).ValuesAsNumpy(),
            "precipitation_sum": daily.Variables(3).ValuesAsNumpy(),
            "precipitation_hours": daily.Variables(4).ValuesAsNumpy()
        }
        daily_df = pd.DataFrame(daily_data)

        weather_codes = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Fog", 48: "Depositing rime fog", 51: "Drizzle: Light", 53: "Drizzle: Moderate",
            55: "Drizzle: Dense intensity",
            56: "Freezing Drizzle: Light", 57: "Freezing Drizzle: Dense intensity", 61: "Rain: Slight",
            63: "Rain: Moderate", 65: "Rain: Heavy intensity",
            66: "Freezing Rain: Light", 67: "Freezing Rain: Heavy intensity", 71: "Snow fall: Slight",
            73: "Snow fall: Moderate", 75: "Snow fall: Heavy intensity",
            77: "Snow grains", 80: "Rain showers: Slight", 81: "Rain showers: Moderate", 82: "Rain showers: Violent",
            85: "Snow showers: Slight", 86: "Snow showers: Heavy", 95: "Thunderstorm: Slight or moderate",
            96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail"
        }
        current_data['weather_description'] = weather_codes.get(current_data['weather_code'], "Unknown")
        daily_df['weather_description'] = daily_df['weather_code'].map(weather_codes)

        return {"current": current_data, "daily": daily_df.to_dict(orient='records')}
    except Exception as e:
        st.error(f"Error fetching weather data: {e}. Check internet connection or Open-Meteo API.")
        return {}


# --- Market Data Functions ---
@st.cache_data(ttl=3600)  # Cache market data for 1 hour
def fetch_market_data(state=None, district=None, market=None, commodity=None):
    """Fetch data from the agricultural market API.
    If the API fails or returns empty data, fallback to the CSV file.
    Filters (state, district, market, commodity) are applied manually on CSV data.
    """
    base_url = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0079"  # Corrected URL

    params = {
        "api-key": DATA_GOV_IN_API_KEY,
        "format": "json",
        "limit": 500  # Increased limit for more data
    }

    df = pd.DataFrame()
    st.info("Attempting to fetch market data...")  # Debug: start
    try:
        response = requests.get(base_url, params=params, timeout=10)  # Added timeout
        st.write(f"API Response Status Code: {response.status_code}")  # Debug: API status
        if response.status_code == 200:
            data = response.json()
            records = data.get("records", [])
            st.write(f"Number of records from API: {len(records)}")  # Debug: API records count
            if records:
                df = pd.DataFrame(records)
                rename_mapping_api = {
                    'state': 'state', 'district': 'district', 'market': 'market',
                    'commodity': 'commodity', 'variety': 'variety', 'grade': 'grade',
                    'arrival_date': 'arrival_date', 'min_price': 'min_price',
                    'max_price': 'max_price', 'modal_price': 'modal_price'
                }
                df.rename(columns=rename_mapping_api, inplace=True)
                st.write("API data fetched successfully.")  # Debug: API success
            else:
                st.warning("API returned empty records. Falling back to CSV file.")
                # We don't raise an exception here to let the CSV fallback handle it smoothly
        else:
            st.warning(f"API Error: {response.status_code}. Falling back to CSV file.")
    except requests.exceptions.RequestException as req_e:
        st.error(f"Network or API request error: {req_e}. Falling back to CSV file.")  # Debug: Specific network error
    except Exception as e:
        st.error(
            f"An unexpected error fetching from API: {str(e)}. Falling back to CSV file.")  # Debug: General API error

    # If df is still empty after API attempt, or if an error occurred, try CSV
    if df.empty:
        st.info("Attempting to load data from final_price_data.csv as a fallback.")  # Debug: CSV path check
        if os.path.exists("final_price_data.csv"):
            try:
                df_csv = pd.read_csv("final_price_data.csv")
                rename_mapping_csv = {
                    'State': 'state', 'District': 'district', 'Market': 'market',
                    'Commodity': 'commodity', 'Variety': 'variety', 'Grade': 'grade',
                    'Arrival_Date': 'arrival_date', 'Min_x0020_Price': 'min_price',
                    'Max_x0020_Price': 'max_price', 'Modal_x0020_Price': 'modal_price'
                }
                df_csv.rename(columns=rename_mapping_csv, inplace=True)
                df = df_csv  # Use CSV data if loaded successfully
                st.write(f"CSV data loaded. Initial rows: {len(df)}")  # Debug: CSV loaded count
            except Exception as csv_e:
                st.error(
                    f"Error loading CSV file: {csv_e}. Check CSV format and content (e.g., column names).")  # Debug: CSV read error
                return pd.DataFrame()  # Return empty if CSV fails too
        else:
            st.error("CSV fallback file 'final_price_data.csv' not found. Please ensure it's in the same directory.")

    if df.empty:
        st.info("No market data available after attempting API and CSV load.")  # Debug: Final check
        return pd.DataFrame()

    # Convert prices to numeric
    for col in ['min_price', 'max_price', 'modal_price']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    st.write(f"Rows before dropping NaNs (modal_price): {len(df)}")  # Debug: Before dropna
    df.dropna(subset=['modal_price'], inplace=True)  # Drop rows where modal price couldn't be converted
    st.write(f"Rows after dropping NaNs (modal_price): {len(df)}")  # Debug: After dropna

    # Apply filters
    if state:
        df = df[df['state'] == state]
    if district:
        df = df[df['district'] == district]
    if market:
        df = df[df['market'] == market]
    if commodity:
        df = df[df['commodity'] == commodity]

    st.write(f"Final rows after all filters: {len(df)}")  # Debug: Final filtered count
    return df


# --- Enhanced AI Insight Function ---
def get_ai_insights(market_data_df, weather_data, selected_state, selected_district, selected_market=None,
                    selected_commodity=None):
    """Get enhanced insights from Gemini API with focus on profitable suggestions for vendors,
       incorporating market and weather data.
    """
    insight_text = ""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')

        # Prepare market data for prompt
        market_summary = ""
        if not market_data_df.empty:
            market_summary += "\n--- Market Data Summary ---\n"
            market_summary += f"Data for State: {selected_state}, District: {selected_district}\n"
            if selected_market: market_summary += f"Specific Market: {selected_market}\n"
            if selected_commodity: market_summary += f"Specific Commodity: {selected_commodity}\n"

            # Basic stats
            market_summary += f"Total unique commodities: {market_data_df['commodity'].nunique()}\n"
            market_summary += f"Average Modal Price: ₹{market_data_df['modal_price'].mean():.2f}\n"

            # Top 5 cheapest/costliest commodities (if not specific commodity selected)
            if not selected_commodity:
                cheapest_crops = market_data_df.sort_values('modal_price', ascending=True).head(5)
                costliest_crops = market_data_df.sort_values('modal_price', ascending=False).head(5)
                if not cheapest_crops.empty:
                    market_summary += "\nTop 5 Cheapest Commodities (Modal Price):\n"
                    for _, row in cheapest_crops.iterrows():
                        market_summary += f"- {row['commodity']} ({row['market']}): ₹{row['modal_price']:.2f}/unit\n"
                if not costliest_crops.empty:
                    market_summary += "\nTop 5 Costliest Commodities (Modal Price):\n"
                    for _, row in costliest_crops.iterrows():
                        market_summary += f"- {row['commodity']} ({row['market']}): ₹{row['modal_price']:.2f}/unit\n"
            else:
                # Specific commodity details
                comm_data = market_data_df[market_data_df['commodity'] == selected_commodity]
                if not comm_data.empty:
                    market_summary += f"\nDetails for {selected_commodity}:\n"
                    market_summary += f"  Min Price: ₹{comm_data['min_price'].min():.2f}\n"
                    market_summary += f"  Max Price: ₹{comm_data['max_price'].max():.2f}\n"
                    market_summary += f"  Modal Price: ₹{comm_data['modal_price'].mean():.2f}\n"
                    market_summary += f"  Price variance (Std Dev): ₹{comm_data['modal_price'].std():.2f}\n"
                    # Add simple trend if multiple dates for commodity
                    comm_data['arrival_date'] = pd.to_datetime(comm_data['arrival_date'], errors='coerce')
                    comm_data = comm_data.sort_values('arrival_date').dropna(subset=['arrival_date'])
                    if len(comm_data['arrival_date'].unique()) > 1:
                        oldest_price = comm_data.iloc[0]['modal_price']
                        newest_price = comm_data.iloc[-1]['modal_price']
                        price_diff = newest_price - oldest_price
                        trend = "increased" if price_diff > 0 else (
                            "decreased" if price_diff < 0 else "remained stable")
                        market_summary += f"  Price trend over available dates: {trend} by ₹{abs(price_diff):.2f}\n"
        else:
            market_summary += "\nNo specific market data available for selected filters.\n"

        # Prepare weather data for prompt
        weather_summary = "\n--- Weather Data Summary ---\n"
        if weather_data and weather_data.get('current') and weather_data.get('daily'):
            current = weather_data['current']
            daily = pd.DataFrame(weather_data['daily'])
            weather_summary += f"Current Weather ({current['time']}):\n"
            weather_summary += f"- Temperature: {current['temperature_2m']}°C (Feels like {current['apparent_temperature']}°C)\n"
            weather_summary += f"- Condition: {current['weather_description']} (Precipitation: {current['precipitation']} mm)\n"
            weather_summary += f"- Wind: {current['wind_speed_10m']} km/h\n"

            weather_summary += "\nUpcoming 3-Day Forecast:\n"
            for _, row in daily.head(3).iterrows():
                weather_summary += f"- {row['time'].strftime('%Y-%m-%d')}: {row['weather_description']}, Max Temp: {row['temperature_2m_max']}°C, Min Temp: {row['temperature_2m_min']}°C, Total Rain: {row['precipitation_sum']} mm\n"
        else:
            weather_summary += "No weather data available.\n"

        prompt = f"""
        You are an AI assistant for a street food vendor supply chain platform (RasoiLink).
        Provide smart and actionable insights and suggestions for a street food vendor based on the following market and weather data.
        Focus on helping them make profitable purchasing decisions, avoid spoilage, and optimize stock.

        {market_summary}
        {weather_summary}

        Analyze this combined data and provide:

        1.  **Market Price Trends & Profitability:**
            *   Detailed insights on price changes for specific commodities (e.g., "Tomato prices have increased by X% this week...").
            *   Which commodities are currently cheapest/costliest and why (if data suggests).
            *   Suggestions for profitable buying or avoiding certain commodities.

        2.  **Weather Impact & Risk Mitigation:**
            *   How the current and forecasted weather might affect supply, demand, or spoilage of perishable goods (e.g., "Heavy rains predicted may disrupt vegetable supply, consider stocking non-perishables.").
            *   Recommendations for storage or purchasing based on weather.

        3.  **Smart Buying Suggestions (Actionable Advice):**
            *   Specific, actionable tips for the vendor (e.g., "Buy onions in bulk this week", "Consider cauliflower as an alternative to cabbage due to price difference.").
            *   Suggestions related to group buying or peak buying times if applicable.

        Ensure all insights are clear, concise, and directly applicable to a street food vendor's daily operations. Use emojis for readability.
        """

        # Use a higher temperature for more creative/diverse insights if needed
        response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0.7))
        insight_text = response.text
    except Exception as e:
        insight_text = f"An error occurred while fetching AI insights: {e}. Please ensure your Gemini API key is correct and try again."
        st.error(insight_text)
    return insight_text


# --- Authentication Functions ---
def register_user(email, password, name, phone, user_type, address=""):
    """Register a new user"""
    users = load_data('users.json')
    if email in users:
        return False, "User already exists"

    users[email] = {
        'password': password,  # In real app, hash this!
        'name': name,
        'phone': phone,
        'user_type': user_type,
        'address': address,
        'created_at': datetime.now().isoformat()
    }
    save_data(users, 'users.json')
    return True, "Registration successful"


def login_user(email, password):
    """Login user"""
    users = load_data('users.json')
    if email in users and users[email]['password'] == password:
        return True, users[email]
    return False, None


# --- Product Management ---
def add_product(supplier_email, name, price, category, stock, delivery_area, description=""):
    """Add product by supplier"""
    products = load_data('products.json')
    product_id = str(uuid.uuid4())

    products[product_id] = {
        'supplier_email': supplier_email,
        'name': name,
        'price': float(price),
        'category': category,
        'stock': int(stock),
        'delivery_area': delivery_area,
        'description': description,
        'created_at': datetime.now().isoformat(),
        'active': True
    }
    save_data(products, 'products.json')
    return product_id


def get_products():
    """Get all active products"""
    products = load_data('products.json')
    return {k: v for k, v in products.items() if v.get('active', True)}


# --- Order Management ---
def place_order(vendor_email, items, total_amount):
    """Place an order"""
    orders = load_data('orders.json')
    order_id = str(uuid.uuid4())

    orders[order_id] = {
        'vendor_email': vendor_email,
        'items': items,
        'total_amount': float(total_amount),
        'status': 'pending',
        'created_at': datetime.now().isoformat(),
        'estimated_delivery': (datetime.now() + timedelta(days=2)).isoformat()
    }
    save_data(orders, 'orders.json')
    return order_id


def get_vendor_orders(vendor_email):
    """Get orders for a vendor"""
    orders = load_data('orders.json')
    return {k: v for k, v in orders.items() if v['vendor_email'] == vendor_email}


def get_supplier_orders(supplier_email):
    """Get orders for a supplier"""
    orders = load_data('orders.json')
    products = get_products()
    supplier_orders = {}

    for order_id, order in orders.items():
        for item in order['items']:
            if item['product_id'] in products and products[item['product_id']]['supplier_email'] == supplier_email:
                if order_id not in supplier_orders:
                    supplier_orders[order_id] = order.copy()
                    supplier_orders[order_id]['relevant_items'] = []
                supplier_orders[order_id]['relevant_items'].append(item)

    return supplier_orders


# --- CSS for better styling ---
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #FF6B35;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .insight-box {
        background: linear-gradient(90deg, #FF6B35, #F7931E);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .product-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: #f9f9f9;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .sidebar-logo {
        text-align: center;
        font-size: 2rem;
        color: #FF6B35;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-box h3 {
        color: #FF6B35;
        font-size: 1.5rem;
    }
    .metric-box p {
        font-size: 1.2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar Navigation ---
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🍛 RasoiLink</div>', unsafe_allow_html=True)

    if not st.session_state.logged_in:
        st.write("### Welcome!")
        st.write("Connect street food vendors with suppliers")

        if st.button("🏠 Home", use_container_width=True):
            st.session_state.page = 'home'
        if st.button("👨‍🍳 Vendor Login", use_container_width=True):
            st.session_state.page = 'vendor_login'
        if st.button("🏪 Supplier Login", use_container_width=True):
            st.session_state.page = 'supplier_login'
        if st.button("📝 Register", use_container_width=True):
            st.session_state.page = 'register'
    else:
        st.write(f"Welcome, **{st.session_state.user_data['name']}**!")
        st.write(f"Type: {st.session_state.user_data['user_type'].capitalize()}")

        if st.session_state.user_data['user_type'] == 'vendor':
            if st.button("🛍️ Browse Products", use_container_width=True):
                st.session_state.page = 'vendor_dashboard'
            if st.button("📦 My Orders", use_container_width=True):
                st.session_state.page = 'vendor_orders'
            if st.button("🛒 Cart ({})".format(len(st.session_state.cart)), use_container_width=True):
                st.session_state.page = 'cart'
            if st.button("🤖 AI Assistant", use_container_width=True):
                st.session_state.page = 'ai_assistant'
        else:
            if st.button("📊 Dashboard", use_container_width=True):
                st.session_state.page = 'supplier_dashboard'
            if st.button("➕ Add Product", use_container_width=True):
                st.session_state.page = 'add_product'
            if st.button("📋 Orders", use_container_width=True):
                st.session_state.page = 'supplier_orders'

        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user_data = {}
            st.session_state.cart = []
            st.session_state.page = 'home'
            st.rerun()

# --- Main content area ---
if st.session_state.page == 'home':
    st.markdown('<h1 class="main-header">🍛 RasoiLink</h1>', unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; font-size: 1.2rem; margin-bottom: 2rem;">
        <p>🎯 <strong>Connecting Street Food Vendors with Suppliers</strong></p>
        <p>Smart supply chain solution powered by AI insights</p>
    </div>
    """, unsafe_allow_html=True)

    # Fetch and display current weather and market data for general insights on home page
    with st.spinner("Fetching general market and weather data..."):
        st.session_state.weather_data = get_current_weather()
        st.session_state.market_data = fetch_market_data()  # Fetch all data for general overview

    # Market Insights Section
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown("### 📊 Today's Market Insights & Weather Overview")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Current Weather")
        if st.session_state.weather_data and st.session_state.weather_data.get('current'):
            current_w = st.session_state.weather_data['current']
            st.write(f"📍 Location: Mumbai, India (Default)")
            st.write(
                f"🌡️ Temperature: **{current_w['temperature_2m']}°C** (Feels like {current_w['apparent_temperature']}°C)")
            st.write(f"☁️ Condition: **{current_w['weather_description']}**")
            st.write(f"💧 Humidity: {current_w['relative_humidity_2m']}%")
            st.write(f"💨 Wind: {current_w['wind_speed_10m']} km/h")
        else:
            st.info("Weather data not available.")

    with col2:
        st.subheader("Market Overview (General)")
        if not st.session_state.market_data.empty:
            total_commodities = st.session_state.market_data['commodity'].nunique()
            avg_price = st.session_state.market_data['modal_price'].mean()
            st.write(f"📦 Total Commodities tracked: **{total_commodities}**")
            st.write(f"💰 Average Modal Price: **₹{avg_price:.2f}/unit**")

            # Display a few top/bottom commodities for general trends
            cheapest = st.session_state.market_data.sort_values('modal_price').head(3)
            costliest = st.session_state.market_data.sort_values('modal_price', ascending=False).head(3)
            st.write("---")
            st.write("Lowest Price Commodities:")
            for _, row in cheapest.iterrows():
                st.write(f"• {row['commodity']} ({row['market']}): ₹{row['modal_price']:.2f}")
            st.write("Highest Price Commodities:")
            for _, row in costliest.iterrows():
                st.write(f"• {row['commodity']} ({row['market']}): ₹{row['modal_price']:.2f}")
        else:
            st.info("Market data not available.")

    st.markdown('</div>', unsafe_allow_html=True)

    # Smart Suggestions based on general market trends
    st.markdown("### 💡 AI-Powered General Suggestions")
    if not st.session_state.market_data.empty and st.session_state.weather_data:
        general_insights = get_ai_insights(st.session_state.market_data, st.session_state.weather_data, "Maharashtra",
                                           "Mumbai")  # General insights for Mumbai
        st.markdown(general_insights)
    else:
        st.info("AI suggestions will appear here once market and weather data are available.")

elif st.session_state.page == 'register':
    st.markdown('<h1 class="main-header">📝 Register</h1>', unsafe_allow_html=True)

    with st.form("register_form"):
        user_type = st.selectbox("I am a:", ["vendor", "supplier"])
        name = st.text_input("Full Name*")
        email = st.text_input("Email*")
        phone = st.text_input("Phone Number*")
        address = st.text_area("Address")
        password = st.text_input("Password*", type="password")
        confirm_password = st.text_input("Confirm Password*", type="password")

        submitted = st.form_submit_button("Register", use_container_width=True)

        if submitted:
            if not all([name, email, phone, password, confirm_password]):
                st.error("Please fill all required fields")
            elif password != confirm_password:
                st.error("Passwords don't match")
            else:
                success, message = register_user(email, password, name, phone, user_type, address)
                if success:
                    st.success(message)
                    st.balloons()
                    time.sleep(2)
                    st.session_state.page = 'vendor_login' if user_type == 'vendor' else 'supplier_login'
                    st.rerun()
                else:
                    st.error(message)

elif st.session_state.page in ['vendor_login', 'supplier_login']:
    user_type = 'vendor' if st.session_state.page == 'vendor_login' else 'supplier'
    icon = '👨‍🍳' if user_type == 'vendor' else '🏪'

    st.markdown(f'<h1 class="main-header">{icon} {user_type.capitalize()} Login</h1>', unsafe_allow_html=True)

    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        submitted = st.form_submit_button("Login", use_container_width=True)

        if submitted:
            success, user_data = login_user(email, password)
            if success and user_data['user_type'] == user_type:
                st.session_state.logged_in = True
                st.session_state.user_data = user_data
                st.session_state.user_data['email'] = email
                st.success(f"Welcome back, {user_data['name']}!")
                time.sleep(1)
                st.session_state.page = 'vendor_dashboard' if user_type == 'vendor' else 'supplier_dashboard'
                st.rerun()
            else:
                st.error("Invalid credentials or user type mismatch")

elif st.session_state.page == 'vendor_dashboard':
    st.markdown('<h1 class="main-header">🛍️ Product Catalog</h1>', unsafe_allow_html=True)

    # Fetch all market data once for filtering
    all_market_data = fetch_market_data()
    st.session_state.market_data = all_market_data  # Update session state

    # Market Insights & Weather
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown("### 📊 Local Market & Weather Insights")

    # Dynamic location for weather/market insights (simplified for demo)
    st.write("Select location for AI insights (default: Mumbai):")
    market_states = sorted(all_market_data['state'].dropna().unique().tolist()) if not all_market_data.empty else [
        "Maharashtra"]
    selected_state = st.selectbox("State", market_states, key="insight_state_sel")

    market_districts = sorted(all_market_data[all_market_data['state'] == selected_state][
                                  'district'].dropna().unique().tolist()) if not all_market_data.empty else ["Mumbai"]
    selected_district = st.selectbox("District", market_districts, key="insight_district_sel")

    col_market_filters, col_weather_display = st.columns(2)
    with col_market_filters:
        current_markets = sorted(all_market_data[(all_market_data['state'] == selected_state) & (
                    all_market_data['district'] == selected_district)][
                                     'market'].dropna().unique().tolist()) if not all_market_data.empty else ["All"]
        selected_market_filter_ai = st.selectbox("Specific Market (for AI)", ["All"] + current_markets,
                                                 key="insight_market_filter_ai")
        current_commodities = sorted(all_market_data[(all_market_data['state'] == selected_state) & (
                    all_market_data['district'] == selected_district)][
                                         'commodity'].dropna().unique().tolist()) if not all_market_data.empty else [
            "All"]
        selected_commodity_filter_ai = st.selectbox("Specific Commodity (for AI)", ["All"] + current_commodities,
                                                    key="insight_commodity_filter_ai")

        # Apply filters to market data for AI
        filtered_market_data_for_ai = all_market_data[
            (all_market_data['state'] == selected_state) &
            (all_market_data['district'] == selected_district)
            ]
        if selected_market_filter_ai != "All":
            filtered_market_data_for_ai = filtered_market_data_for_ai[
                filtered_market_data_for_ai['market'] == selected_market_filter_ai]
        if selected_commodity_filter_ai != "All":
            filtered_market_data_for_ai = filtered_market_data_for_ai[
                filtered_market_data_for_ai['commodity'] == selected_commodity_filter_ai]

        if st.button("Generate AI Insights", key="gen_ai_insight_dashboard"):
            with st.spinner("Generating AI insights based on market and weather data..."):
                st.session_state.weather_data = get_current_weather()  # Re-fetch weather if needed
                ai_insights_text = get_ai_insights(
                    filtered_market_data_for_ai,
                    st.session_state.weather_data,
                    selected_state,
                    selected_district,
                    selected_market_filter_ai if selected_market_filter_ai != "All" else None,
                    selected_commodity_filter_ai if selected_commodity_filter_ai != "All" else None
                )
                st.session_state.last_ai_insights = ai_insights_text

    with col_weather_display:
        st.subheader("Current Weather Forecast:")
        if st.session_state.weather_data and st.session_state.weather_data.get('current'):
            current_w = st.session_state.weather_data['current']
            st.write(f"🌡️ Temp: {current_w['temperature_2m']}°C, Feels like {current_w['apparent_temperature']}°C")
            st.write(f"☁️ Condition: {current_w['weather_description']}")
            st.write(f"💧 Rain (last hour): {current_w['precipitation']} mm")

            st.write("---")
            st.write("Next 3 Days:")
            daily_df = pd.DataFrame(st.session_state.weather_data['daily'])
            for _, row in daily_df.head(3).iterrows():
                st.write(
                    f"**{row['time'].strftime('%b %d')}:** {row['weather_description']}, {row['temperature_2m_min']} - {row['temperature_2m_max']}°C, Rain: {row['precipitation_sum']} mm")
        else:
            st.info("Weather data not available. Check internet connection or API keys.")

    if 'last_ai_insights' in st.session_state:
        st.markdown(st.session_state.last_ai_insights)
    else:
        st.info("Select filters and click 'Generate AI Insights' to get tailored suggestions.")

    st.markdown('</div>', unsafe_allow_html=True)

    # Product Filters for browsing
    st.markdown("### 🛒 Browse Products")
    col1, col2, col3 = st.columns(3)
    with col1:
        category_filter = st.selectbox("Filter by Category",
                                       ["All", "Vegetables", "Grains", "Spices", "Oil", "Dairy", "Other"])
    with col2:
        price_range = st.selectbox("Filter by Price Range", ["All", "Under ₹50", "₹50-₹100", "Above ₹100"])
    with col3:
        sort_by = st.selectbox("Sort products by", ["Name", "Price (Low to High)", "Price (High to Low)"])

    products = get_products()
    filtered_products = []

    for product_id, product in products.items():
        # Apply category filter
        if category_filter != "All" and product['category'] != category_filter:
            continue

        # Apply price range filter
        if price_range == "Under ₹50" and product['price'] >= 50:
            continue
        elif price_range == "₹50-₹100" and (product['price'] < 50 or product['price'] > 100):
            continue
        elif price_range == "Above ₹100" and product['price'] <= 100:
            continue
        filtered_products.append((product_id, product))

    # Apply sorting
    if sort_by == "Name":
        filtered_products.sort(key=lambda x: x[1]['name'])
    elif sort_by == "Price (Low to High)":
        filtered_products.sort(key=lambda x: x[1]['price'])
    elif sort_by == "Price (High to Low)":
        filtered_products.sort(key=lambda x: x[1]['price'], reverse=True)

    if filtered_products:
        for product_id, product in filtered_products:
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 2, 2, 2])

                with col1:
                    st.write(f"**{product['name']}**")
                    st.write(f"*{product['category']}*")
                    if product.get('description'):
                        st.write(f"_{product['description']}_")

                with col2:
                    st.write(f"**₹{product['price']}/kg**")
                    st.write(f"Stock: {product['stock']} kg")

                with col3:
                    st.write(f"📍 {product['delivery_area']}")

                with col4:
                    quantity = st.number_input(f"Qty (kg)", min_value=0.0, max_value=float(product['stock']),
                                               step=0.5, key=f"qty_{product_id}")
                    if st.button(f"Add to Cart", key=f"cart_{product_id}"):
                        if quantity > 0:
                            cart_item = {
                                'product_id': product_id,
                                'name': product['name'],
                                'price': product['price'],
                                'quantity': quantity,
                                'total': product['price'] * quantity,
                                'supplier_email': product['supplier_email']
                            }
                            st.session_state.cart.append(cart_item)
                            st.success(f"Added {quantity}kg of {product['name']} to cart!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.warning("Please select quantity")

                st.divider()
    else:
        st.info("No products available matching your filters.")

elif st.session_state.page == 'cart':
    st.markdown('<h1 class="main-header">🛒 Shopping Cart</h1>', unsafe_allow_html=True)

    if st.session_state.cart:
        total_amount = 0

        for i, item in enumerate(st.session_state.cart):
            col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 1])

            with col1:
                st.write(f"**{item['name']}**")
            with col2:
                st.write(f"₹{item['price']}/kg")
            with col3:
                st.write(f"{item['quantity']} kg")
            with col4:
                st.write(f"**₹{item['total']:.2f}**")
            with col5:
                if st.button("🗑️", key=f"remove_{i}"):
                    st.session_state.cart.pop(i)
                    st.rerun()

            total_amount += item['total']
            st.divider()

        st.markdown(f"### Total Amount: ₹{total_amount:.2f}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Cart", type="secondary", use_container_width=True):
                st.session_state.cart = []
                st.rerun()

        with col2:
            if st.button("Place Order", type="primary", use_container_width=True):
                order_id = place_order(st.session_state.user_data['email'],
                                       st.session_state.cart, total_amount)
                st.success(f"Order placed successfully! Order ID: {order_id}")
                st.balloons()
                st.session_state.cart = []
                time.sleep(2)
                st.session_state.page = 'vendor_orders'
                st.rerun()
    else:
        st.info("Your cart is empty. Browse products to add items!")
        if st.button("Browse Products"):
            st.session_state.page = 'vendor_dashboard'
            st.rerun()

elif st.session_state.page == 'vendor_orders':
    st.markdown('<h1 class="main-header">📦 My Orders</h1>', unsafe_allow_html=True)

    orders = get_vendor_orders(st.session_state.user_data['email'])

    if orders:
        for order_id, order in orders.items():
            with st.expander(f"Order #{order_id[:8]} - ₹{order['total_amount']:.2f} ({order['status'].upper()})"):
                st.write(f"**Order Date:** {datetime.fromisoformat(order['created_at']).strftime('%Y-%m-%d %H:%M')}")
                st.write(f"**Status:** {order['status'].upper()}")
                st.write(
                    f"**Estimated Delivery:** {datetime.fromisoformat(order['estimated_delivery']).strftime('%Y-%m-%d')}")

                st.write("**Items:**")
                for item in order['items']:
                    st.write(f"• {item['name']} - {item['quantity']}kg @ ₹{item['price']}/kg = ₹{item['total']:.2f}")

                if st.button(f"Reorder", key=f"reorder_{order_id}"):
                    st.session_state.cart = order['items'].copy()
                    st.success("Items added to cart!")
                    st.session_state.page = 'cart'
                    st.rerun()
    else:
        st.info("No orders yet. Start shopping to place your first order!")

elif st.session_state.page == 'ai_assistant':
    st.markdown('<h1 class="main-header">🤖 AI Market Assistant</h1>', unsafe_allow_html=True)

    st.write("Get tailored market insights and weather-based suggestions for your purchasing decisions.")

    # Fetch all market data and weather once for the AI Assistant page
    all_market_data = fetch_market_data()
    st.session_state.market_data = all_market_data  # Update session state
    st.session_state.weather_data = get_current_weather()  # Update session state

    st.markdown("### ⚙️ Configure Your Query")

    col1, col2 = st.columns(2)
    with col1:
        market_states = sorted(all_market_data['state'].dropna().unique().tolist()) if not all_market_data.empty else [
            "Maharashtra"]
        selected_state = st.selectbox("State", market_states, key="ai_state_sel")
    with col2:
        market_districts = sorted(all_market_data[all_market_data['state'] == selected_state][
                                      'district'].dropna().unique().tolist()) if not all_market_data.empty else [
            "Mumbai"]
        selected_district = st.selectbox("District", market_districts, key="ai_district_sel")

    col3, col4 = st.columns(2)
    with col3:
        current_markets = sorted(all_market_data[(all_market_data['state'] == selected_state) & (
                    all_market_data['district'] == selected_district)][
                                     'market'].dropna().unique().tolist()) if not all_market_data.empty else ["All"]
        selected_market = st.selectbox("Specific Market", ["All"] + current_markets, key="ai_market_sel")
    with col4:
        current_commodities = sorted(all_market_data[(all_market_data['state'] == selected_state) & (
                    all_market_data['district'] == selected_district)][
                                         'commodity'].dropna().unique().tolist()) if not all_market_data.empty else [
            "All"]
        selected_commodity = st.selectbox("Specific Commodity", ["All"] + current_commodities, key="ai_commodity_sel")

    # Apply filters to market data based on user selection for AI processing
    filtered_market_data = all_market_data[
        (all_market_data['state'] == selected_state) &
        (all_market_data['district'] == selected_district)
        ]
    if selected_market != "All":
        filtered_market_data = filtered_market_data[filtered_market_data['market'] == selected_market]
    if selected_commodity != "All":
        filtered_market_data = filtered_market_data[filtered_market_data['commodity'] == selected_commodity]

    if st.button("Generate AI Assistant Insights", use_container_width=True, key="trigger_ai_assistant"):
        if selected_state and selected_district:
            with st.spinner("Analyzing market and weather data for your insights..."):
                # Pass the filtered market data and weather data to the AI function
                ai_response = get_ai_insights(
                    filtered_market_data,
                    st.session_state.weather_data,
                    selected_state,
                    selected_district,
                    selected_market if selected_market != "All" else None,
                    selected_commodity if selected_commodity != "All" else None
                )
                st.session_state.ai_assistant_response = ai_response
        else:
            st.warning("Please select a State and District to generate insights.")

    st.markdown("### 📈 Your AI Insights:")
    if 'ai_assistant_response' in st.session_state:
        st.markdown(st.session_state.ai_assistant_response)
    else:
        st.info(
            "Select your location and commodity preferences above and click 'Generate AI Assistant Insights' to get started.")

    st.markdown("---")
    st.markdown("### ☀️ Current Weather for Default Location (Mumbai)")
    if st.session_state.weather_data and st.session_state.weather_data.get('current'):
        current_w = st.session_state.weather_data['current']
        st.write(f"**Time:** {current_w['time']}")
        st.write(f"**Temperature:** {current_w['temperature_2m']}°C (Feels like {current_w['apparent_temperature']}°C)")
        st.write(f"**Condition:** {current_w['weather_description']} (Precipitation: {current_w['precipitation']} mm)")
        st.write(f"**Humidity:** {current_w['relative_humidity_2m']}%")
        st.write(f"**Wind:** {current_w['wind_speed_10m']} km/h from {current_w['wind_direction_10m']}°")

        st.subheader("Upcoming 3-Day Weather Forecast:")
        daily_df = pd.DataFrame(st.session_state.weather_data['daily'])
        if not daily_df.empty:
            for i, row in daily_df.head(3).iterrows():
                st.write(
                    f"**{row['time'].strftime('%Y-%m-%d')}:** {row['weather_description']}, Max Temp: {row['temperature_2m_min']}°C, Min Temp: {row['temperature_2m_min']}°C, Total Rain: {row['precipitation_sum']} mm")
        else:
            st.write("No daily forecast available.")
    else:
        st.info("Weather data not available. Ensure internet connection and API key.")


elif st.session_state.page == 'supplier_dashboard':
    st.markdown('<h1 class="main-header">📊 Supplier Dashboard</h1>', unsafe_allow_html=True)

    # Dashboard metrics
    products = {k: v for k, v in get_products().items()
                if v['supplier_email'] == st.session_state.user_data['email']}
    orders = get_supplier_orders(st.session_state.user_data['email'])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-box"><h3>Total Products</h3><p>{len(products)}</p></div>',
                    unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-box"><h3>Total Orders</h3><p>{len(orders)}</p></div>', unsafe_allow_html=True)
    with col3:
        total_revenue = sum(order['total_amount'] for order in orders.values())
        st.markdown(f'<div class="metric-box"><h3>Total Revenue</h3><p>₹{total_revenue:.2f}</p></div>',
                    unsafe_allow_html=True)
    with col4:
        pending_orders = len([o for o in orders.values() if o['status'] == 'pending'])
        st.markdown(f'<div class="metric-box"><h3>Pending Orders</h3><p>{pending_orders}</p></div>',
                    unsafe_allow_html=True)

    # Recent orders
    if orders:
        st.markdown("### 📋 Recent Orders")
        recent_orders = sorted(orders.items(),
                               key=lambda x: x[1]['created_at'], reverse=True)[:5]

        for order_id, order in recent_orders:
            with st.expander(f"Order #{order_id[:8]} - ₹{order['total_amount']:.2f} ({order['status'].upper()})"):
                st.write(f"**Customer:** {order['vendor_email']}")
                st.write(f"**Date:** {datetime.fromisoformat(order['created_at']).strftime('%Y-%m-%d %H:%M')}")
                st.write(f"**Status:** {order['status'].upper()}")

                if 'relevant_items' in order:
                    st.write("**Your Products in this Order:**")
                    for item in order['relevant_items']:
                        st.write(f"• {item['name']} - {item['quantity']}kg")

    # Product management
    st.markdown("### 🛍️ My Products")
    if products:
        for product_id, product in products.items():
            col1, col2, col3, col4 = st.columns([3, 2, 2, 2])

            with col1:
                st.write(f"**{product['name']}**")
                st.write(f"*{product['category']}*")
            with col2:
                st.write(f"₹{product['price']}/kg")
            with col3:
                st.write(f"Stock: {product['stock']} kg")
            with col4:
                # Placeholder for edit functionality
                if st.button(f"Edit", key=f"edit_{product_id}"):
                    st.info(f"Editing functionality for {product['name']} coming soon!")

            st.divider()
    else:
        st.info("No products added yet. Add your first product to start selling!")

elif st.session_state.page == 'add_product':
    st.markdown('<h1 class="main-header">➕ Add New Product</h1>', unsafe_allow_html=True)

    with st.form("add_product_form"):
        col1, col2 = st.columns(2)

        with col1:
            product_name = st.text_input("Product Name*")
            category = st.selectbox("Category*",
                                    ["Vegetables", "Grains", "Spices", "Oil", "Dairy", "Other"])
            price = st.number_input("Price per kg (₹)*", min_value=0.0, step=0.1)

        with col2:
            stock = st.number_input("Stock (kg)*", min_value=0, step=1)
            delivery_area = st.text_input("Delivery Area*",
                                          placeholder="e.g., Mumbai Central, Pune")
            description = st.text_area("Description (optional)")

        submitted = st.form_submit_button("Add Product", use_container_width=True)

        if submitted:
            if not all([product_name, category, price > 0, stock > 0, delivery_area]):
                st.error("Please fill all required fields")
            else:
                product_id = add_product(
                    st.session_state.user_data['email'],
                    product_name, price, category, stock,
                    delivery_area, description
                )
                st.success(f"Product '{product_name}' added successfully!")
                st.balloons()
                time.sleep(2)
                st.session_state.page = 'supplier_dashboard'
                st.rerun()

elif st.session_state.page == 'supplier_orders':
    st.markdown('<h1 class="main-header">📋 Order Management</h1>', unsafe_allow_html=True)

    orders = get_supplier_orders(st.session_state.user_data['email'])

    if orders:
        # Filter options
        status_filter = st.selectbox("Filter by Status:", ["All", "pending", "accepted", "rejected"])

        filtered_orders = orders
        if status_filter != "All":
            filtered_orders = {k: v for k, v in orders.items() if v['status'] == status_filter}

        for order_id, order in filtered_orders.items():
            with st.expander(f"Order #{order_id[:8]} - ₹{order['total_amount']:.2f} ({order['status'].upper()})"):
                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"**Customer:** {order['vendor_email']}")
                    st.write(
                        f"**Order Date:** {datetime.fromisoformat(order['created_at']).strftime('%Y-%m-%d %H:%M')}")
                    st.write(f"**Status:** {order['status'].upper()}")

                with col2:
                    st.write(f"**Total Amount:** ₹{order['total_amount']:.2f}")
                    st.write(
                        f"**Delivery Date:** {datetime.fromisoformat(order['estimated_delivery']).strftime('%Y-%m-%d')}")

                if 'relevant_items' in order:
                    st.write("**Your Products in this Order:**")
                    for item in order['relevant_items']:
                        st.write(f"• {item['name']} - {item['quantity']}kg @ ₹{item['price']}/kg")

                if order['status'] == 'pending':
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"✅ Accept", key=f"accept_{order_id}", type="primary"):
                            # Update order status
                            orders_data = load_data('orders.json')
                            orders_data[order_id]['status'] = 'accepted'
                            save_data(orders_data, 'orders.json')
                            st.success("Order accepted!")
                            st.rerun()

                    with col2:
                        if st.button(f"❌ Reject", key=f"reject_{order_id}", type="secondary"):
                            # Update order status
                            orders_data = load_data('orders.json')
                            orders_data[order_id]['status'] = 'rejected'
                            save_data(orders_data, 'orders.json')
                            st.error("Order rejected!")
                            st.rerun()
    else:
        st.info("No orders received yet. Add products to start receiving orders!")

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>🍛 <strong>RasoiLink</strong> - Empowering Street Food Vendors with Smart Supply Chain Solutions</p>
    <p>Built with ❤️ using Streamlit | Powered by AI Market Insights</p>
</div>
""", unsafe_allow_html=True)


# --- Initialize demo data on first run ---
def initialize_demo_data():
    """Initialize with some demo data for better showcase"""
    # Demo products
    if not os.path.exists('products.json'):
        demo_products = {
            "demo_001": {
                "supplier_email": "supplier1@demo.com",
                "name": "Fresh Tomatoes",
                "price": 35.0,
                "category": "Vegetables",
                "stock": 100,
                "delivery_area": "Mumbai Central",
                "description": "Fresh red tomatoes, Grade A quality",
                "created_at": datetime.now().isoformat(),
                "active": True
            },
            "demo_002": {
                "supplier_email": "supplier1@demo.com",
                "name": "Basmati Rice",
                "price": 85.0,
                "category": "Grains",
                "stock": 500,
                "delivery_area": "Mumbai, Pune",
                "description": "Premium quality basmati rice",
                "created_at": datetime.now().isoformat(),
                "active": True
            },
            "demo_003": {
                "supplier_email": "supplier2@demo.com",
                "name": "Sunflower Oil",
                "price": 120.0,
                "category": "Oil",
                "stock": 200,
                "delivery_area": "Maharashtra",
                "description": "Pure sunflower cooking oil",
                "created_at": datetime.now().isoformat(),
                "active": True
            },
            "demo_004": {
                "supplier_email": "supplier2@demo.com",
                "name": "Red Chili Powder",
                "price": 180.0,
                "category": "Spices",
                "stock": 50,
                "delivery_area": "Mumbai, Thane",
                "description": "Premium quality red chili powder",
                "created_at": datetime.now().isoformat(),
                "active": True
            },
            "demo_005": {
                "supplier_email": "supplier1@demo.com",
                "name": "Fresh Onions",
                "price": 25.0,
                "category": "Vegetables",
                "stock": 300,
                "delivery_area": "Mumbai Central",
                "description": "Fresh quality onions",
                "created_at": datetime.now().isoformat(),
                "active": True
            }
        }
        save_data(demo_products, 'products.json')

    # Demo users
    if not os.path.exists('users.json'):
        demo_users = {
            "vendor@demo.com": {
                "password": "demo123",
                "name": "Raj Sharma",
                "phone": "9876543210",
                "user_type": "vendor",
                "address": "Street Food Stall, Mumbai Central",
                "created_at": datetime.now().isoformat()
            },
            "supplier1@demo.com": {
                "password": "demo123",
                "name": "Fresh Supplies Co.",
                "phone": "9876543211",
                "user_type": "supplier",
                "address": "Wholesale Market, Mumbai",
                "created_at": datetime.now().isoformat()
            },
            "supplier2@demo.com": {
                "password": "demo123",
                "name": "Spice World Distributors",
                "phone": "9876543212",
                "user_type": "supplier",
                "address": "Spice Market, Mumbai",
                "created_at": datetime.now().isoformat()
            }
        }
        save_data(demo_users, 'users.json')


# Initialize demo data
initialize_demo_data()

# Add some helpful info in sidebar for demo
if not st.session_state.logged_in:
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🎯 Demo Credentials")
        st.markdown("**Vendor Login:**")
        st.code("Email: vendor@demo.com\nPassword: demo123")
        st.markdown("**Supplier Login:**")
        st.code("Email: supplier1@demo.com\nPassword: demo123")

        st.markdown("### 🚀 Quick Start")
        st.write("1. Login as vendor to browse & order")
        st.write("2. Login as supplier to manage products")
        st.write("3. Try the AI Assistant for market insights")

# Auto-refresh for real-time updates (optional)
# This is generally not recommended for production apps as it causes constant reruns,
# but can be useful for development/testing if you want to see immediate data updates.
# You can uncomment and adjust this if desired.
# if st.session_state.logged_in:
#     # Add a small refresh button for real-time updates
#     with st.sidebar:
#         st.markdown("---")
#         if st.button("🔄 Refresh Data"):
#             st.cache_data.clear() # Clear all caches
#             st.rerun()