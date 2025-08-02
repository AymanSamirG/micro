import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page Configuration
st.set_page_config(
    page_title="Comprehensive Pricing Strategy Framework",
    page_icon="💰",
    layout="wide"
)

class PricingStrategy:
    """Comprehensive pricing strategy framework with multiple methodologies"""
    
    def __init__(self):
        self.pricing_methods = {
            "Cost-Plus": self.cost_plus_pricing,
            "Value-Based": self.value_based_pricing,
            "Competitive": self.competitive_pricing,
            "Psychological": self.psychological_pricing,
            "Dynamic": self.dynamic_pricing
        }
    
    def cost_plus_pricing(self, cost: float, markup_percentage: float) -> float:
        """Calculate cost-plus pricing"""
        return cost * (1 + markup_percentage / 100)
    
    def value_based_pricing(self, customer_value: float, value_capture_rate: float = 0.3) -> float:
        """Calculate value-based pricing"""
        return customer_value * value_capture_rate
    
    def competitive_pricing(self, competitor_prices: List[float], strategy: str = "match") -> float:
        """Calculate competitive pricing based on strategy"""
        avg_price = np.mean(competitor_prices)
        if strategy == "premium":
            return avg_price * 1.15
        elif strategy == "discount":
            return avg_price * 0.85
        else:  # match
            return avg_price
    
    def psychological_pricing(self, base_price: float, strategy: str = "charm") -> float:
        """Apply psychological pricing strategies"""
        if strategy == "charm":
            # Charm pricing (ending in 9)
            return np.floor(base_price) + 0.99
        elif strategy == "prestige":
            # Round numbers for prestige
            return round(base_price / 10) * 10
        elif strategy == "bundle":
            # Bundling discount
            return base_price * 0.9
        return base_price
    
    def dynamic_pricing(self, base_price: float, demand_factor: float, supply_factor: float) -> float:
        """Calculate dynamic pricing based on demand and supply"""
        return base_price * demand_factor * (2 - supply_factor)

class PriceElasticityAnalyzer:
    """Analyze price elasticity and sensitivity"""
    
    @staticmethod
    def calculate_elasticity(price_changes: List[float], quantity_changes: List[float]) -> float:
        """Calculate price elasticity of demand"""
        if len(price_changes) != len(quantity_changes) or len(price_changes) < 2:
            return 0
        
        price_elasticity = []
        for i in range(1, len(price_changes)):
            price_pct_change = (price_changes[i] - price_changes[i-1]) / price_changes[i-1]
            quantity_pct_change = (quantity_changes[i] - quantity_changes[i-1]) / quantity_changes[i-1]
            
            if price_pct_change != 0:
                elasticity = quantity_pct_change / price_pct_change
                price_elasticity.append(elasticity)
        
        return np.mean(price_elasticity) if price_elasticity else 0
    
    @staticmethod
    def sensitivity_analysis(base_price: float, price_range: Tuple[float, float], 
                           elasticity: float) -> pd.DataFrame:
        """Perform price sensitivity analysis"""
        prices = np.linspace(price_range[0], price_range[1], 20)
        results = []
        
        for price in prices:
            price_change = (price - base_price) / base_price
            quantity_change = elasticity * price_change
            quantity = 100 * (1 + quantity_change)  # Base quantity = 100
            revenue = price * quantity
            
            results.append({
                'Price': price,
                'Price_Change_%': price_change * 100,
                'Quantity': max(0, quantity),
                'Revenue': revenue,
                'Elasticity_Impact': quantity_change * 100
            })
        
        return pd.DataFrame(results)

class CompetitiveAnalyzer:
    """Analyze competitive landscape and pricing positions"""
    
    @staticmethod
    def market_position_analysis(your_price: float, competitor_data: Dict[str, float]) -> Dict:
        """Analyze market position relative to competitors"""
        competitor_prices = list(competitor_data.values())
        
        analysis = {
            'your_price': your_price,
            'market_avg': np.mean(competitor_prices),
            'market_median': np.median(competitor_prices),
            'price_percentile': len([p for p in competitor_prices if p < your_price]) / len(competitor_prices) * 100,
            'premium_vs_avg': (your_price / np.mean(competitor_prices) - 1) * 100,
            'closest_competitor': min(competitor_data.items(), key=lambda x: abs(x[1] - your_price)),
            'price_gap_to_leader': max(competitor_prices) - your_price,
            'price_advantage_vs_lowest': your_price - min(competitor_prices)
        }
        
        return analysis

def main():
    st.title("🎯 Comprehensive Pricing Strategy Framework")
    st.markdown("*A complete toolkit for developing and optimizing your pricing strategy*")
    
    # Initialize classes
    pricing_strategy = PricingStrategy()
    elasticity_analyzer = PriceElasticityAnalyzer()
    competitive_analyzer = CompetitiveAnalyzer()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    sections = [
        "📊 Pricing Method Comparison",
        "💰 Cost-Plus Pricing",
        "💎 Value-Based Pricing", 
        "🏆 Competitive Analysis",
        "🧠 Psychological Pricing",
        "📈 Price Elasticity Analysis",
        "🎯 Dynamic Pricing",
        "📋 Pricing Strategy Recommendations"
    ]
    
    selected_section = st.sidebar.selectbox("Choose a section:", sections)
    
    if selected_section == "📊 Pricing Method Comparison":
        st.header("Pricing Method Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Input Parameters")
            cost = st.number_input("Product Cost ($)", value=50.0, min_value=0.0)
            markup = st.slider("Markup Percentage (%)", 0, 200, 40)
            customer_value = st.number_input("Customer Value ($)", value=200.0, min_value=0.0)
            value_capture = st.slider("Value Capture Rate (%)", 10, 50, 30)
            
            competitor_prices_input = st.text_area(
                "Competitor Prices (comma-separated)", 
                value="75, 85, 95, 105, 115"
            )
            
        with col2:
            st.subheader("Pricing Results")
            
            # Parse competitor prices
            try:
                competitor_prices = [float(x.strip()) for x in competitor_prices_input.split(',')]
            except:
                competitor_prices = [75, 85, 95, 105, 115]
            
            # Calculate prices using different methods
            cost_plus_price = pricing_strategy.cost_plus_pricing(cost, markup)
            value_based_price = pricing_strategy.value_based_pricing(customer_value, value_capture/100)
            competitive_price = pricing_strategy.competitive_pricing(competitor_prices, "match")
            psychological_price = pricing_strategy.psychological_pricing(cost_plus_price, "charm")
            
            results_df = pd.DataFrame({
                'Method': ['Cost-Plus', 'Value-Based', 'Competitive', 'Psychological'],
                'Price': [cost_plus_price, value_based_price, competitive_price, psychological_price],
                'Margin': [
                    markup,
                    ((value_based_price - cost) / value_based_price) * 100,
                    ((competitive_price - cost) / competitive_price) * 100,
                    ((psychological_price - cost) / psychological_price) * 100
                ]
            })
            
            st.dataframe(results_df, use_container_width=True)
            
            # Visualization
            fig = px.bar(
                results_df, 
                x='Method', 
                y='Price',
                title='Pricing Methods Comparison',
                color='Price',
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif selected_section == "💰 Cost-Plus Pricing":
        st.header("Cost-Plus Pricing Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Cost Structure")
            
            # Cost breakdown
            material_cost = st.number_input("Material Cost ($)", value=20.0, min_value=0.0)
            labor_cost = st.number_input("Labor Cost ($)", value=15.0, min_value=0.0)
            overhead_cost = st.number_input("Overhead Cost ($)", value=10.0, min_value=0.0)
            
            total_cost = material_cost + labor_cost + overhead_cost
            st.metric("Total Cost", f"${total_cost:.2f}")
            
            markup_percentage = st.slider("Markup Percentage (%)", 0, 200, 40)
            
        with col2:
            st.subheader("Pricing Analysis")
            
            final_price = pricing_strategy.cost_plus_pricing(total_cost, markup_percentage)
            profit = final_price - total_cost
            margin = (profit / final_price) * 100
            
            st.metric("Selling Price", f"${final_price:.2f}")
            st.metric("Profit", f"${profit:.2f}")
            st.metric("Profit Margin", f"{margin:.1f}%")
            
            # Cost breakdown chart
            cost_data = {
                'Component': ['Material', 'Labor', 'Overhead', 'Profit'],
                'Amount': [material_cost, labor_cost, overhead_cost, profit]
            }
            
            fig = px.pie(
                cost_data, 
                values='Amount', 
                names='Component',
                title='Cost and Profit Breakdown'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif selected_section == "💎 Value-Based Pricing":
        st.header("Value-Based Pricing Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Value Proposition")
            
            # Value drivers
            time_saved_hours = st.number_input("Time Saved (hours/month)", value=10.0, min_value=0.0)
            hourly_rate = st.number_input("Customer's Hourly Rate ($)", value=50.0, min_value=0.0)
            cost_reduction = st.number_input("Cost Reduction ($/month)", value=200.0, min_value=0.0)
            revenue_increase = st.number_input("Revenue Increase ($/month)", value=500.0, min_value=0.0)
            
            total_monthly_value = (time_saved_hours * hourly_rate) + cost_reduction + revenue_increase
            annual_value = total_monthly_value * 12
            
            st.metric("Monthly Value Created", f"${total_monthly_value:.2f}")
            st.metric("Annual Value Created", f"${annual_value:.2f}")
            
        with col2:
            st.subheader("Value-Based Pricing")
            
            value_capture_rate = st.slider("Value Capture Rate (%)", 10, 50, 30) / 100
            
            monthly_price = total_monthly_value * value_capture_rate
            annual_price = annual_value * value_capture_rate
            
            st.metric("Recommended Monthly Price", f"${monthly_price:.2f}")
            st.metric("Recommended Annual Price", f"${annual_price:.2f}")
            
            # ROI for customer
            customer_roi = ((total_monthly_value - monthly_price) / monthly_price) * 100
            st.metric("Customer ROI", f"{customer_roi:.1f}%")
            
            # Value breakdown
            value_components = {
                'Value Driver': ['Time Savings', 'Cost Reduction', 'Revenue Increase'],
                'Monthly Value': [time_saved_hours * hourly_rate, cost_reduction, revenue_increase]
            }
            
            fig = px.bar(
                value_components,
                x='Value Driver',
                y='Monthly Value',
                title='Value Driver Breakdown'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif selected_section == "🏆 Competitive Analysis":
        st.header("Competitive Pricing Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Competitor Data")
            
            your_price = st.number_input("Your Current Price ($)", value=99.0, min_value=0.0)
            
            # Competitor input
            num_competitors = st.number_input("Number of Competitors", value=5, min_value=1, max_value=10)
            
            competitor_data = {}
            for i in range(num_competitors):
                competitor_name = st.text_input(f"Competitor {i+1} Name", value=f"Competitor {i+1}")
                competitor_price = st.number_input(f"Competitor {i+1} Price ($)", value=100.0 + (i*10), min_value=0.0)
                competitor_data[competitor_name] = competitor_price
        
        with col2:
            st.subheader("Market Position Analysis")
            
            analysis = competitive_analyzer.market_position_analysis(your_price, competitor_data)
            
            st.metric("Market Average", f"${analysis['market_avg']:.2f}")
            st.metric("Your Price Percentile", f"{analysis['price_percentile']:.1f}%")
            st.metric("Premium vs Average", f"{analysis['premium_vs_avg']:.1f}%")
            
            # Competitive positioning chart
            competitor_df = pd.DataFrame(list(competitor_data.items()), columns=['Competitor', 'Price'])
            competitor_df = pd.concat([
                competitor_df,
                pd.DataFrame([['Your Product', your_price]], columns=['Competitor', 'Price'])
            ])
            
            fig = px.bar(
                competitor_df,
                x='Competitor',
                y='Price',
                title='Competitive Price Comparison',
                color='Price',
                color_continuous_scale='RdYlBu'
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    
    elif selected_section == "🧠 Psychological Pricing":
        st.header("Psychological Pricing Strategies")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Base Price Input")
            base_price = st.number_input("Base Price ($)", value=100.0, min_value=0.0)
            
            st.subheader("Psychological Strategies")
            
            # Different psychological pricing strategies
            charm_price = pricing_strategy.psychological_pricing(base_price, "charm")
            prestige_price = pricing_strategy.psychological_pricing(base_price, "prestige")
            bundle_price = pricing_strategy.psychological_pricing(base_price, "bundle")
            
            strategies_df = pd.DataFrame({
                'Strategy': ['Original', 'Charm Pricing', 'Prestige Pricing', 'Bundle Discount'],
                'Price': [base_price, charm_price, prestige_price, bundle_price],
                'Description': [
                    'Base price',
                    'Ends in .99 for perceived value',
                    'Round numbers for premium feel',
                    '10% bundle discount'
                ]
            })
            
            st.dataframe(strategies_df, use_container_width=True)
            
        with col2:
            st.subheader("Psychological Impact Analysis")
            
            # Price anchoring simulation
            anchor_high = base_price * 1.5
            anchor_medium = base_price * 1.2
            anchor_low = base_price * 0.8
            
            anchoring_data = {
                'Anchor Strategy': ['High Anchor', 'Medium Anchor', 'Low Anchor', 'No Anchor'],
                'Anchor Price': [anchor_high, anchor_medium, anchor_low, base_price],
                'Perceived Value': [anchor_high * 0.7, anchor_medium * 0.8, anchor_low * 1.1, base_price]
            }
            
            fig = px.scatter(
                anchoring_data,
                x='Anchor Price',
                y='Perceived Value',
                color='Anchor Strategy',
                title='Price Anchoring Effect on Perceived Value',
                size_max=15
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Price ending analysis
            st.subheader("Price Ending Psychology")
            
            endings_data = {
                'Price Ending': ['.00', '.49', '.99', '.95'],
                'Perceived Quality': [85, 70, 65, 68],
                'Purchase Intent': [60, 75, 85, 80]
            }
            
            endings_df = pd.DataFrame(endings_data)
            
            fig = px.bar(
                endings_df,
                x='Price Ending',
                y=['Perceived Quality', 'Purchase Intent'],
                title='Impact of Price Endings on Consumer Behavior',
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif selected_section == "📈 Price Elasticity Analysis":
        st.header("Price Elasticity & Sensitivity Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Historical Data Input")
            
            # Sample data for elasticity calculation
            st.markdown("Enter historical price and quantity data:")
            
            historical_data = st.text_area(
                "Price,Quantity (one pair per line)",
                value="100,1000\n110,950\n120,900\n130,850\n140,800"
            )
            
            base_price = st.number_input("Current Price ($)", value=120.0, min_value=0.0)
            price_range_min = st.number_input("Price Range Min ($)", value=80.0, min_value=0.0)
            price_range_max = st.number_input("Price Range Max ($)", value=200.0, min_value=0.0)
            
        with col2:
            st.subheader("Elasticity Analysis")
            
            # Parse historical data
            try:
                lines = historical_data.strip().split('\n')
                prices = []
                quantities = []
                for line in lines:
                    price, quantity = line.split(',')
                    prices.append(float(price.strip()))
                    quantities.append(float(quantity.strip()))
                
                elasticity = elasticity_analyzer.calculate_elasticity(prices, quantities)
                
                st.metric("Price Elasticity of Demand", f"{elasticity:.2f}")
                
                if abs(elasticity) > 1:
                    st.warning("Demand is elastic - price changes significantly affect quantity")
                else:
                    st.info("Demand is inelastic - price changes have moderate effect on quantity")
                
                # Sensitivity analysis
                sensitivity_df = elasticity_analyzer.sensitivity_analysis(
                    base_price, (price_range_min, price_range_max), elasticity
                )
                
                # Find optimal price (max revenue)
                optimal_idx = sensitivity_df['Revenue'].idxmax()
                optimal_price = sensitivity_df.loc[optimal_idx, 'Price']
                optimal_revenue = sensitivity_df.loc[optimal_idx, 'Revenue']
                
                st.metric("Optimal Price", f"${optimal_price:.2f}")
                st.metric("Max Revenue", f"${optimal_revenue:.2f}")
                
            except Exception as e:
                st.error("Please check your data format")
                elasticity = -1.5  # Default elasticity
                sensitivity_df = elasticity_analyzer.sensitivity_analysis(
                    base_price, (price_range_min, price_range_max), elasticity
                )
        
        # Visualization
        st.subheader("Price Sensitivity Curve")
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Revenue vs Price', 'Quantity vs Price'),
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Scatter(x=sensitivity_df['Price'], y=sensitivity_df['Revenue'], 
                      name='Revenue', line=dict(color='green')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=sensitivity_df['Price'], y=sensitivity_df['Quantity'], 
                      name='Quantity', line=dict(color='blue')),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="Price ($)", row=2, col=1)
        fig.update_yaxes(title_text="Revenue ($)", row=1, col=1)
        fig.update_yaxes(title_text="Quantity", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif selected_section == "🎯 Dynamic Pricing":
        st.header("Dynamic Pricing Strategy")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Market Conditions")
            
            base_price = st.number_input("Base Price ($)", value=100.0, min_value=0.0)
            
            # Demand factors
            st.subheader("Demand Factors")
            demand_level = st.select_slider(
                "Current Demand", 
                options=["Very Low", "Low", "Normal", "High", "Very High"],
                value="Normal"
            )
            
            seasonal_factor = st.slider("Seasonal Adjustment", 0.5, 2.0, 1.0, 0.1)
            event_multiplier = st.slider("Special Event Multiplier", 0.8, 3.0, 1.0, 0.1)
            
            # Supply factors
            st.subheader("Supply Factors")
            inventory_level = st.select_slider(
                "Inventory Level",
                options=["Very Low", "Low", "Normal", "High", "Very High"],
                value="Normal"
            )
            
            competition_intensity = st.slider("Competition Intensity", 0.5, 2.0, 1.0, 0.1)
            
        with col2:
            st.subheader("Dynamic Pricing Results")
            
            # Convert categorical inputs to numerical factors
            demand_mapping = {"Very Low": 0.6, "Low": 0.8, "Normal": 1.0, "High": 1.3, "Very High": 1.6}
            supply_mapping = {"Very Low": 1.5, "Low": 1.2, "Normal": 1.0, "High": 0.8, "Very High": 0.6}
            
            demand_factor = demand_mapping[demand_level] * seasonal_factor * event_multiplier
            supply_factor = supply_mapping[inventory_level] * competition_intensity
            
            dynamic_price = pricing_strategy.dynamic_pricing(base_price, demand_factor, supply_factor)
            price_change = ((dynamic_price - base_price) / base_price) * 100
            
            st.metric("Dynamic Price", f"${dynamic_price:.2f}")
            st.metric("Price Change", f"{price_change:+.1f}%")
            
            # Price factors breakdown
            factors_data = {
                'Factor': ['Base Price', 'Demand Impact', 'Supply Impact', 'Final Price'],
                'Value': [base_price, base_price * demand_factor, base_price * supply_factor, dynamic_price]
            }
            
            fig = px.waterfall(
                x=factors_data['Factor'],
                y=[base_price, (demand_factor-1)*base_price, (supply_factor-1)*base_price, 0],
                title='Dynamic Pricing Waterfall'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Time-based pricing simulation
            st.subheader("24-Hour Pricing Simulation")
            
            hours = np.arange(24)
            hourly_demand = 1 + 0.3 * np.sin((hours - 6) * np.pi / 12)  # Peak around 6 PM
            hourly_prices = [pricing_strategy.dynamic_pricing(base_price, d, 1.0) for d in hourly_demand]
            
            hourly_df = pd.DataFrame({
                'Hour': hours,
                'Demand_Factor': hourly_demand,
                'Price': hourly_prices
            })
            
            fig = px.line(hourly_df, x='Hour', y='Price', title='24-Hour Dynamic Pricing')
            st.plotly_chart(fig, use_container_width=True)
    
    elif selected_section == "📋 Pricing Strategy Recommendations":
        st.header("Pricing Strategy Recommendations")
        
        # Business context inputs
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Business Context")
            
            business_stage = st.selectbox(
                "Business Stage",
                ["Startup", "Growth", "Mature", "Declining"]
            )
            
            market_position = st.selectbox(
                "Market Position",
                ["Market Leader", "Challenger", "Follower", "Niche Player"]
            )
            
            product_type = st.selectbox(
                "Product Type",
                ["Commodity", "Differentiated", "Premium", "Luxury"]
            )
            
            target_customer = st.selectbox(
                "Target Customer",
                ["Price Sensitive", "Value Conscious", "Premium Buyers", "Mixed"]
            )
            
        with col2:
            st.subheader("Strategic Objectives")
            
            primary_objective = st.selectbox(
                "Primary Objective",
                ["Maximize Revenue", "Maximize Profit", "Market Share Growth", "Premium Positioning"]
            )
            
            time_horizon = st.selectbox(
                "Time Horizon",
                ["Short-term (< 1 year)", "Medium-term (1-3 years)", "Long-term (> 3 years)"]
            )
            
            risk_tolerance = st.selectbox(
                "Risk Tolerance",
                ["Conservative", "Moderate", "Aggressive"]
            )
        
        # Generate recommendations
        st.subheader("Recommended Pricing Strategy")
        
        recommendations = generate_pricing_recommendations(
            business_stage, market_position, product_type, target_customer,
            primary_objective, time_horizon, risk_tolerance
        )
        
        for i, rec in enumerate(recommendations, 1):
            st.write(f"**{i}. {rec['strategy']}**")
            st.write(f"   {rec['description']}")
            st.write(f"   *{rec['rationale']}*")
            st.write("")
        
        # Implementation roadmap
        st.subheader("Implementation Roadmap")
        
        roadmap = generate_implementation_roadmap(primary_objective, time_horizon)
        
        for phase in roadmap:
            st.write(f"**{phase['phase']} ({phase['timeline']})**")
            for task in phase['tasks']:
                st.write(f"   • {task}")
            st.write("")

def generate_pricing_recommendations(business_stage, market_position, product_type, 
                                   target_customer, primary_objective, time_horizon, risk_tolerance):
    """Generate personalized pricing recommendations based on business context"""
    
    recommendations = []
    
    # Base strategy selection
    if business_stage == "Startup":
        if primary_objective == "Market Share Growth":
            recommendations.append({
                "strategy": "Penetration Pricing",
                "description": "Start with low prices to gain market share quickly",
                "rationale": "As a startup, gaining traction and customer base is crucial"
            })
        else:
            recommendations.append({
                "strategy": "Value-Based Pricing",
                "description": "Price based on customer value perception",
                "rationale": "Focus on demonstrating value to early adopters"
            })
    
    elif market_position == "Market Leader":
        recommendations.append({
            "strategy": "Premium Pricing",
            "description": "Maintain price leadership position",
            "rationale": "Leverage market position to command premium prices"
        })
    
    # Product type considerations
    if product_type == "Commodity":
        recommendations.append({
            "strategy": "Competitive Pricing",
            "description": "Match or slightly undercut competitor prices",
            "rationale": "In commodity markets, price competition is intense"
        })
    elif product_type == "Premium" or product_type == "Luxury":
        recommendations.append({
            "strategy": "Prestige Pricing",
            "description": "Use high prices to signal quality and exclusivity",
            "rationale": "Premium customers expect to pay more for perceived quality"
        })
    
    # Dynamic pricing for growth stage
    if business_stage == "Growth" and risk_tolerance == "Aggressive":
        recommendations.append({
            "strategy": "Dynamic Pricing",
            "description": "Implement real-time pricing based on demand and supply",
            "rationale": "Maximize revenue during growth phase with flexible pricing"
        })
    
    # Psychological pricing for consumer products
    if target_customer in ["Price Sensitive", "Value Conscious"]:
        recommendations.append({
            "strategy": "Psychological Pricing",
            "description": "Use charm pricing and anchoring techniques",
            "rationale": "Price-conscious customers respond well to psychological cues"
        })
    
    return recommendations

def generate_implementation_roadmap(primary_objective, time_horizon):
    """Generate implementation roadmap based on objectives and timeline"""
    
    if time_horizon == "Short-term (< 1 year)":
        return [
            {
                "phase": "Phase 1: Quick Wins",
                "timeline": "0-3 months",
                "tasks": [
                    "Implement psychological pricing tactics",
                    "Analyze competitor pricing weekly",
                    "A/B test different price points",
                    "Optimize pricing page design"
                ]
            },
            {
                "phase": "Phase 2: Data Collection",
                "timeline": "3-6 months", 
                "tasks": [
                    "Gather customer value feedback",
                    "Track price elasticity metrics",
                    "Build pricing analytics dashboard",
                    "Segment customers by price sensitivity"
                ]
            },
            {
                "phase": "Phase 3: Optimization",
                "timeline": "6-12 months",
                "tasks": [
                    "Implement dynamic pricing for key products",
                    "Launch tiered pricing strategy",
                    "Optimize bundle pricing",
                    "Measure and report ROI"
                ]
            }
        ]
    else:
        return [
            {
                "phase": "Phase 1: Foundation",
                "timeline": "0-6 months",
                "tasks": [
                    "Conduct comprehensive market research",
                    "Build pricing team and capabilities",
                    "Implement pricing analytics infrastructure",
                    "Develop pricing governance framework"
                ]
            },
            {
                "phase": "Phase 2: Strategic Implementation", 
                "timeline": "6-18 months",
                "tasks": [
                    "Launch value-based pricing methodology",
                    "Implement sophisticated segmentation",
                    "Deploy AI-powered dynamic pricing",
                    "Build competitive intelligence system"
                ]
            },
            {
                "phase": "Phase 3: Advanced Optimization",
                "timeline": "18+ months",
                "tasks": [
                    "Implement predictive pricing models",
                    "Launch personalized pricing",
                    "Optimize across full customer lifecycle",
                    "Continuous innovation and testing"
                ]
            }
        ]

if __name__ == "__main__":
    main()