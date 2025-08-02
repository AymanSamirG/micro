#!/usr/bin/env python3
"""
Pricing Strategy Examples
Demonstrates practical usage of different pricing methodologies
"""

import pandas as pd
import numpy as np
from pricing_strategy import PricingStrategy, PriceElasticityAnalyzer, CompetitiveAnalyzer

def demo_cost_plus_pricing():
    """Demonstrate cost-plus pricing methodology"""
    print("=" * 60)
    print("COST-PLUS PRICING EXAMPLE")
    print("=" * 60)
    
    pricing = PricingStrategy()
    
    # Example: Manufacturing a smartphone case
    material_cost = 5.00
    labor_cost = 3.00
    overhead_cost = 2.00
    total_cost = material_cost + labor_cost + overhead_cost
    
    markup_percentages = [25, 50, 75, 100]
    
    print(f"Product: Smartphone Case")
    print(f"Material Cost: ${material_cost:.2f}")
    print(f"Labor Cost: ${labor_cost:.2f}")
    print(f"Overhead Cost: ${overhead_cost:.2f}")
    print(f"Total Cost: ${total_cost:.2f}")
    print()
    
    for markup in markup_percentages:
        price = pricing.cost_plus_pricing(total_cost, markup)
        profit = price - total_cost
        margin = (profit / price) * 100
        
        print(f"Markup: {markup}% -> Price: ${price:.2f}, Profit: ${profit:.2f}, Margin: {margin:.1f}%")
    
    print()

def demo_value_based_pricing():
    """Demonstrate value-based pricing methodology"""
    print("=" * 60)
    print("VALUE-BASED PRICING EXAMPLE")
    print("=" * 60)
    
    pricing = PricingStrategy()
    
    # Example: Project management software
    print("Product: Project Management Software (Monthly)")
    
    # Value drivers
    time_saved_hours = 20  # hours per month
    hourly_rate = 75  # customer's hourly rate
    cost_reduction = 500  # reduced external tool costs
    revenue_increase = 1000  # improved efficiency leads to more revenue
    
    total_value = (time_saved_hours * hourly_rate) + cost_reduction + revenue_increase
    
    print(f"Value Drivers:")
    print(f"  Time Saved: {time_saved_hours} hours/month × ${hourly_rate}/hour = ${time_saved_hours * hourly_rate}")
    print(f"  Cost Reduction: ${cost_reduction}/month")
    print(f"  Revenue Increase: ${revenue_increase}/month")
    print(f"  Total Monthly Value: ${total_value}")
    print()
    
    capture_rates = [0.1, 0.2, 0.3, 0.4]
    
    for rate in capture_rates:
        price = pricing.value_based_pricing(total_value, rate)
        customer_roi = ((total_value - price) / price) * 100
        
        print(f"Value Capture {rate*100:.0f}% -> Price: ${price:.2f}, Customer ROI: {customer_roi:.1f}%")
    
    print()

def demo_competitive_pricing():
    """Demonstrate competitive pricing methodology"""
    print("=" * 60)
    print("COMPETITIVE PRICING EXAMPLE")
    print("=" * 60)
    
    pricing = PricingStrategy()
    analyzer = CompetitiveAnalyzer()
    
    # Example: Cloud hosting service
    print("Product: Cloud Hosting Service")
    
    competitor_data = {
        "AWS": 120,
        "Google Cloud": 115,
        "Azure": 125,
        "DigitalOcean": 95,
        "Linode": 100
    }
    
    your_price = 110
    
    print("Competitor Pricing:")
    for name, price in competitor_data.items():
        print(f"  {name}: ${price}")
    print(f"  Your Current Price: ${your_price}")
    print()
    
    # Competitive strategies
    strategies = ["discount", "match", "premium"]
    
    for strategy in strategies:
        competitive_price = pricing.competitive_pricing(list(competitor_data.values()), strategy)
        print(f"{strategy.title()} Strategy: ${competitive_price:.2f}")
    
    print()
    
    # Market position analysis
    analysis = analyzer.market_position_analysis(your_price, competitor_data)
    
    print("Market Position Analysis:")
    print(f"  Market Average: ${analysis['market_avg']:.2f}")
    print(f"  Your Price Percentile: {analysis['price_percentile']:.1f}%")
    print(f"  Premium vs Average: {analysis['premium_vs_avg']:.1f}%")
    print(f"  Closest Competitor: {analysis['closest_competitor'][0]} (${analysis['closest_competitor'][1]})")
    
    print()

def demo_psychological_pricing():
    """Demonstrate psychological pricing strategies"""
    print("=" * 60)
    print("PSYCHOLOGICAL PRICING EXAMPLE")
    print("=" * 60)
    
    pricing = PricingStrategy()
    
    # Example: Consumer electronics
    base_prices = [99.50, 149.75, 249.25, 499.80]
    
    print("Product: Consumer Electronics")
    print("Psychological Pricing Strategies:")
    print()
    
    for base_price in base_prices:
        charm_price = pricing.psychological_pricing(base_price, "charm")
        prestige_price = pricing.psychological_pricing(base_price, "prestige")
        bundle_price = pricing.psychological_pricing(base_price, "bundle")
        
        print(f"Base Price: ${base_price:.2f}")
        print(f"  Charm Pricing (ends in .99): ${charm_price:.2f}")
        print(f"  Prestige Pricing (round numbers): ${prestige_price:.2f}")
        print(f"  Bundle Discount (10% off): ${bundle_price:.2f}")
        print()

def demo_price_elasticity():
    """Demonstrate price elasticity analysis"""
    print("=" * 60)
    print("PRICE ELASTICITY ANALYSIS EXAMPLE")
    print("=" * 60)
    
    analyzer = PriceElasticityAnalyzer()
    
    # Example: SaaS subscription service
    print("Product: SaaS Subscription Service")
    
    # Historical data: price changes and corresponding quantity changes
    historical_prices = [50, 55, 60, 65, 70, 75]
    historical_quantities = [1000, 950, 900, 830, 750, 680]
    
    print("Historical Data:")
    for price, quantity in zip(historical_prices, historical_quantities):
        print(f"  Price: ${price} -> Quantity: {quantity} subscribers")
    print()
    
    # Calculate elasticity
    elasticity = analyzer.calculate_elasticity(historical_prices, historical_quantities)
    
    print(f"Price Elasticity of Demand: {elasticity:.2f}")
    
    if abs(elasticity) > 1:
        print("Interpretation: Demand is ELASTIC - customers are price-sensitive")
    else:
        print("Interpretation: Demand is INELASTIC - customers are less price-sensitive")
    
    print()
    
    # Sensitivity analysis
    current_price = 65
    price_range = (40, 100)
    
    sensitivity_df = analyzer.sensitivity_analysis(current_price, price_range, elasticity)
    
    # Find optimal price
    optimal_idx = sensitivity_df['Revenue'].idxmax()
    optimal_price = sensitivity_df.loc[optimal_idx, 'Price']
    optimal_revenue = sensitivity_df.loc[optimal_idx, 'Revenue']
    
    print(f"Current Price: ${current_price}")
    print(f"Optimal Price for Max Revenue: ${optimal_price:.2f}")
    print(f"Maximum Revenue: ${optimal_revenue:.2f}")
    
    print()

def demo_dynamic_pricing():
    """Demonstrate dynamic pricing methodology"""
    print("=" * 60)
    print("DYNAMIC PRICING EXAMPLE")
    print("=" * 60)
    
    pricing = PricingStrategy()
    
    # Example: Ride-sharing service
    print("Product: Ride-sharing Service")
    
    base_price = 15.00
    
    scenarios = [
        {"name": "Normal Conditions", "demand": 1.0, "supply": 1.0},
        {"name": "Rush Hour", "demand": 1.5, "supply": 0.8},
        {"name": "Weekend Evening", "demand": 1.8, "supply": 0.6},
        {"name": "Holiday Season", "demand": 2.0, "supply": 0.7},
        {"name": "Low Demand Period", "demand": 0.6, "supply": 1.4}
    ]
    
    print(f"Base Price: ${base_price:.2f}")
    print()
    
    for scenario in scenarios:
        dynamic_price = pricing.dynamic_pricing(
            base_price, 
            scenario["demand"], 
            scenario["supply"]
        )
        
        price_change = ((dynamic_price - base_price) / base_price) * 100
        
        print(f"{scenario['name']}:")
        print(f"  Demand Factor: {scenario['demand']:.1f}")
        print(f"  Supply Factor: {scenario['supply']:.1f}")
        print(f"  Dynamic Price: ${dynamic_price:.2f} ({price_change:+.1f}%)")
        print()

def demo_pricing_comparison():
    """Compare all pricing methods for the same product"""
    print("=" * 60)
    print("PRICING METHODS COMPARISON")
    print("=" * 60)
    
    pricing = PricingStrategy()
    
    # Example: Business consulting service
    print("Product: Business Consulting Service (Monthly)")
    
    # Cost-plus inputs
    total_cost = 2000
    markup = 50
    
    # Value-based inputs
    customer_value = 10000
    value_capture_rate = 0.25
    
    # Competitive inputs
    competitor_prices = [3500, 4000, 4500, 5000, 5500]
    
    # Calculate prices
    cost_plus_price = pricing.cost_plus_pricing(total_cost, markup)
    value_based_price = pricing.value_based_pricing(customer_value, value_capture_rate)
    competitive_price = pricing.competitive_pricing(competitor_prices, "match")
    psychological_price = pricing.psychological_pricing(cost_plus_price, "charm")
    
    # Display results
    methods = [
        ("Cost-Plus", cost_plus_price),
        ("Value-Based", value_based_price),
        ("Competitive", competitive_price),
        ("Psychological", psychological_price)
    ]
    
    print("Pricing Method Comparison:")
    print("-" * 40)
    
    for method, price in methods:
        margin = ((price - total_cost) / price) * 100 if price > 0 else 0
        print(f"{method:<15}: ${price:>8.2f} (Margin: {margin:>5.1f}%)")
    
    print()
    
    # Recommendations
    print("Recommendations:")
    print(f"• Cost-Plus suggests conservative pricing at ${cost_plus_price:.2f}")
    print(f"• Value-Based suggests premium pricing at ${value_based_price:.2f}")
    print(f"• Competitive analysis suggests market rate at ${competitive_price:.2f}")
    print(f"• Psychological pricing optimizes perception at ${psychological_price:.2f}")
    
    print()

def main():
    """Run all pricing strategy demonstrations"""
    print("COMPREHENSIVE PRICING STRATEGY EXAMPLES")
    print("=" * 60)
    print("This demo showcases various pricing methodologies")
    print("and their practical applications.")
    print()
    
    demo_cost_plus_pricing()
    demo_value_based_pricing()
    demo_competitive_pricing()
    demo_psychological_pricing()
    demo_price_elasticity()
    demo_dynamic_pricing()
    demo_pricing_comparison()
    
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Key Takeaways:")
    print("• Cost-plus pricing ensures profitability but may miss market opportunities")
    print("• Value-based pricing maximizes revenue but requires clear value demonstration")
    print("• Competitive pricing ensures market competitiveness but may lead to price wars")
    print("• Psychological pricing influences customer perception and purchase behavior")
    print("• Price elasticity analysis helps optimize revenue through data-driven decisions")
    print("• Dynamic pricing adapts to market conditions for real-time optimization")
    print()
    print("Choose the method that best fits your:")
    print("• Business model and objectives")
    print("• Market position and competition")
    print("• Customer characteristics and behavior")
    print("• Product type and differentiation")
    print()
    print("For interactive analysis, run: streamlit run pricing_strategy.py")

if __name__ == "__main__":
    main()