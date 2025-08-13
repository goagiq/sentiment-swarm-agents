"""
Scenario Comparison Interface

Provides what-if scenario analysis capabilities including:
- Scenario creation and management
- Impact analysis visualization
- Risk assessment interface
- Decision support tools
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
from typing import Dict, Any, List

# API configuration
API_BASE_URL = "http://localhost:8003"


def create_scenario(scenario_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new scenario via API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/scenario/create",
            json=scenario_data,
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Request Error: {str(e)}"}


def compare_scenarios(scenario_ids: List[str]) -> Dict[str, Any]:
    """Compare multiple scenarios via API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/scenario/compare",
            json={"scenario_ids": scenario_ids},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Request Error: {str(e)}"}


def get_scenario_list() -> Dict[str, Any]:
    """Get list of available scenarios via API."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/scenario/list",
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Request Error: {str(e)}"}


def create_scenario_comparison_chart(scenarios: List[Dict]) -> go.Figure:
    """Create a scenario comparison chart."""
    fig = go.Figure()
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']
    
    for i, scenario in enumerate(scenarios):
        color = colors[i % len(colors)]
        
        # Add main scenario line
        fig.add_trace(go.Scatter(
            x=scenario['timeline'],
            y=scenario['values'],
            mode='lines+markers',
            name=scenario['name'],
            line=dict(color=color, width=3),
            marker=dict(size=8)
        ))
        
        # Add confidence intervals if available
        if 'lower_bound' in scenario and 'upper_bound' in scenario:
            fig.add_trace(go.Scatter(
                x=scenario['timeline'],
                y=scenario['upper_bound'],
                mode='lines',
                name=f"{scenario['name']} Upper",
                line=dict(color=color, width=1, dash='dot'),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=scenario['timeline'],
                y=scenario['lower_bound'],
                mode='lines',
                fill='tonexty',
                name=f"{scenario['name']} CI",
                line=dict(color=color, width=1, dash='dot'),
                fillcolor=f'rgba({color},0.1)',
                showlegend=False
            ))
    
    fig.update_layout(
        title="Scenario Comparison",
        xaxis_title="Time Period",
        yaxis_title="Value",
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def create_impact_radar_chart(impacts: Dict[str, float]) -> go.Figure:
    """Create a radar chart for impact analysis."""
    categories = list(impacts.keys())
    values = list(impacts.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Impact Score',
        line_color='blue'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )),
        showlegend=False,
        title="Impact Analysis"
    )
    
    return fig


def create_risk_matrix(risks: List[Dict]) -> go.Figure:
    """Create a risk matrix visualization."""
    # Prepare data
    likelihood = [risk['likelihood'] for risk in risks]
    impact = [risk['impact'] for risk in risks]
    risk_names = [risk['name'] for risk in risks]
    risk_scores = [risk['score'] for risk in risks]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=likelihood,
        y=impact,
        mode='markers+text',
        text=risk_names,
        textposition="top center",
        marker=dict(
            size=risk_scores,
            color=risk_scores,
            colorscale='RdYlGn_r',
            showscale=True,
            colorbar=dict(title="Risk Score")
        ),
        name='Risks'
    ))
    
    fig.update_layout(
        title="Risk Matrix",
        xaxis_title="Likelihood",
        yaxis_title="Impact",
        xaxis=dict(range=[0, 10]),
        yaxis=dict(range=[0, 10])
    )
    
    return fig


def display_scenario_creation():
    """Display scenario creation interface."""
    st.markdown("## 🎯 Create New Scenario")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Scenario basic info
        scenario_name = st.text_input("Scenario Name:", placeholder="e.g., Market Expansion")
        description = st.text_area("Description:", placeholder="Describe the scenario...")
        
        # Parameters
        st.markdown("### 📊 Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            market_growth = st.slider("Market Growth Rate (%):", 0.0, 20.0, 5.0, 0.5)
            cost_reduction = st.slider("Cost Reduction (%):", 0.0, 30.0, 10.0, 1.0)
            time_horizon = st.selectbox("Time Horizon:", [6, 12, 18, 24, 36], index=1)
        
        with col2:
            investment_level = st.slider("Investment Level ($M):", 0.0, 100.0, 25.0, 5.0)
            risk_tolerance = st.selectbox("Risk Tolerance:", ["Low", "Medium", "High"])
        
        # Assumptions
        st.markdown("### 📝 Key Assumptions")
        assumptions = []
        
        assumption1 = st.text_input("Assumption 1:", placeholder="e.g., Stable market conditions")
        if assumption1:
            assumptions.append(assumption1)
        
        assumption2 = st.text_input("Assumption 2:", placeholder="e.g., No major competitors")
        if assumption2:
            assumptions.append(assumption2)
        
        assumption3 = st.text_input("Assumption 3:", placeholder="e.g., Regulatory approval")
        if assumption3:
            assumptions.append(assumption3)
        
        if st.button("Create Scenario", type="primary"):
            if scenario_name and description:
                scenario_data = {
                    "name": scenario_name,
                    "description": description,
                    "parameters": {
                        "market_growth": market_growth,
                        "cost_reduction": cost_reduction,
                        "investment_level": investment_level,
                        "risk_tolerance": risk_tolerance,
                        "time_horizon": time_horizon
                    },
                    "assumptions": assumptions
                }
                
                with st.spinner("Creating scenario..."):
                    result = create_scenario(scenario_data)
                    
                    if "error" not in result:
                        st.success("Scenario created successfully!")
                        st.json(result)
                    else:
                        st.error(f"Error creating scenario: {result['error']}")
            else:
                st.warning("Please provide scenario name and description.")
    
    with col2:
        st.markdown("### 📋 Scenario Template")
        st.markdown("""
        **Template Parameters:**
        - **Market Growth**: Expected market expansion
        - **Cost Reduction**: Operational efficiency gains
        - **Investment Level**: Required capital investment
        - **Risk Tolerance**: Acceptable risk level
        - **Time Horizon**: Planning period
        """)
        
        st.markdown("### 🎲 Scenario Types")
        st.markdown("""
        **Common Scenarios:**
        - **Baseline**: Current trajectory
        - **Optimistic**: Best-case outcomes
        - **Pessimistic**: Worst-case outcomes
        - **Market Expansion**: Growth focus
        - **Cost Reduction**: Efficiency focus
        """)


def display_scenario_comparison():
    """Display scenario comparison interface."""
    st.markdown("## 📊 Scenario Comparison")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Get available scenarios
        scenarios_result = get_scenario_list()
        
        if "error" not in scenarios_result:
            scenarios = scenarios_result.get("scenarios", [])
            
            if scenarios:
                # Scenario selection
                selected_scenarios = st.multiselect(
                    "Select Scenarios to Compare:",
                    options=[s['name'] for s in scenarios],
                    default=[s['name'] for s in scenarios[:3]]
                )
                
                if len(selected_scenarios) >= 2:
                    if st.button("Compare Scenarios", type="primary"):
                        with st.spinner("Comparing scenarios..."):
                            # Get scenario IDs
                            scenario_ids = [
                                s['id'] for s in scenarios 
                                if s['name'] in selected_scenarios
                            ]
                            
                            comparison_result = compare_scenarios(scenario_ids)
                            
                            if "error" not in comparison_result:
                                # Create comparison chart
                                scenarios_data = comparison_result.get("scenarios", [])
                                if scenarios_data:
                                    fig = create_scenario_comparison_chart(scenarios_data)
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # Display comparison metrics
                                metrics = comparison_result.get("comparison_metrics", {})
                                if metrics:
                                    st.markdown("### 📈 Comparison Metrics")
                                    
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        st.metric("Best NPV", f"${metrics.get('best_npv', 0):.2f}M")
                                    
                                    with col2:
                                        st.metric("Worst NPV", f"${metrics.get('worst_npv', 0):.2f}M")
                                    
                                    with col3:
                                        st.metric("Range", f"${metrics.get('range', 0):.2f}M")
                                
                                # Display recommendations
                                recommendations = comparison_result.get("recommendations", [])
                                if recommendations:
                                    st.markdown("### 💡 Recommendations")
                                    for rec in recommendations:
                                        st.info(rec)
                            else:
                                st.error(f"Error comparing scenarios: {comparison_result['error']}")
                else:
                    st.warning("Please select at least 2 scenarios to compare.")
            else:
                st.info("No scenarios available. Create some scenarios first.")
        else:
            st.error(f"Error loading scenarios: {scenarios_result['error']}")
    
    with col2:
        st.markdown("### 📊 Comparison Metrics")
        st.markdown("""
        **Key Metrics:**
        - **NPV**: Net Present Value
        - **IRR**: Internal Rate of Return
        - **Payback Period**: Time to break even
        - **Risk Score**: Overall risk assessment
        - **ROI**: Return on Investment
        """)
        
        st.markdown("### 🎯 Decision Criteria")
        st.markdown("""
        **Evaluation Factors:**
        - **Financial Impact**: Revenue and cost effects
        - **Strategic Fit**: Alignment with goals
        - **Risk Level**: Uncertainty assessment
        - **Implementation**: Feasibility and timeline
        - **Resources**: Required capabilities
        """)


def display_impact_analysis():
    """Display impact analysis interface."""
    st.markdown("## 📈 Impact Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Impact categories
        st.markdown("### 📊 Impact Assessment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            financial_impact = st.slider("Financial Impact:", 1, 10, 7)
            operational_impact = st.slider("Operational Impact:", 1, 10, 6)
            strategic_impact = st.slider("Strategic Impact:", 1, 10, 8)
        
        with col2:
            market_impact = st.slider("Market Impact:", 1, 10, 7)
            technology_impact = st.slider("Technology Impact:", 1, 10, 5)
            regulatory_impact = st.slider("Regulatory Impact:", 1, 10, 4)
        
        if st.button("Analyze Impact", type="primary"):
            # Create impact data
            impacts = {
                "Financial": financial_impact,
                "Operational": operational_impact,
                "Strategic": strategic_impact,
                "Market": market_impact,
                "Technology": technology_impact,
                "Regulatory": regulatory_impact
            }
            
            # Create radar chart
            fig = create_impact_radar_chart(impacts)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display insights
            st.markdown("### 💡 Impact Insights")
            
            max_impact = max(impacts.items(), key=lambda x: x[1])
            min_impact = min(impacts.items(), key=lambda x: x[1])
            
            st.success(f"**Highest Impact**: {max_impact[0]} ({max_impact[1]}/10)")
            st.warning(f"**Lowest Impact**: {min_impact[0]} ({min_impact[1]}/10)")
            
            # Recommendations
            if max_impact[1] >= 8:
                st.info(f"Focus on {max_impact[0]} impact as it's critical for success.")
            
            if min_impact[1] <= 3:
                st.info(f"Consider if {min_impact[0]} impact needs more attention.")
    
    with col2:
        st.markdown("### 📋 Impact Categories")
        st.markdown("""
        **Impact Types:**
        - **Financial**: Revenue, costs, profitability
        - **Operational**: Processes, efficiency, capacity
        - **Strategic**: Long-term goals, positioning
        - **Market**: Customer, competition, demand
        - **Technology**: Systems, infrastructure, innovation
        - **Regulatory**: Compliance, legal, policy
        """)
        
        st.markdown("### 📊 Scoring Guide")
        st.markdown("""
        **Impact Scale:**
        - **1-3**: Low impact
        - **4-6**: Medium impact
        - **7-8**: High impact
        - **9-10**: Critical impact
        """)


def display_risk_assessment():
    """Display risk assessment interface."""
    st.markdown("## ⚠️ Risk Assessment")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Risk identification
        st.markdown("### 🎯 Risk Identification")
        
        risks = []
        
        # Risk 1
        with st.expander("Risk 1: Market Risk"):
            risk1_name = st.text_input("Risk Name:", value="Market Volatility", key="risk1")
            risk1_likelihood = st.slider("Likelihood (1-10):", 1, 10, 6, key="like1")
            risk1_impact = st.slider("Impact (1-10):", 1, 10, 7, key="imp1")
            risk1_score = risk1_likelihood * risk1_impact
            
            if risk1_name:
                risks.append({
                    "name": risk1_name,
                    "likelihood": risk1_likelihood,
                    "impact": risk1_impact,
                    "score": risk1_score
                })
        
        # Risk 2
        with st.expander("Risk 2: Technology Risk"):
            risk2_name = st.text_input("Risk Name:", value="Technology Failure", key="risk2")
            risk2_likelihood = st.slider("Likelihood (1-10):", 1, 10, 4, key="like2")
            risk2_impact = st.slider("Impact (1-10):", 1, 10, 8, key="imp2")
            risk2_score = risk2_likelihood * risk2_impact
            
            if risk2_name:
                risks.append({
                    "name": risk2_name,
                    "likelihood": risk2_likelihood,
                    "impact": risk2_impact,
                    "score": risk2_score
                })
        
        # Risk 3
        with st.expander("Risk 3: Operational Risk"):
            risk3_name = st.text_input("Risk Name:", value="Resource Shortage", key="risk3")
            risk3_likelihood = st.slider("Likelihood (1-10):", 1, 10, 5, key="like3")
            risk3_impact = st.slider("Impact (1-10):", 1, 10, 6, key="imp3")
            risk3_score = risk3_likelihood * risk3_impact
            
            if risk3_name:
                risks.append({
                    "name": risk3_name,
                    "likelihood": risk3_likelihood,
                    "impact": risk3_impact,
                    "score": risk3_score
                })
        
        if risks and st.button("Generate Risk Matrix", type="primary"):
            # Create risk matrix
            fig = create_risk_matrix(risks)
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk summary
            st.markdown("### 📊 Risk Summary")
            
            high_risks = [r for r in risks if r['score'] >= 40]
            medium_risks = [r for r in risks if 20 <= r['score'] < 40]
            low_risks = [r for r in risks if r['score'] < 20]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("High Risks", len(high_risks))
            
            with col2:
                st.metric("Medium Risks", len(medium_risks))
            
            with col3:
                st.metric("Low Risks", len(low_risks))
            
            # Risk recommendations
            if high_risks:
                st.warning("**High Priority Risks:**")
                for risk in high_risks:
                    st.write(f"• {risk['name']} (Score: {risk['score']})")
    
    with col2:
        st.markdown("### ⚠️ Risk Categories")
        st.markdown("""
        **Risk Types:**
        - **Market Risk**: Competition, demand, pricing
        - **Technology Risk**: Systems, infrastructure, security
        - **Operational Risk**: Processes, resources, capacity
        - **Financial Risk**: Funding, costs, revenue
        - **Regulatory Risk**: Compliance, legal, policy
        - **Strategic Risk**: Goals, positioning, execution
        """)
        
        st.markdown("### 📊 Risk Matrix")
        st.markdown("""
        **Risk Levels:**
        - **Low (1-20)**: Monitor
        - **Medium (21-40)**: Mitigate
        - **High (41-60)**: Act immediately
        - **Critical (61-100)**: Emergency response
        """)


def main():
    """Main scenario comparison interface."""
    st.set_page_config(
        page_title="Scenario Comparison",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown('<h1 class="main-header">🎯 Scenario Comparison</h1>', unsafe_allow_html=True)
    
    # Navigation tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Create", 
        "📊 Compare", 
        "📈 Impact", 
        "⚠️ Risk"
    ])
    
    with tab1:
        display_scenario_creation()
    
    with tab2:
        display_scenario_comparison()
    
    with tab3:
        display_impact_analysis()
    
    with tab4:
        display_risk_assessment()


if __name__ == "__main__":
    main()
