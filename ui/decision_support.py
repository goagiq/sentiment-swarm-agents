"""
Decision Support Interface

Provides AI-powered decision support capabilities including:
- AI recommendations
- Decision assistance
- Action prioritization
- Implementation planning
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


def get_recommendations(context: Dict[str, Any]) -> Dict[str, Any]:
    """Get AI recommendations from API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/decision/recommendations",
            json=context,
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Request Error: {str(e)}"}


def prioritize_actions(actions: List[Dict]) -> Dict[str, Any]:
    """Prioritize actions via API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/decision/prioritize",
            json={"actions": actions},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Request Error: {str(e)}"}


def create_implementation_plan(goal: str, constraints: Dict[str, Any]) -> Dict[str, Any]:
    """Create implementation plan via API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/decision/plan",
            json={"goal": goal, "constraints": constraints},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Request Error: {str(e)}"}


def create_recommendation_chart(recommendations: List[Dict]) -> go.Figure:
    """Create a recommendation visualization chart."""
    if not recommendations:
        fig = go.Figure()
        fig.add_annotation(
            text="No recommendations available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(title="Recommendations")
        return fig
    
    # Extract data for visualization
    categories = []
    scores = []
    impacts = []
    
    for rec in recommendations:
        categories.append(rec.get('category', 'Unknown'))
        scores.append(rec.get('confidence_score', 0))
        impacts.append(rec.get('impact_score', 0))
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=categories,
        y=scores,
        name='Confidence Score',
        marker_color='blue'
    ))
    
    fig.add_trace(go.Bar(
        x=categories,
        y=impacts,
        name='Impact Score',
        marker_color='red'
    ))
    
    fig.update_layout(
        title="Recommendation Analysis",
        xaxis_title="Category",
        yaxis_title="Score",
        barmode='group'
    )
    
    return fig


def create_priority_matrix(prioritized_actions: List[Dict]) -> go.Figure:
    """Create a priority matrix visualization."""
    if not prioritized_actions:
        fig = go.Figure()
        fig.add_annotation(
            text="No actions to prioritize",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(title="Priority Matrix")
        return fig
    
    # Extract data
    actions = [action.get('name', 'Unknown') for action in prioritized_actions]
    priorities = [action.get('priority_score', 0) for action in prioritized_actions]
    efforts = [action.get('effort_score', 0) for action in prioritized_actions]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=efforts,
        y=priorities,
        mode='markers+text',
        text=actions,
        textposition="top center",
        marker=dict(
            size=20,
            color=priorities,
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Priority Score")
        ),
        name='Actions'
    ))
    
    fig.update_layout(
        title="Action Priority Matrix",
        xaxis_title="Effort Required",
        yaxis_title="Priority Score",
        xaxis=dict(range=[0, 10]),
        yaxis=dict(range=[0, 10])
    )
    
    return fig


def create_timeline_chart(phases: List[Dict]) -> go.Figure:
    """Create a timeline visualization for implementation plan."""
    if not phases:
        fig = go.Figure()
        fig.add_annotation(
            text="No implementation plan available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(title="Implementation Timeline")
        return fig
    
    # Create Gantt chart data
    fig = go.Figure()
    
    for i, phase in enumerate(phases):
        start_date = phase.get('start_date', f'Phase {i+1}')
        end_date = phase.get('end_date', f'Phase {i+1}')
        duration = phase.get('duration_days', 30)
        
        fig.add_trace(go.Bar(
            name=phase.get('name', f'Phase {i+1}'),
            x=[duration],
            y=[phase.get('name', f'Phase {i+1}')],
            orientation='h',
            text=f"{duration} days",
            textposition='auto',
        ))
    
    fig.update_layout(
        title="Implementation Timeline",
        xaxis_title="Duration (days)",
        yaxis_title="Phase",
        barmode='stack'
    )
    
    return fig


def display_recommendation_engine():
    """Display the AI recommendation engine."""
    st.markdown("## 🤖 AI Recommendation Engine")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Business context input
        st.markdown("### 📋 Business Context")
        
        business_goal = st.text_area(
            "Business Goal:",
            placeholder="Describe your primary business objective...",
            height=100
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            industry = st.selectbox(
                "Industry:",
                ["Technology", "Healthcare", "Finance", "Retail", "Manufacturing", "Other"]
            )
            
            company_size = st.selectbox(
                "Company Size:",
                ["Startup", "Small", "Medium", "Large", "Enterprise"]
            )
        
        with col2:
            budget_range = st.selectbox(
                "Budget Range:",
                ["< $10K", "$10K - $50K", "$50K - $100K", "$100K - $500K", "> $500K"]
            )
            
            timeline = st.selectbox(
                "Timeline:",
                ["< 1 month", "1-3 months", "3-6 months", "6-12 months", "> 1 year"]
            )
        
        # Current challenges
        challenges = st.multiselect(
            "Current Challenges:",
            ["Performance Issues", "Scalability", "Cost Optimization", "Security", "User Experience", "Data Quality", "Integration", "Compliance"]
        )
        
        if st.button("Generate Recommendations", type="primary"):
            if business_goal.strip():
                context = {
                    "business_goal": business_goal,
                    "industry": industry,
                    "company_size": company_size,
                    "budget_range": budget_range,
                    "timeline": timeline,
                    "challenges": challenges
                }
                
                with st.spinner("Generating AI recommendations..."):
                    result = get_recommendations(context)
                    
                    if "error" not in result:
                        recommendations = result.get("recommendations", [])
                        
                        if recommendations:
                            # Create recommendation chart
                            fig = create_recommendation_chart(recommendations)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display recommendations
                            st.markdown("### 💡 AI Recommendations")
                            
                            for i, rec in enumerate(recommendations, 1):
                                with st.expander(f"Recommendation {i}: {rec.get('title', 'Untitled')}"):
                                    st.write(f"**Description**: {rec.get('description', 'No description')}")
                                    st.write(f"**Category**: {rec.get('category', 'Unknown')}")
                                    st.write(f"**Confidence**: {rec.get('confidence_score', 0):.1f}/10")
                                    st.write(f"**Impact**: {rec.get('impact_score', 0):.1f}/10")
                                    st.write(f"**Effort**: {rec.get('effort_score', 0):.1f}/10")
                                    
                                    if rec.get('rationale'):
                                        st.write(f"**Rationale**: {rec['rationale']}")
                                    
                                    if rec.get('risks'):
                                        st.write("**Risks**:")
                                        for risk in rec['risks']:
                                            st.write(f"• {risk}")
                        else:
                            st.info("No recommendations generated. Please try different parameters.")
                    else:
                        st.error(f"Error generating recommendations: {result['error']}")
            else:
                st.warning("Please provide a business goal.")
    
    with col2:
        st.markdown("### 🎯 Recommendation Types")
        st.markdown("""
        **AI Analysis:**
        - **Strategic**: Long-term business strategy
        - **Operational**: Process improvements
        - **Technical**: Technology solutions
        - **Financial**: Cost optimization
        - **Risk**: Risk mitigation strategies
        """)
        
        st.markdown("### 📊 Scoring Criteria")
        st.markdown("""
        **Evaluation Factors:**
        - **Confidence**: AI model certainty
        - **Impact**: Expected business value
        - **Effort**: Implementation complexity
        - **Risk**: Potential downsides
        - **ROI**: Return on investment
        """)


def display_action_prioritization():
    """Display action prioritization interface."""
    st.markdown("## 📊 Action Prioritization")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎯 Action Input")
        
        # Action 1
        with st.expander("Action 1"):
            action1_name = st.text_input("Action Name:", key="action1_name")
            action1_impact = st.slider("Impact (1-10):", 1, 10, 7, key="action1_impact")
            action1_effort = st.slider("Effort (1-10):", 1, 10, 5, key="action1_effort")
            action1_urgency = st.slider("Urgency (1-10):", 1, 10, 6, key="action1_urgency")
        
        # Action 2
        with st.expander("Action 2"):
            action2_name = st.text_input("Action Name:", key="action2_name")
            action2_impact = st.slider("Impact (1-10):", 1, 10, 8, key="action2_impact")
            action2_effort = st.slider("Effort (1-10):", 1, 10, 7, key="action2_effort")
            action2_urgency = st.slider("Urgency (1-10):", 1, 10, 5, key="action2_urgency")
        
        # Action 3
        with st.expander("Action 3"):
            action3_name = st.text_input("Action Name:", key="action3_name")
            action3_impact = st.slider("Impact (1-10):", 1, 10, 6, key="action3_impact")
            action3_effort = st.slider("Effort (1-10):", 1, 10, 3, key="action3_effort")
            action3_urgency = st.slider("Urgency (1-10):", 1, 10, 8, key="action3_urgency")
        
        if st.button("Prioritize Actions", type="primary"):
            actions = []
            
            if action1_name:
                actions.append({
                    "name": action1_name,
                    "impact_score": action1_impact,
                    "effort_score": action1_effort,
                    "urgency_score": action1_urgency
                })
            
            if action2_name:
                actions.append({
                    "name": action2_name,
                    "impact_score": action2_impact,
                    "effort_score": action2_effort,
                    "urgency_score": action2_urgency
                })
            
            if action3_name:
                actions.append({
                    "name": action3_name,
                    "impact_score": action3_impact,
                    "effort_score": action3_effort,
                    "urgency_score": action3_urgency
                })
            
            if actions:
                with st.spinner("Prioritizing actions..."):
                    result = prioritize_actions(actions)
                    
                    if "error" not in result:
                        prioritized_actions = result.get("prioritized_actions", [])
                        
                        if prioritized_actions:
                            # Create priority matrix
                            fig = create_priority_matrix(prioritized_actions)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display prioritized list
                            st.markdown("### 📋 Prioritized Actions")
                            
                            for i, action in enumerate(prioritized_actions, 1):
                                st.write(f"**{i}. {action.get('name', 'Unknown')}**")
                                st.write(f"   Priority Score: {action.get('priority_score', 0):.1f}")
                                st.write(f"   Impact: {action.get('impact_score', 0)}/10")
                                st.write(f"   Effort: {action.get('effort_score', 0)}/10")
                                st.write(f"   Urgency: {action.get('urgency_score', 0)}/10")
                                st.write("---")
                        else:
                            st.info("No prioritized actions returned.")
                    else:
                        st.error(f"Error prioritizing actions: {result['error']}")
            else:
                st.warning("Please provide at least one action.")
    
    with col2:
        st.markdown("### ⚖️ Prioritization Factors")
        st.markdown("""
        **Scoring Criteria:**
        - **Impact**: Business value (1-10)
        - **Effort**: Implementation complexity (1-10)
        - **Urgency**: Time sensitivity (1-10)
        - **Risk**: Potential downsides
        - **Resources**: Required capabilities
        """)
        
        st.markdown("### 📊 Priority Matrix")
        st.markdown("""
        **Quadrants:**
        - **High Impact, Low Effort**: Quick wins
        - **High Impact, High Effort**: Major projects
        - **Low Impact, Low Effort**: Fill-ins
        - **Low Impact, High Effort**: Avoid
        """)


def display_implementation_planning():
    """Display implementation planning interface."""
    st.markdown("## 📅 Implementation Planning")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎯 Goal Definition")
        
        goal = st.text_area(
            "Implementation Goal:",
            placeholder="Describe what you want to achieve...",
            height=100
        )
        
        st.markdown("### 🚧 Constraints")
        
        col1, col2 = st.columns(2)
        
        with col1:
            budget = st.number_input("Budget ($):", min_value=0, value=50000)
            timeline_months = st.slider("Timeline (months):", 1, 24, 6)
        
        with col2:
            team_size = st.slider("Team Size:", 1, 20, 5)
            risk_tolerance = st.selectbox(
                "Risk Tolerance:",
                ["Low", "Medium", "High"]
            )
        
        # Additional constraints
        constraints = st.multiselect(
            "Additional Constraints:",
            ["Regulatory Compliance", "Technical Debt", "Legacy Systems", "Security Requirements", "Performance Requirements", "Scalability Needs"]
        )
        
        if st.button("Generate Implementation Plan", type="primary"):
            if goal.strip():
                constraint_data = {
                    "budget": budget,
                    "timeline_months": timeline_months,
                    "team_size": team_size,
                    "risk_tolerance": risk_tolerance,
                    "additional_constraints": constraints
                }
                
                with st.spinner("Generating implementation plan..."):
                    result = create_implementation_plan(goal, constraint_data)
                    
                    if "error" not in result:
                        plan = result.get("implementation_plan", {})
                        phases = plan.get("phases", [])
                        
                        if phases:
                            # Create timeline chart
                            fig = create_timeline_chart(phases)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display plan details
                            st.markdown("### 📋 Implementation Plan")
                            
                            for i, phase in enumerate(phases, 1):
                                with st.expander(f"Phase {i}: {phase.get('name', 'Untitled')}"):
                                    st.write(f"**Duration**: {phase.get('duration_days', 0)} days")
                                    st.write(f"**Budget**: ${phase.get('budget', 0):,.0f}")
                                    st.write(f"**Team**: {phase.get('team_size', 0)} people")
                                    
                                    tasks = phase.get('tasks', [])
                                    if tasks:
                                        st.write("**Tasks**:")
                                        for task in tasks:
                                            st.write(f"• {task}")
                                    
                                    risks = phase.get('risks', [])
                                    if risks:
                                        st.write("**Risks**:")
                                        for risk in risks:
                                            st.write(f"• {risk}")
                            
                            # Summary metrics
                            st.markdown("### 📊 Plan Summary")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                total_duration = sum(phase.get('duration_days', 0) for phase in phases)
                                st.metric("Total Duration", f"{total_duration} days")
                            
                            with col2:
                                total_budget = sum(phase.get('budget', 0) for phase in phases)
                                st.metric("Total Budget", f"${total_budget:,.0f}")
                            
                            with col3:
                                success_probability = plan.get('success_probability', 0)
                                st.metric("Success Probability", f"{success_probability:.1f}%")
                        else:
                            st.info("No implementation plan generated.")
                    else:
                        st.error(f"Error generating plan: {result['error']}")
            else:
                st.warning("Please provide an implementation goal.")
    
    with col2:
        st.markdown("### 📋 Planning Elements")
        st.markdown("""
        **Plan Components:**
        - **Phases**: Logical project stages
        - **Tasks**: Specific activities
        - **Timeline**: Duration estimates
        - **Resources**: Team and budget
        - **Risks**: Potential issues
        """)
        
        st.markdown("### 🎯 Success Factors")
        st.markdown("""
        **Key Success Factors:**
        - Clear objectives
        - Realistic timelines
        - Adequate resources
        - Risk mitigation
        - Stakeholder alignment
        """)


def main():
    """Main decision support interface."""
    st.set_page_config(
        page_title="Decision Support",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown('<h1 class="main-header">🤖 Decision Support Interface</h1>', unsafe_allow_html=True)
    
    # Navigation tabs
    tab1, tab2, tab3 = st.tabs([
        "🤖 Recommendations", 
        "📊 Prioritization", 
        "📅 Implementation"
    ])
    
    with tab1:
        display_recommendation_engine()
    
    with tab2:
        display_action_prioritization()
    
    with tab3:
        display_implementation_planning()


if __name__ == "__main__":
    main()
