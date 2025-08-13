#!/usr/bin/env python3
"""
Strategic Analysis Demonstration using Advanced Analytics System
This script demonstrates how to use the system for general strategic analysis
using historical data and analytical frameworks.
"""

import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

class StrategicAnalysisDemo:
    def __init__(self, base_url="http://127.0.0.1:8000"):
        self.base_url = base_url
        self.advanced_analytics_url = f"{base_url}/advanced-analytics"
        
    def check_system_health(self):
        """Check if the system is running."""
        try:
            response = requests.get(f"{self.base_url}/health")
            if response.status_code == 200:
                print("✅ System is healthy and running")
                return True
            else:
                print("❌ System health check failed")
                return False
        except Exception as e:
            print(f"❌ Cannot connect to system: {e}")
            return False
    
    def generate_strategic_data(self):
        """Generate sample strategic analysis data."""
        # Historical economic and stability indicators (fictional data)
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='M')
        
        data = []
        for i, date in enumerate(dates):
            # Simulate realistic patterns with some volatility
            base_economic = 100 + i * 0.5 + np.random.normal(0, 2)
            base_stability = 85 - i * 0.1 + np.random.normal(0, 3)
            base_trade = 1000 + i * 10 + np.random.normal(0, 50)
            
            # Add some strategic events impact
            if i % 12 == 0:  # Annual cycles
                base_stability += np.random.normal(-5, 2)
            
            data.append({
                "date": date.strftime("%Y-%m-%d"),
                "economic_indicator": max(50, min(150, base_economic)),
                "stability_index": max(30, min(100, base_stability)),
                "trade_volume": max(500, min(2000, base_trade)),
                "strategic_confidence": max(20, min(90, 70 + np.random.normal(0, 5)))
            })
        
        return data
    
    def demonstrate_forecasting(self, data):
        """Demonstrate multivariate forecasting capabilities."""
        print("\n🔮 STRATEGIC FORECASTING ANALYSIS")
        print("=" * 50)
        
        # Prepare data for forecasting
        forecast_data = data[-24:]  # Last 2 years
        
        request_payload = {
            "data": forecast_data,
            "target_variables": ["economic_indicator", "stability_index", "trade_volume"],
            "forecast_horizon": 12,
            "model_type": "ensemble"
        }
        
        try:
            response = requests.post(
                f"{self.advanced_analytics_url}/forecasting/multivariate",
                json=request_payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Forecasting analysis completed")
                print(f"📊 Forecast horizon: 12 months")
                print(f"🎯 Target variables: Economic, Stability, Trade")
                print(f"🤖 Model type: Ensemble (multiple algorithms)")
                
                # Display key insights
                if 'forecast' in result:
                    forecast = result['forecast']
                    print(f"\n📈 Key Forecast Insights:")
                    print(f"   • Economic trend: {'↗️ Rising' if forecast.get('economic_indicator', {}).get('trend', '') == 'up' else '↘️ Declining'}")
                    print(f"   • Stability outlook: {'🟢 Stable' if forecast.get('stability_index', {}).get('trend', '') == 'stable' else '🟡 Variable'}")
                    print(f"   • Trade projection: {'📈 Growing' if forecast.get('trade_volume', {}).get('trend', '') == 'up' else '📉 Declining'}")
                
                return result
            else:
                print(f"❌ Forecasting failed: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Forecasting error: {e}")
            return None
    
    def demonstrate_scenario_analysis(self, data):
        """Demonstrate scenario analysis capabilities."""
        print("\n🎭 SCENARIO ANALYSIS")
        print("=" * 50)
        
        # Create different scenarios
        scenarios = [
            {
                "name": "Optimistic Scenario",
                "economic_multiplier": 1.2,
                "stability_boost": 5,
                "trade_growth": 1.15
            },
            {
                "name": "Baseline Scenario", 
                "economic_multiplier": 1.0,
                "stability_boost": 0,
                "trade_growth": 1.0
            },
            {
                "name": "Conservative Scenario",
                "economic_multiplier": 0.8,
                "stability_boost": -3,
                "trade_growth": 0.9
            }
        ]
        
        request_payload = {
            "data": data[-12:],  # Last year's data
            "scenarios": [s["name"] for s in scenarios],
            "variables": ["economic_indicator", "stability_index", "trade_volume"],
            "timeframe": "12_months"
        }
        
        try:
            response = requests.post(
                f"{self.advanced_analytics_url}/scenario/analysis",
                json=request_payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Scenario analysis completed")
                print(f"📊 Analyzed {len(scenarios)} scenarios")
                print(f"⏰ Timeframe: 12 months")
                
                # Display scenario comparisons
                if 'scenarios' in result:
                    print(f"\n📋 Scenario Comparison:")
                    for scenario in result['scenarios']:
                        name = scenario.get('name', 'Unknown')
                        confidence = scenario.get('confidence', 0)
                        risk_level = scenario.get('risk_level', 'Medium')
                        print(f"   • {name}: Confidence {confidence}%, Risk {risk_level}")
                
                return result
            else:
                print(f"❌ Scenario analysis failed: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Scenario analysis error: {e}")
            return None
    
    def demonstrate_anomaly_detection(self, data):
        """Demonstrate anomaly detection capabilities."""
        print("\n🚨 ANOMALY DETECTION")
        print("=" * 50)
        
        # Extract numerical values for anomaly detection
        stability_values = [d["stability_index"] for d in data]
        
        request_payload = {
            "data": stability_values,
            "method": "isolation_forest",
            "threshold": 0.95
        }
        
        try:
            response = requests.post(
                f"{self.advanced_analytics_url}/anomaly/detection",
                json=request_payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Anomaly detection completed")
                print(f"🔍 Method: Isolation Forest")
                print(f"📊 Threshold: 95% confidence")
                
                # Display anomalies found
                if 'anomalies' in result:
                    anomalies = result['anomalies']
                    print(f"\n⚠️  Anomalies Detected: {len(anomalies)}")
                    for i, anomaly in enumerate(anomalies[:5]):  # Show first 5
                        index = anomaly.get('index', i)
                        value = anomaly.get('value', 0)
                        score = anomaly.get('anomaly_score', 0)
                        print(f"   • Point {index}: Value {value:.1f}, Score {score:.3f}")
                
                return result
            else:
                print(f"❌ Anomaly detection failed: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Anomaly detection error: {e}")
            return None
    
    def demonstrate_causal_analysis(self, data):
        """Demonstrate causal analysis capabilities."""
        print("\n🔗 CAUSAL ANALYSIS")
        print("=" * 50)
        
        # Create treatment/outcome data
        causal_data = []
        for i, point in enumerate(data[-24:]):  # Last 2 years
            # Simulate treatment effect (e.g., policy changes)
            treatment = 1 if i % 6 == 0 else 0  # Every 6 months
            outcome = point["economic_indicator"]
            covariates = [point["stability_index"], point["trade_volume"]]
            
            causal_data.append({
                "treatment": treatment,
                "outcome": outcome,
                "covariates": covariates
            })
        
        request_payload = {
            "data": causal_data,
            "treatment_variable": "treatment",
            "outcome_variable": "outcome",
            "method": "propensity_score"
        }
        
        try:
            response = requests.post(
                f"{self.advanced_analytics_url}/causal/analysis",
                json=request_payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Causal analysis completed")
                print(f"🔬 Method: Propensity Score Matching")
                print(f"📊 Data points: {len(causal_data)}")
                
                # Display causal effects
                if 'causal_effect' in result:
                    effect = result['causal_effect']
                    print(f"\n📈 Causal Effect Analysis:")
                    print(f"   • Average Treatment Effect: {effect.get('ate', 0):.2f}")
                    print(f"   • Confidence Interval: {effect.get('confidence_interval', [0, 0])}")
                    print(f"   • P-value: {effect.get('p_value', 0):.4f}")
                
                return result
            else:
                print(f"❌ Causal analysis failed: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Causal analysis error: {e}")
            return None
    
    def generate_strategic_report(self, results):
        """Generate a comprehensive strategic analysis report."""
        print("\n📋 STRATEGIC ANALYSIS REPORT")
        print("=" * 60)
        print(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🔧 Analysis Tools: Advanced Analytics System v1.0")
        print(f"📊 Data Period: 2020-2023 (48 months)")
        
        print(f"\n🎯 KEY FINDINGS:")
        print(f"   • Forecasting: {'✅ Completed' if results.get('forecasting') else '❌ Failed'}")
        print(f"   • Scenarios: {'✅ Completed' if results.get('scenarios') else '❌ Failed'}")
        print(f"   • Anomalies: {'✅ Completed' if results.get('anomalies') else '❌ Failed'}")
        print(f"   • Causality: {'✅ Completed' if results.get('causality') else '❌ Failed'}")
        
        print(f"\n💡 STRATEGIC INSIGHTS:")
        print(f"   • Economic trends show {'positive' if results.get('forecasting') else 'variable'} momentum")
        print(f"   • Stability indicators suggest {'favorable' if results.get('scenarios') else 'mixed'} conditions")
        print(f"   • Trade patterns indicate {'growth' if results.get('forecasting') else 'stability'} trajectory")
        print(f"   • Risk factors identified: {'Low' if results.get('anomalies') else 'Medium'}")
        
        print(f"\n🎭 SCENARIO RECOMMENDATIONS:")
        print(f"   • Optimistic: {'Favorable' if results.get('scenarios') else 'Cautious'} outlook")
        print(f"   • Baseline: {'Stable' if results.get('scenarios') else 'Variable'} expectations")
        print(f"   • Conservative: {'Risk-averse' if results.get('scenarios') else 'Balanced'} approach")
        
        print(f"\n⚠️  RISK ASSESSMENT:")
        print(f"   • Overall Risk Level: {'Low' if results.get('anomalies') else 'Medium'}")
        print(f"   • Key Risk Factors: Economic volatility, stability fluctuations")
        print(f"   • Mitigation Strategies: Diversification, monitoring, adaptive planning")
        
        print(f"\n📈 NEXT STEPS:")
        print(f"   • Continue monitoring key indicators")
        print(f"   • Update forecasts monthly")
        print(f"   • Adjust strategies based on new data")
        print(f"   • Maintain scenario planning framework")
    
    def run_complete_analysis(self):
        """Run the complete strategic analysis demonstration."""
        print("🎯 STRATEGIC ANALYSIS DEMONSTRATION")
        print("=" * 60)
        print("This demonstration shows how to use the Advanced Analytics System")
        print("for general strategic analysis using historical data and frameworks.")
        print("=" * 60)
        
        # Check system health
        if not self.check_system_health():
            print("❌ System not available. Please start the server first.")
            return
        
        # Generate sample data
        print("\n📊 Generating strategic analysis data...")
        data = self.generate_strategic_data()
        print(f"✅ Generated {len(data)} data points (2020-2023)")
        
        # Run analyses
        results = {}
        
        # Forecasting
        results['forecasting'] = self.demonstrate_forecasting(data)
        
        # Scenario analysis
        results['scenarios'] = self.demonstrate_scenario_analysis(data)
        
        # Anomaly detection
        results['anomalies'] = self.demonstrate_anomaly_detection(data)
        
        # Causal analysis
        results['causality'] = self.demonstrate_causal_analysis(data)
        
        # Generate report
        self.generate_strategic_report(results)
        
        print(f"\n✅ Strategic analysis demonstration completed!")
        print(f"📁 Results saved in memory for further analysis")
        print(f"🔗 API endpoints available for custom analysis")

def main():
    """Main function to run the demonstration."""
    demo = StrategicAnalysisDemo()
    demo.run_complete_analysis()

if __name__ == "__main__":
    main()
