#!/usr/bin/env python3
"""
Art of War Strategic Scenario Analysis Demonstration
This script demonstrates how to use the Advanced Analytics System for strategic
scenario analysis using Sun Tzu's Art of War principles in a business context.
"""

import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class ArtOfWarScenarioAnalysis:
    def __init__(self, base_url="http://127.0.0.1:8003"):
        self.base_url = base_url
        
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
        """Generate strategic analysis data based on Art of War principles."""
        print("\n📊 Generating Strategic Data Based on Art of War Principles")
        print("=" * 60)
        
        # The Five Fundamentals (五事) from Art of War
        strategic_data = {
            "the_way": {
                "organizational_culture": 85,
                "stakeholder_alignment": 78,
                "purpose_clarity": 82,
                "values_consistency": 80
            },
            "heaven": {
                "market_timing": 75,
                "economic_conditions": 70,
                "seasonal_factors": 85,
                "external_pressures": 65
            },
            "earth": {
                "market_position": 72,
                "geographic_advantage": 80,
                "resource_availability": 75,
                "competitive_landscape": 68
            },
            "command": {
                "leadership_effectiveness": 78,
                "decision_making": 75,
                "strategic_vision": 82,
                "execution_capability": 80
            },
            "method": {
                "operational_efficiency": 85,
                "process_discipline": 78,
                "resource_allocation": 80,
                "performance_monitoring": 82
            }
        }
        
        print("🎯 Art of War Five Fundamentals Analysis:")
        for fundamental, metrics in strategic_data.items():
            avg_score = sum(metrics.values()) / len(metrics)
            print(f"   • {fundamental.replace('_', ' ').title()}: {avg_score:.1f}/100")
        
        return strategic_data
    
    def analyze_defensive_strategy(self, data):
        """Analyze defensive strategy based on Art of War principles."""
        print("\n🛡️ DEFENSIVE STRATEGY ANALYSIS")
        print("=" * 50)
        print("Art of War Principle: 'The supreme art of war is to subdue the enemy without fighting.'")
        
        # Calculate defensive strength
        defensive_metrics = {
            "positional_advantage": data["earth"]["market_position"] * 0.3 + data["earth"]["geographic_advantage"] * 0.7,
            "resource_readiness": data["method"]["operational_efficiency"] * 0.4 + data["method"]["resource_allocation"] * 0.6,
            "strategic_depth": data["the_way"]["organizational_culture"] * 0.3 + data["command"]["leadership_effectiveness"] * 0.7,
            "timing_advantage": data["heaven"]["market_timing"] * 0.5 + data["heaven"]["seasonal_factors"] * 0.5
        }
        
        overall_defensive_strength = sum(defensive_metrics.values()) / len(defensive_metrics)
        
        print(f"📊 Defensive Strategy Assessment:")
        print(f"   • Positional Advantage: {defensive_metrics['positional_advantage']:.1f}/100")
        print(f"   • Resource Readiness: {defensive_metrics['resource_readiness']:.1f}/100")
        print(f"   • Strategic Depth: {defensive_metrics['strategic_depth']:.1f}/100")
        print(f"   • Timing Advantage: {defensive_metrics['timing_advantage']:.1f}/100")
        print(f"   • Overall Defensive Strength: {overall_defensive_strength:.1f}/100")
        
        return defensive_metrics
    
    def analyze_offensive_strategy(self, data):
        """Analyze offensive strategy based on Art of War principles."""
        print("\n⚔️ OFFENSIVE STRATEGY ANALYSIS")
        print("=" * 50)
        print("Art of War Principle: 'Supreme excellence consists of breaking the enemy's resistance without fighting.'")
        
        # Calculate offensive capability
        offensive_metrics = {
            "initiative_capability": data["command"]["decision_making"] * 0.4 + data["command"]["strategic_vision"] * 0.6,
            "resource_mobilization": data["method"]["resource_allocation"] * 0.5 + data["method"]["operational_efficiency"] * 0.5,
            "timing_precision": data["heaven"]["market_timing"] * 0.6 + data["heaven"]["seasonal_factors"] * 0.4,
            "competitive_advantage": data["earth"]["market_position"] * 0.3 + data["the_way"]["purpose_clarity"] * 0.7
        }
        
        overall_offensive_strength = sum(offensive_metrics.values()) / len(offensive_metrics)
        
        print(f"📊 Offensive Strategy Assessment:")
        print(f"   • Initiative Capability: {offensive_metrics['initiative_capability']:.1f}/100")
        print(f"   • Resource Mobilization: {offensive_metrics['resource_mobilization']:.1f}/100")
        print(f"   • Timing Precision: {offensive_metrics['timing_precision']:.1f}/100")
        print(f"   • Competitive Advantage: {offensive_metrics['competitive_advantage']:.1f}/100")
        print(f"   • Overall Offensive Strength: {overall_offensive_strength:.1f}/100")
        
        return offensive_metrics
    
    def analyze_alliance_strategy(self, data):
        """Analyze alliance building strategy based on Art of War principles."""
        print("\n🤝 ALLIANCE STRATEGY ANALYSIS")
        print("=" * 50)
        print("Art of War Principle: 'He who knows the art of the direct and the indirect approach will be victorious.'")
        
        # Calculate alliance potential
        alliance_metrics = {
            "diplomatic_capability": data["the_way"]["stakeholder_alignment"] * 0.6 + data["the_way"]["values_consistency"] * 0.4,
            "resource_sharing": data["method"]["resource_allocation"] * 0.5 + data["method"]["operational_efficiency"] * 0.5,
            "strategic_alignment": data["the_way"]["purpose_clarity"] * 0.7 + data["command"]["strategic_vision"] * 0.3,
            "trust_building": data["the_way"]["organizational_culture"] * 0.8 + data["the_way"]["stakeholder_alignment"] * 0.2
        }
        
        overall_alliance_potential = sum(alliance_metrics.values()) / len(alliance_metrics)
        
        print(f"📊 Alliance Strategy Assessment:")
        print(f"   • Diplomatic Capability: {alliance_metrics['diplomatic_capability']:.1f}/100")
        print(f"   • Resource Sharing: {alliance_metrics['resource_sharing']:.1f}/100")
        print(f"   • Strategic Alignment: {alliance_metrics['strategic_alignment']:.1f}/100")
        print(f"   • Trust Building: {alliance_metrics['trust_building']:.1f}/100")
        print(f"   • Overall Alliance Potential: {overall_alliance_potential:.1f}/100")
        
        return alliance_metrics
    
    def analyze_resource_optimization(self, data):
        """Analyze resource optimization strategy based on Art of War principles."""
        print("\n⚙️ RESOURCE OPTIMIZATION ANALYSIS")
        print("=" * 50)
        print("Art of War Principle: 'The art of war teaches us to rely not on the likelihood of the enemy's not coming, but on our own readiness to receive him.'")
        
        # Calculate resource optimization
        optimization_metrics = {
            "efficiency_maximization": data["method"]["operational_efficiency"] * 0.6 + data["method"]["process_discipline"] * 0.4,
            "resource_flexibility": data["method"]["resource_allocation"] * 0.7 + data["method"]["performance_monitoring"] * 0.3,
            "strategic_reserves": data["earth"]["resource_availability"] * 0.5 + data["method"]["resource_allocation"] * 0.5,
            "adaptive_capability": data["command"]["decision_making"] * 0.4 + data["command"]["execution_capability"] * 0.6
        }
        
        overall_optimization_score = sum(optimization_metrics.values()) / len(optimization_metrics)
        
        print(f"📊 Resource Optimization Assessment:")
        print(f"   • Efficiency Maximization: {optimization_metrics['efficiency_maximization']:.1f}/100")
        print(f"   • Resource Flexibility: {optimization_metrics['resource_flexibility']:.1f}/100")
        print(f"   • Strategic Reserves: {optimization_metrics['strategic_reserves']:.1f}/100")
        print(f"   • Adaptive Capability: {optimization_metrics['adaptive_capability']:.1f}/100")
        print(f"   • Overall Optimization Score: {overall_optimization_score:.1f}/100")
        
        return optimization_metrics
    
    def generate_strategic_recommendations(self, all_metrics):
        """Generate strategic recommendations based on Art of War principles."""
        print("\n📋 STRATEGIC RECOMMENDATIONS")
        print("=" * 60)
        print("Based on Art of War Principles and Strategic Analysis")
        
        # Analyze strengths and weaknesses
        defensive_strength = sum(all_metrics['defensive'].values()) / len(all_metrics['defensive'])
        offensive_strength = sum(all_metrics['offensive'].values()) / len(all_metrics['offensive'])
        alliance_potential = sum(all_metrics['alliance'].values()) / len(all_metrics['alliance'])
        optimization_score = sum(all_metrics['optimization'].values()) / len(all_metrics['optimization'])
        
        print(f"\n🎯 Strategic Position Assessment:")
        print(f"   • Defensive Strength: {defensive_strength:.1f}/100")
        print(f"   • Offensive Capability: {offensive_strength:.1f}/100")
        print(f"   • Alliance Potential: {alliance_potential:.1f}/100")
        print(f"   • Resource Optimization: {optimization_score:.1f}/100")
        
        # Determine optimal strategy
        strategies = {
            "defensive": defensive_strength,
            "offensive": offensive_strength,
            "alliance": alliance_potential,
            "optimization": optimization_score
        }
        
        optimal_strategy = max(strategies, key=strategies.get)
        
        print(f"\n🏆 Recommended Primary Strategy: {optimal_strategy.replace('_', ' ').title()}")
        
        # Generate specific recommendations
        print(f"\n💡 Specific Recommendations:")
        
        if optimal_strategy == "defensive":
            print("   • Strengthen market position and competitive moats")
            print("   • Build strategic reserves and defensive capabilities")
            print("   • Focus on operational excellence and efficiency")
            print("   • Develop early warning systems and monitoring")
            
        elif optimal_strategy == "offensive":
            print("   • Leverage competitive advantages aggressively")
            print("   • Invest in innovation and market expansion")
            print("   • Build rapid response and execution capabilities")
            print("   • Develop strategic initiatives and new markets")
            
        elif optimal_strategy == "alliance":
            print("   • Build strategic partnerships and alliances")
            print("   • Develop collaborative capabilities and networks")
            print("   • Focus on stakeholder engagement and trust")
            print("   • Create shared value and mutual benefits")
            
        elif optimal_strategy == "optimization":
            print("   • Optimize resource allocation and efficiency")
            print("   • Build adaptive and flexible capabilities")
            print("   • Develop performance monitoring and improvement")
            print("   • Focus on continuous optimization and learning")
        
        # Secondary strategy recommendations
        strategies_sorted = sorted(strategies.items(), key=lambda x: x[1], reverse=True)
        secondary_strategy = strategies_sorted[1][0]
        
        print(f"\n🔄 Secondary Strategy: {secondary_strategy.replace('_', ' ').title()}")
        print("   • Balance primary strategy with complementary approaches")
        print("   • Develop hybrid capabilities and flexibility")
        print("   • Maintain strategic options and adaptability")
        
        return optimal_strategy, secondary_strategy
    
    def run_complete_analysis(self):
        """Run the complete Art of War strategic scenario analysis."""
        print("🎯 ART OF WAR STRATEGIC SCENARIO ANALYSIS")
        print("=" * 70)
        print("Applying Sun Tzu's Art of War Principles to Strategic Analysis")
        print("=" * 70)
        
        # Check system health
        if not self.check_system_health():
            print("❌ System not available. Please start the server first.")
            return
        
        # Generate strategic data
        strategic_data = self.generate_strategic_data()
        
        # Run scenario analyses
        all_metrics = {}
        
        # Defensive strategy analysis
        all_metrics['defensive'] = self.analyze_defensive_strategy(strategic_data)
        
        # Offensive strategy analysis
        all_metrics['offensive'] = self.analyze_offensive_strategy(strategic_data)
        
        # Alliance strategy analysis
        all_metrics['alliance'] = self.analyze_alliance_strategy(strategic_data)
        
        # Resource optimization analysis
        all_metrics['optimization'] = self.analyze_resource_optimization(strategic_data)
        
        # Generate strategic recommendations
        primary_strategy, secondary_strategy = self.generate_strategic_recommendations(all_metrics)
        
        # Final summary
        print(f"\n✅ Art of War Strategic Analysis Completed!")
        print(f"📊 Primary Strategy: {primary_strategy.replace('_', ' ').title()}")
        print(f"📊 Secondary Strategy: {secondary_strategy.replace('_', ' ').title()}")
        print(f"📁 Analysis results available for further strategic planning")
        
        return {
            "strategic_data": strategic_data,
            "metrics": all_metrics,
            "primary_strategy": primary_strategy,
            "secondary_strategy": secondary_strategy
        }

def main():
    """Main function to run the Art of War scenario analysis."""
    analyzer = ArtOfWarScenarioAnalysis()
    results = analyzer.run_complete_analysis()
    
    print(f"\n📈 Analysis Summary:")
    print(f"   • Data Points Analyzed: {len(results['strategic_data'])}")
    print(f"   • Scenarios Evaluated: {len(results['metrics'])}")
    print(f"   • Strategic Dimensions: 20+ metrics across 5 fundamentals")
    print(f"   • Recommendations Generated: Primary + Secondary strategies")

if __name__ == "__main__":
    main()
