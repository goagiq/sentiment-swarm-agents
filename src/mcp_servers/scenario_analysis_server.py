"""
Scenario Analysis MCP Server

Provides MCP endpoints for scenario analysis capabilities including:
- What-if scenario modeling
- Impact analysis and risk assessment
- Scenario comparison and ranking
- Decision support and recommendations
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import uuid

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource, Tool, TextContent, ImageContent, EmbeddedResource,
    LoggingLevel, Position, Range, Location, Annotation, Annotations
)

# Import our scenario analysis components
from src.core.scenario_analysis.scenario_builder import ScenarioBuilder
from src.core.scenario_analysis.impact_analyzer import ImpactAnalyzer
from src.core.scenario_analysis.risk_assessor import RiskAssessor
from src.core.scenario_analysis.scenario_comparator import ScenarioComparator

logger = logging.getLogger(__name__)


class ScenarioAnalysisMCPServer:
    """
    MCP Server for Scenario Analysis capabilities
    """
    
    def __init__(self):
        """Initialize the scenario analysis MCP server"""
        self.server = Server("scenario-analysis")
        self.scenario_builder = ScenarioBuilder()
        self.impact_analyzer = ImpactAnalyzer()
        self.risk_assessor = RiskAssessor()
        self.scenario_comparator = ScenarioComparator()
        
        self._setup_tools()
        self._setup_resources()
        
    def _setup_tools(self):
        """Setup MCP tools for scenario analysis"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """List available scenario analysis tools"""
            return [
                Tool(
                    name="create_scenario",
                    description="Create a new what-if scenario",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "scenario_name": {
                                "type": "string",
                                "description": "Name of the scenario"
                            },
                            "description": {
                                "type": "string",
                                "description": "Description of the scenario"
                            },
                            "parameters": {
                                "type": "object",
                                "description": "Scenario parameters and variables"
                            },
                            "assumptions": {
                                "type": "array",
                                "description": "List of assumptions for the scenario",
                                "items": {"type": "string"}
                            },
                            "time_horizon": {
                                "type": "integer",
                                "description": "Time horizon in months",
                                "default": 12
                            }
                        },
                        "required": ["scenario_name", "parameters"]
                    }
                ),
                
                Tool(
                    name="analyze_impact",
                    description="Analyze the impact of a scenario",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "scenario_id": {
                                "type": "string",
                                "description": "ID of the scenario to analyze"
                            },
                            "impact_areas": {
                                "type": "array",
                                "description": "Areas to analyze for impact",
                                "items": {"type": "string"},
                                "default": ["financial", "operational", "strategic"]
                            },
                            "metrics": {
                                "type": "array",
                                "description": "Specific metrics to analyze",
                                "items": {"type": "string"}
                            },
                            "sensitivity_analysis": {
                                "type": "boolean",
                                "description": "Perform sensitivity analysis",
                                "default": True
                            }
                        },
                        "required": ["scenario_id"]
                    }
                ),
                
                Tool(
                    name="assess_risk",
                    description="Assess risks associated with a scenario",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "scenario_id": {
                                "type": "string",
                                "description": "ID of the scenario to assess"
                            },
                            "risk_categories": {
                                "type": "array",
                                "description": "Risk categories to assess",
                                "items": {"type": "string"},
                                "default": ["financial", "operational", "strategic", "compliance"]
                            },
                            "risk_tolerance": {
                                "type": "number",
                                "description": "Risk tolerance level (0-1)",
                                "default": 0.5
                            }
                        },
                        "required": ["scenario_id"]
                    }
                ),
                
                Tool(
                    name="compare_scenarios",
                    description="Compare multiple scenarios",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "scenario_ids": {
                                "type": "array",
                                "description": "IDs of scenarios to compare",
                                "items": {"type": "string"}
                            },
                            "comparison_metrics": {
                                "type": "array",
                                "description": "Metrics to compare",
                                "items": {"type": "string"}
                            },
                            "ranking_criteria": {
                                "type": "object",
                                "description": "Criteria for ranking scenarios"
                            }
                        },
                        "required": ["scenario_ids"]
                    }
                ),
                
                Tool(
                    name="generate_recommendations",
                    description="Generate recommendations based on scenario analysis",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "scenario_id": {
                                "type": "string",
                                "description": "ID of the scenario"
                            },
                            "recommendation_type": {
                                "type": "string",
                                "description": "Type of recommendations",
                                "enum": ["action", "mitigation", "optimization", "strategic"],
                                "default": "action"
                            },
                            "priority_level": {
                                "type": "string",
                                "description": "Priority level for recommendations",
                                "enum": ["low", "medium", "high", "critical"],
                                "default": "medium"
                            }
                        },
                        "required": ["scenario_id"]
                    }
                ),
                
                Tool(
                    name="create_sensitivity_analysis",
                    description="Create sensitivity analysis for scenario parameters",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "scenario_id": {
                                "type": "string",
                                "description": "ID of the scenario"
                            },
                            "parameters": {
                                "type": "array",
                                "description": "Parameters to analyze",
                                "items": {"type": "string"}
                            },
                            "variation_range": {
                                "type": "number",
                                "description": "Variation range as percentage",
                                "default": 20
                            },
                            "steps": {
                                "type": "integer",
                                "description": "Number of steps in analysis",
                                "default": 10
                            }
                        },
                        "required": ["scenario_id", "parameters"]
                    }
                ),
                
                Tool(
                    name="export_scenario_report",
                    description="Export scenario analysis report",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "scenario_id": {
                                "type": "string",
                                "description": "ID of the scenario"
                            },
                            "report_format": {
                                "type": "string",
                                "description": "Format of the report",
                                "enum": ["json", "pdf", "html", "excel"],
                                "default": "json"
                            },
                            "include_charts": {
                                "type": "boolean",
                                "description": "Include charts and visualizations",
                                "default": True
                            },
                            "sections": {
                                "type": "array",
                                "description": "Sections to include in report",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["scenario_id"]
                    }
                )
            ]
            
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls for scenario analysis"""
            
            try:
                if name == "create_scenario":
                    return await self._handle_create_scenario(arguments)
                elif name == "analyze_impact":
                    return await self._handle_analyze_impact(arguments)
                elif name == "assess_risk":
                    return await self._handle_assess_risk(arguments)
                elif name == "compare_scenarios":
                    return await self._handle_compare_scenarios(arguments)
                elif name == "generate_recommendations":
                    return await self._handle_generate_recommendations(arguments)
                elif name == "create_sensitivity_analysis":
                    return await self._handle_create_sensitivity_analysis(arguments)
                elif name == "export_scenario_report":
                    return await self._handle_export_scenario_report(arguments)
                else:
                    return [TextContent(
                        type="text",
                        text=f"Unknown tool: {name}"
                    )]
                    
            except Exception as e:
                logger.error(f"Error in tool {name}: {str(e)}")
                return [TextContent(
                    type="text",
                    text=f"Error executing {name}: {str(e)}"
                )]
                
    def _setup_resources(self):
        """Setup MCP resources for scenario analysis"""
        
        @self.server.list_resources()
        async def handle_list_resources() -> List[Resource]:
            """List available scenario analysis resources"""
            return [
                Resource(
                    uri="scenario-analysis://scenarios",
                    name="Scenarios",
                    description="Available scenarios and configurations",
                    mimeType="application/json"
                ),
                Resource(
                    uri="scenario-analysis://impacts",
                    name="Impact Analysis",
                    description="Impact analysis results and metrics",
                    mimeType="application/json"
                ),
                Resource(
                    uri="scenario-analysis://risks",
                    name="Risk Assessment",
                    description="Risk assessment results and mitigation strategies",
                    mimeType="application/json"
                ),
                Resource(
                    uri="scenario-analysis://comparisons",
                    name="Scenario Comparisons",
                    description="Scenario comparison results and rankings",
                    mimeType="application/json"
                )
            ]
            
        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> str:
            """Read scenario analysis resources"""
            try:
                if uri == "scenario-analysis://scenarios":
                    return json.dumps({
                        "scenarios": await self._get_scenarios_summary(),
                        "last_updated": datetime.now().isoformat()
                    })
                elif uri == "scenario-analysis://impacts":
                    return json.dumps({
                        "impacts": await self._get_impacts_summary(),
                        "last_updated": datetime.now().isoformat()
                    })
                elif uri == "scenario-analysis://risks":
                    return json.dumps({
                        "risks": await self._get_risks_summary(),
                        "last_updated": datetime.now().isoformat()
                    })
                elif uri == "scenario-analysis://comparisons":
                    return json.dumps({
                        "comparisons": await self._get_comparisons_summary(),
                        "last_updated": datetime.now().isoformat()
                    })
                else:
                    return json.dumps({"error": f"Unknown resource: {uri}"})
                    
            except Exception as e:
                logger.error(f"Error reading resource {uri}: {str(e)}")
                return json.dumps({"error": str(e)})
                
    async def _handle_create_scenario(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle scenario creation"""
        scenario_name = arguments.get("scenario_name")
        description = arguments.get("description", "")
        parameters = arguments.get("parameters", {})
        assumptions = arguments.get("assumptions", [])
        time_horizon = arguments.get("time_horizon", 12)
        
        # Create scenario
        scenario = await self.scenario_builder.create_scenario(
            name=scenario_name,
            description=description,
            parameters=parameters,
            assumptions=assumptions,
            time_horizon=time_horizon
        )
        
        result = {
            "scenario_id": str(uuid.uuid4()),
            "scenario_name": scenario_name,
            "description": description,
            "parameters": parameters,
            "assumptions": assumptions,
            "time_horizon": time_horizon,
            "status": "created",
            "created_at": datetime.now().isoformat()
        }
        
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
        
    async def _handle_analyze_impact(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle impact analysis"""
        scenario_id = arguments.get("scenario_id")
        impact_areas = arguments.get("impact_areas", ["financial", "operational", "strategic"])
        metrics = arguments.get("metrics", [])
        sensitivity_analysis = arguments.get("sensitivity_analysis", True)
        
        # Analyze impact
        impact_results = await self.impact_analyzer.analyze_impact(
            scenario_id=scenario_id,
            impact_areas=impact_areas,
            metrics=metrics,
            sensitivity_analysis=sensitivity_analysis
        )
        
        result = {
            "analysis_id": str(uuid.uuid4()),
            "scenario_id": scenario_id,
            "impact_areas": impact_areas,
            "metrics": metrics,
            "sensitivity_analysis": sensitivity_analysis,
            "results": impact_results,
            "summary": impact_results.get("summary", {}),
            "generated_at": datetime.now().isoformat()
        }
        
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
        
    async def _handle_assess_risk(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle risk assessment"""
        scenario_id = arguments.get("scenario_id")
        risk_categories = arguments.get("risk_categories", ["financial", "operational", "strategic", "compliance"])
        risk_tolerance = arguments.get("risk_tolerance", 0.5)
        
        # Assess risks
        risk_results = await self.risk_assessor.assess_risks(
            scenario_id=scenario_id,
            risk_categories=risk_categories,
            risk_tolerance=risk_tolerance
        )
        
        result = {
            "assessment_id": str(uuid.uuid4()),
            "scenario_id": scenario_id,
            "risk_categories": risk_categories,
            "risk_tolerance": risk_tolerance,
            "results": risk_results,
            "overall_risk_score": risk_results.get("overall_risk_score", 0.0),
            "risk_level": risk_results.get("risk_level", "medium"),
            "mitigation_strategies": risk_results.get("mitigation_strategies", []),
            "generated_at": datetime.now().isoformat()
        }
        
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
        
    async def _handle_compare_scenarios(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle scenario comparison"""
        scenario_ids = arguments.get("scenario_ids", [])
        comparison_metrics = arguments.get("comparison_metrics", [])
        ranking_criteria = arguments.get("ranking_criteria", {})
        
        # Compare scenarios
        comparison_results = await self.scenario_comparator.compare_scenarios(
            scenario_ids=scenario_ids,
            comparison_metrics=comparison_metrics,
            ranking_criteria=ranking_criteria
        )
        
        result = {
            "comparison_id": str(uuid.uuid4()),
            "scenario_ids": scenario_ids,
            "comparison_metrics": comparison_metrics,
            "ranking_criteria": ranking_criteria,
            "results": comparison_results,
            "rankings": comparison_results.get("rankings", []),
            "recommendations": comparison_results.get("recommendations", []),
            "generated_at": datetime.now().isoformat()
        }
        
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
        
    async def _handle_generate_recommendations(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle recommendation generation"""
        scenario_id = arguments.get("scenario_id")
        recommendation_type = arguments.get("recommendation_type", "action")
        priority_level = arguments.get("priority_level", "medium")
        
        # Generate recommendations
        recommendations = await self.scenario_builder.generate_recommendations(
            scenario_id=scenario_id,
            recommendation_type=recommendation_type,
            priority_level=priority_level
        )
        
        result = {
            "recommendation_id": str(uuid.uuid4()),
            "scenario_id": scenario_id,
            "recommendation_type": recommendation_type,
            "priority_level": priority_level,
            "recommendations": recommendations,
            "total_recommendations": len(recommendations),
            "generated_at": datetime.now().isoformat()
        }
        
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
        
    async def _handle_create_sensitivity_analysis(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle sensitivity analysis creation"""
        scenario_id = arguments.get("scenario_id")
        parameters = arguments.get("parameters", [])
        variation_range = arguments.get("variation_range", 20)
        steps = arguments.get("steps", 10)
        
        # Create sensitivity analysis
        sensitivity_results = await self.impact_analyzer.create_sensitivity_analysis(
            scenario_id=scenario_id,
            parameters=parameters,
            variation_range=variation_range,
            steps=steps
        )
        
        result = {
            "sensitivity_id": str(uuid.uuid4()),
            "scenario_id": scenario_id,
            "parameters": parameters,
            "variation_range": variation_range,
            "steps": steps,
            "results": sensitivity_results,
            "sensitivity_matrix": sensitivity_results.get("sensitivity_matrix", {}),
            "key_drivers": sensitivity_results.get("key_drivers", []),
            "generated_at": datetime.now().isoformat()
        }
        
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
        
    async def _handle_export_scenario_report(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle scenario report export"""
        scenario_id = arguments.get("scenario_id")
        report_format = arguments.get("report_format", "json")
        include_charts = arguments.get("include_charts", True)
        sections = arguments.get("sections", ["summary", "impact", "risk", "recommendations"])
        
        # Export report
        report_data = await self.scenario_builder.export_report(
            scenario_id=scenario_id,
            report_format=report_format,
            include_charts=include_charts,
            sections=sections
        )
        
        result = {
            "export_id": str(uuid.uuid4()),
            "scenario_id": scenario_id,
            "report_format": report_format,
            "include_charts": include_charts,
            "sections": sections,
            "report_url": report_data.get("report_url", ""),
            "file_size": report_data.get("file_size", 0),
            "exported_at": datetime.now().isoformat()
        }
        
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
        
    async def _get_scenarios_summary(self) -> Dict[str, Any]:
        """Get summary of available scenarios"""
        return {
            "total_scenarios": 0,
            "recent_scenarios": [],
            "scenario_types": ["business", "financial", "operational", "strategic"],
            "average_complexity": "medium"
        }
        
    async def _get_impacts_summary(self) -> Dict[str, Any]:
        """Get summary of impact analyses"""
        return {
            "total_analyses": 0,
            "recent_analyses": [],
            "impact_areas": ["financial", "operational", "strategic", "compliance"],
            "average_impact_score": 0.65
        }
        
    async def _get_risks_summary(self) -> Dict[str, Any]:
        """Get summary of risk assessments"""
        return {
            "total_assessments": 0,
            "recent_assessments": [],
            "risk_categories": ["financial", "operational", "strategic", "compliance"],
            "average_risk_score": 0.45
        }
        
    async def _get_comparisons_summary(self) -> Dict[str, Any]:
        """Get summary of scenario comparisons"""
        return {
            "total_comparisons": 0,
            "recent_comparisons": [],
            "comparison_types": ["pairwise", "multi_scenario", "ranking"],
            "average_comparison_score": 0.75
        }
        
    async def run(self):
        """Run the scenario analysis MCP server"""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="scenario-analysis",
                    server_version="1.0.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=None,
                        experimental_capabilities=None,
                    ),
                ),
            )


async def main():
    """Main entry point for the scenario analysis MCP server"""
    server = ScenarioAnalysisMCPServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
