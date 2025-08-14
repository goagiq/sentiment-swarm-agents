# Comprehensive Test Plan for MCP Tools and APIs
## Using The Art of War and Classical Chinese Grammar Books

### Overview
This test plan utilizes two books stored in your vector and knowledge graph database:
1. **The Art of War (孫子兵法)** - Classical Chinese military treatise by Sun Tzu
2. **Classical Chinese Grammar Book** - Functional approach to classical Chinese by Kai Li and James Erwin Dew

The plan tests all major features and functionalities of your sentiment analysis system, including MCP tools, APIs, decision support systems, and advanced analytics capabilities.

---

## Phase 1: Basic System Health and Connectivity Tests

### Test 1.1: System Status Check
**Objective**: Verify system is running and accessible

**Steps**:
1. Start the system: `python main.py`
2. Check API health: `curl http://localhost:8000/health`
3. Verify MCP server status: `curl http://localhost:8000/mcp/status`
4. Check available tools: `curl http://localhost:8000/tools/list`

**Expected Results**: All endpoints should return 200 OK with system status information.

---

## Phase 2: Content Analysis and Sentiment Processing

### Test 2.1: Text Sentiment Analysis
**Objective**: Test basic sentiment analysis on book content

**Steps**:
1. Extract a passage from The Art of War (Chapter 1):
   ```
   "兵者，國之大事，死生之地，存亡之道，不可不察也。"
   (War is a matter of vital importance to the state; a matter of life and death, the road either to survival or to ruin.)
   ```

2. Send to sentiment analysis API:
   ```bash
   curl -X POST http://localhost:8000/analyze/text \
     -H "Content-Type: application/json" \
     -d '{
       "content": "兵者，國之大事，死生之地，存亡之道，不可不察也。",
       "language": "zh",
       "analysis_type": "comprehensive"
     }'
   ```

**Expected Results**: Should return sentiment analysis with confidence scores and metadata.

### Test 2.2: Multilingual Content Processing
**Objective**: Test processing of mixed Chinese-English content

**Steps**:
1. Use content from the Classical Chinese Grammar book that contains both languages
2. Test with language detection enabled
3. Verify translation capabilities work

**API Call**:
```bash
curl -X POST http://localhost:8000/analyze/text \
  -H "Content-Type: application/json" \
  -d '{
    "content": "文言章句 Classical Chinese A Functional Approach",
    "language": "auto",
    "include_translation": true
  }'
```

---

## Phase 3: Vector Database and Semantic Search

### Test 3.1: Vector Database Query
**Objective**: Test semantic search capabilities

**Steps**:
1. Search for military strategy concepts:
   ```bash
   curl -X POST http://localhost:8000/search/semantic \
     -H "Content-Type: application/json" \
     -d '{
       "query": "military strategy and tactics",
       "n_results": 5,
       "similarity_threshold": 0.7
     }'
   ```

2. Search for classical Chinese grammar concepts:
   ```bash
   curl -X POST http://localhost:8000/search/semantic \
     -H "Content-Type: application/json" \
     -d '{
       "query": "classical Chinese grammar patterns",
       "n_results": 5,
       "similarity_threshold": 0.7
     }'
   ```

**Expected Results**: Should return relevant passages from both books with similarity scores.

### Test 3.2: Knowledge Graph Queries
**Objective**: Test knowledge graph functionality

**Steps**:
1. Query for entities related to military strategy:
   ```bash
   curl -X POST http://localhost:8000/kg/search \
     -H "Content-Type: application/json" \
     -d '{
       "query": "Sun Tzu military principles",
       "entity_types": ["person", "concept", "strategy"]
     }'
   ```

2. Query for linguistic concepts:
   ```bash
   curl -X POST http://localhost:8000/kg/search \
     -H "Content-Type: application/json" \
     -d '{
       "query": "classical Chinese grammar structures",
       "entity_types": ["concept", "grammar", "pattern"]
     }'
   ```

---

## Phase 4: Advanced Analytics and Pattern Recognition

### Test 4.1: Pattern Recognition in Text
**Objective**: Test pattern recognition capabilities

**Steps**:
1. Analyze recurring themes in The Art of War:
   ```bash
   curl -X POST http://localhost:8000/analyze/patterns \
     -H "Content-Type: application/json" \
     -d '{
       "content_source": "art_of_war",
       "pattern_types": ["themes", "strategies", "principles"],
       "min_frequency": 3
     }'
   ```

2. Analyze grammatical patterns in the Chinese grammar book:
   ```bash
   curl -X POST http://localhost:8000/analyze/patterns \
     -H "Content-Type: application/json" \
     -d '{
       "content_source": "classical_chinese_grammar",
       "pattern_types": ["grammar_rules", "sentence_structures", "particles"],
       "min_frequency": 2
     }'
   ```

### Test 4.2: Anomaly Detection
**Objective**: Test anomaly detection in content

**Steps**:
1. Detect unusual patterns or outliers in the texts
2. Identify content that doesn't fit typical patterns

**API Call**:
```bash
curl -X POST http://localhost:8000/analyze/anomalies \
  -H "Content-Type: application/json" \
  -d '{
    "content_sources": ["art_of_war", "classical_chinese_grammar"],
    "anomaly_types": ["content", "style", "structure"]
  }'
```

---

## Phase 5: Decision Support Systems

### Test 5.1: Action Prioritization
**Objective**: Test decision support with military strategy scenarios

**Steps**:
1. Create a business scenario based on Art of War principles:
   ```bash
   curl -X POST http://localhost:8000/decision/prioritize \
     -H "Content-Type: application/json" \
     -d '{
       "scenario": "A company is entering a new competitive market. Based on Sun Tzu principles, what actions should be prioritized?",
       "context": "market_entry_strategy",
       "available_resources": {"budget": 1000000, "time": "6_months", "team_size": 10},
       "constraints": ["regulatory_compliance", "ethical_business_practices"]
     }'
   ```

2. Test with linguistic learning scenario:
   ```bash
   curl -X POST http://localhost:8000/decision/prioritize \
     -H "Content-Type: application/json" \
     -d '{
       "scenario": "A student wants to learn classical Chinese efficiently. What learning strategies should be prioritized?",
       "context": "language_learning",
       "available_resources": {"time": "1_year", "study_hours": "10_per_week"},
       "constraints": ["modern_learning_methods", "practical_application"]
     }'
   ```

### Test 5.2: Scenario Analysis
**Objective**: Test scenario analysis capabilities

**Steps**:
1. Analyze "what-if" scenarios for military strategy:
   ```bash
   curl -X POST http://localhost:8000/decision/scenarios \
     -H "Content-Type: application/json" \
     -d '{
       "base_scenario": "Applying Sun Tzu principles to modern business competition",
       "variations": [
         {"factor": "market_conditions", "value": "recession"},
         {"factor": "competitor_strength", "value": "strong"},
         {"factor": "resource_availability", "value": "limited"}
       ],
       "analysis_depth": "comprehensive"
     }'
   ```

2. Analyze language learning scenarios:
   ```bash
   curl -X POST http://localhost:8000/decision/scenarios \
     -H "Content-Type: application/json" \
     -d '{
       "base_scenario": "Learning classical Chinese grammar",
       "variations": [
         {"factor": "learning_style", "value": "visual"},
         {"factor": "time_availability", "value": "intensive"},
         {"factor": "prior_knowledge", "value": "none"}
       ],
       "analysis_depth": "detailed"
     }'
   ```

---

## Phase 6: Cross-Modal Analysis and Integration

### Test 6.1: Cross-Book Analysis
**Objective**: Test analysis across both books

**Steps**:
1. Compare themes and concepts between the books:
   ```bash
   curl -X POST http://localhost:8000/analyze/cross_modal \
     -H "Content-Type: application/json" \
     -d '{
       "content_sources": ["art_of_war", "classical_chinese_grammar"],
       "analysis_type": "comparative",
       "focus_areas": ["language_patterns", "cultural_context", "structural_principles"]
     }'
   ```

2. Find connections between military strategy and language structure:
   ```bash
   curl -X POST http://localhost:8000/analyze/connections \
     -H "Content-Type: application/json" \
     -d '{
       "source_concepts": ["military_strategy", "tactical_planning"],
       "target_concepts": ["grammar_structure", "linguistic_patterns"],
       "connection_types": ["analogical", "structural", "philosophical"]
     }'
   ```

### Test 6.2: Knowledge Integration
**Objective**: Test knowledge graph integration

**Steps**:
1. Build integrated knowledge graph:
   ```bash
   curl -X POST http://localhost:8000/kg/integrate \
     -H "Content-Type: application/json" \
     -d '{
       "sources": ["art_of_war", "classical_chinese_grammar"],
       "integration_type": "comprehensive",
       "relationship_types": ["semantic", "thematic", "structural"]
     }'
   ```

---

## Phase 7: Advanced MCP Tools Testing

### Test 7.1: MCP Tool Execution
**Objective**: Test individual MCP tools

**Steps**:
1. Test sentiment analysis tool:
   ```bash
   curl -X POST http://localhost:8000/mcp/execute \
     -H "Content-Type: application/json" \
     -d '{
       "tool": "analyze_sentiment",
       "parameters": {
         "text": "知己知彼，百戰不殆。",
         "language": "zh"
       }
     }'
   ```

2. Test knowledge graph tool:
   ```bash
   curl -X POST http://localhost:8000/mcp/execute \
     -H "Content-Type: application/json" \
     -d '{
       "tool": "query_knowledge_graph",
       "parameters": {
         "query": "military strategy principles",
         "max_results": 10
       }
     }'
   ```

3. Test pattern recognition tool:
   ```bash
   curl -X POST http://localhost:8000/mcp/execute \
     -H "Content-Type: application/json" \
     -d '{
       "tool": "detect_patterns",
       "parameters": {
         "content": "art_of_war",
         "pattern_type": "strategic_principles"
       }
     }'
   ```

### Test 7.2: MCP Tool Chaining
**Objective**: Test multiple MCP tools working together

**Steps**:
1. Chain sentiment analysis → pattern detection → knowledge graph:
   ```bash
   curl -X POST http://localhost:8000/mcp/chain \
     -H "Content-Type: application/json" \
     -d '{
       "workflow": [
         {
           "tool": "analyze_sentiment",
           "parameters": {"text": "兵貴神速", "language": "zh"}
         },
         {
           "tool": "extract_entities",
           "parameters": {"content": "previous_result"}
         },
         {
           "tool": "query_knowledge_graph",
           "parameters": {"entities": "previous_result"}
         }
       ]
     }'
   ```

---

## Phase 8: Business Intelligence and Reporting

### Test 8.1: Executive Summary Generation
**Objective**: Test business intelligence capabilities

**Steps**:
1. Generate executive summary of Art of War principles:
   ```bash
   curl -X POST http://localhost:8000/business/summary \
     -H "Content-Type: application/json" \
     -d '{
       "content_source": "art_of_war",
       "summary_type": "executive",
       "focus_areas": ["strategic_principles", "leadership", "competition"],
       "target_audience": "business_executives"
     }'
   ```

2. Generate learning strategy report:
   ```bash
   curl -X POST http://localhost:8000/business/summary \
     -H "Content-Type: application/json" \
     -d '{
       "content_source": "classical_chinese_grammar",
       "summary_type": "educational",
       "focus_areas": ["learning_methods", "grammar_structure", "cultural_context"],
       "target_audience": "language_learners"
     }'
   ```

### Test 8.2: Risk Assessment
**Objective**: Test risk analysis capabilities

**Steps**:
1. Assess risks in applying ancient military principles to modern business:
   ```bash
   curl -X POST http://localhost:8000/business/risk_assessment \
     -H "Content-Type: application/json" \
     -d '{
       "scenario": "Applying Sun Tzu principles to modern business",
       "risk_factors": ["cultural_differences", "ethical_considerations", "regulatory_compliance"],
       "assessment_depth": "comprehensive"
     }'
   ```

---

## Phase 9: Performance and Scalability Testing

### Test 9.1: Load Testing
**Objective**: Test system performance under load

**Steps**:
1. Send multiple concurrent requests to different endpoints
2. Monitor response times and system resources
3. Test with large content chunks

**API Call**:
```bash
# Test with larger content
curl -X POST http://localhost:8000/analyze/text \
  -H "Content-Type: application/json" \
  -d '{
    "content": "[LARGE_EXTRACT_FROM_ART_OF_WAR]",
    "analysis_type": "comprehensive",
    "include_patterns": true,
    "include_entities": true
  }'
```

### Test 9.2: Caching Performance
**Objective**: Test caching mechanisms

**Steps**:
1. Send identical requests multiple times
2. Verify response times improve with caching
3. Test cache invalidation

---

## Phase 10: Error Handling and Edge Cases

### Test 10.1: Error Scenarios
**Objective**: Test system robustness

**Steps**:
1. Send malformed requests
2. Test with unsupported languages
3. Test with empty content
4. Test with extremely long content

**API Calls**:
```bash
# Test malformed request
curl -X POST http://localhost:8000/analyze/text \
  -H "Content-Type: application/json" \
  -d '{"invalid_field": "test"}'

# Test empty content
curl -X POST http://localhost:8000/analyze/text \
  -H "Content-Type: application/json" \
  -d '{"content": ""}'
```

### Test 10.2: Language Edge Cases
**Objective**: Test language processing edge cases

**Steps**:
1. Test mixed language content
2. Test with rare characters
3. Test language detection accuracy

---

## Phase 11: Integration Testing

### Test 11.1: End-to-End Workflows
**Objective**: Test complete workflows

**Steps**:
1. Complete workflow: Content ingestion → Analysis → Knowledge Graph → Decision Support
2. Test feedback loops and iterative improvement
3. Test data persistence and retrieval

### Test 11.2: External Integration
**Objective**: Test external system integration

**Steps**:
1. Test with external translation services
2. Test with external knowledge bases
3. Test with external analytics tools

---

## Phase 12: Advanced Decision Support Scenarios

### Test 12.1: Strategic Planning Scenarios
**Objective**: Test advanced decision support with real-world scenarios

**Steps**:
1. **Market Entry Strategy** (based on Art of War):
   ```bash
   curl -X POST http://localhost:8000/decision/comprehensive \
     -H "Content-Type: application/json" \
     -d '{
       "scenario": "A tech startup is entering a market dominated by large corporations",
       "context": {
         "industry": "technology",
         "market_conditions": "competitive",
         "resources": "limited",
         "timeline": "12_months"
       },
       "analysis_requirements": [
         "competitive_analysis",
         "resource_allocation",
         "risk_assessment",
         "timeline_planning"
       ],
       "knowledge_sources": ["art_of_war", "business_strategy"]
     }'
   ```

2. **Educational Program Design** (based on Classical Chinese Grammar):
   ```bash
   curl -X POST http://localhost:8000/decision/comprehensive \
     -H "Content-Type: application/json" \
     -d '{
       "scenario": "Designing a classical Chinese learning program for university students",
       "context": {
         "audience": "university_students",
         "duration": "semester",
         "prerequisites": "basic_Chinese",
         "goals": "reading_classical_texts"
       },
       "analysis_requirements": [
         "curriculum_design",
         "learning_objectives",
         "assessment_methods",
         "resource_requirements"
       ],
       "knowledge_sources": ["classical_chinese_grammar", "pedagogy"]
     }'
   ```

### Test 12.2: Crisis Management Scenarios
**Objective**: Test decision support in crisis situations

**Steps**:
1. **Business Crisis Response**:
   ```bash
   curl -X POST http://localhost:8000/decision/crisis \
     -H "Content-Type: application/json" \
     -d '{
       "crisis_type": "competitive_threat",
       "situation": "Major competitor launching aggressive pricing strategy",
       "time_constraints": "immediate_response_required",
       "available_resources": "limited",
       "stakeholders": ["customers", "employees", "investors"],
       "knowledge_sources": ["art_of_war", "crisis_management"]
     }'
   ```

---

## Phase 13: Validation and Quality Assurance

### Test 13.1: Result Validation
**Objective**: Validate analysis results for accuracy

**Steps**:
1. Compare sentiment analysis results with expert human analysis
2. Validate knowledge graph relationships
3. Verify pattern recognition accuracy
4. Test decision support recommendations against known good practices

### Test 13.2: Consistency Testing
**Objective**: Test result consistency across different runs

**Steps**:
1. Run identical analyses multiple times
2. Compare results for consistency
3. Test with different model configurations
4. Verify caching doesn't affect result quality

---

## Phase 14: Documentation and Reporting

### Test 14.1: Report Generation
**Objective**: Test comprehensive report generation

**Steps**:
1. Generate detailed analysis reports
2. Test different report formats (JSON, PDF, HTML)
3. Test report customization options
4. Verify report accuracy and completeness

### Test 14.2: API Documentation Testing
**Objective**: Test API documentation accuracy

**Steps**:
1. Verify OpenAPI/Swagger documentation
2. Test all documented endpoints
3. Validate request/response schemas
4. Test error response documentation

---

## Success Criteria

### Functional Success Criteria:
- ✅ All API endpoints respond correctly
- ✅ MCP tools execute without errors
- ✅ Decision support provides actionable recommendations
- ✅ Knowledge graph queries return relevant results
- ✅ Pattern recognition identifies meaningful patterns
- ✅ Cross-modal analysis works across both books

### Performance Success Criteria:
- ✅ Response times under 5 seconds for standard requests
- ✅ System handles concurrent requests without degradation
- ✅ Memory usage remains stable during extended testing
- ✅ Caching improves response times for repeated requests

### Quality Success Criteria:
- ✅ Sentiment analysis accuracy > 85%
- ✅ Knowledge graph relationships are logically sound
- ✅ Decision support recommendations are contextually appropriate
- ✅ Error handling provides meaningful feedback
- ✅ Results are consistent across multiple runs

---

## Notes for Test Execution

1. **Environment Setup**: Ensure the system is running with all services started
2. **Data Preparation**: Verify both books are properly loaded in the vector database
3. **Monitoring**: Use the monitoring dashboard to track system performance
4. **Documentation**: Record any unexpected behaviors or errors
5. **Iteration**: Some tests may need to be run multiple times to verify consistency

This test plan provides comprehensive coverage of your system's capabilities while utilizing the rich content from both books to create meaningful and challenging test scenarios.

