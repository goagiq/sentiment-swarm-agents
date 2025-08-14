# Batch Intelligence Analysis System

This system processes all questions from `intelligence_analysis_queries.md` systematically using MCP tools, with status tracking and user interaction between questions.

## Overview

The batch analysis system consists of three main components:

1. **`batch_intelligence_analysis.py`** - Main batch processor
2. **`run_batch_analysis.py`** - Simple runner script
3. **`check_batch_status.py`** - Status viewer and progress tracker

## Features

- ✅ **Automatic Question Extraction** - Parses all questions from the markdown file
- ✅ **Smart Tool Selection** - Automatically chooses appropriate MCP tools based on question content
- ✅ **Status Tracking** - Maintains progress across sessions
- ✅ **User Interaction** - Pauses between questions for review
- ✅ **Error Handling** - Continues processing even if individual questions fail
- ✅ **Result Storage** - Saves all results to JSON files
- ✅ **Progress Monitoring** - Real-time progress display

## Quick Start

### 1. Run the Batch Analysis

```bash
# Using the simple runner (recommended)
python run_batch_analysis.py

# Or directly with virtual environment
.venv/Scripts/python.exe batch_intelligence_analysis.py
```

### 2. Check Progress

```bash
python check_batch_status.py
```

### 3. Resume After Interruption

The system automatically resumes from where it left off. Simply run the batch analysis again:

```bash
python run_batch_analysis.py
```

## How It Works

### Question Extraction

The system automatically extracts all questions from `intelligence_analysis_queries.md` that follow the pattern:
```markdown
- **Question text here**
```

### Tool Selection Logic

The system intelligently selects MCP tools based on question keywords:

| Question Type | Keywords | MCP Tool |
|---------------|----------|----------|
| Strategic Intelligence | strategic principles, art of war, military strategy | `query_knowledge_graph` |
| Cultural Intelligence | cultural, bias, assumptions, values | `extract_entities` |
| Threat Assessment | threat, conflict, warfare, cyber | `query_knowledge_graph` |
| Language Analysis | language, translation, classical chinese | `process_content` |
| Scenario Analysis | scenario, what if, historical pattern | `generate_knowledge_graph` |
| Pattern Recognition | pattern, recurring, theme | `query_knowledge_graph` |
| Operational Intelligence | operational, intelligence gathering | `extract_entities` |
| Counterintelligence | deception, counterintelligence, warning | `analyze_sentiment` |
| Advanced Analytics | predictive, synthesis, cross-cultural | `analyze_business_intelligence` |

### Status Tracking

The system maintains a `batch_analysis_status.json` file with:

```json
{
  "start_time": "2024-01-01T12:00:00",
  "total_questions": 25,
  "completed_questions": 10,
  "current_question": 11,
  "questions": [
    {
      "id": 1,
      "section": "STRATEGIC INTELLIGENCE QUESTIONS",
      "subsection": "Specific Questions",
      "question": "What strategic principles from The Art of War...",
      "status": "completed",
      "started_at": "2024-01-01T12:00:00",
      "completed_at": "2024-01-01T12:05:00",
      "result": {...},
      "error": null
    }
  ]
}
```

### Result Storage

Each completed question generates a result file in `batch_analysis_results/`:

```
batch_analysis_results/
├── question_001_result.json
├── question_002_result.json
├── question_003_result.json
└── ...
```

## File Structure

```
├── batch_intelligence_analysis.py    # Main batch processor
├── run_batch_analysis.py             # Simple runner script
├── check_batch_status.py             # Status viewer
├── batch_analysis_status.json        # Progress tracking (auto-generated)
├── batch_analysis_results/           # Results directory (auto-generated)
│   ├── question_001_result.json
│   ├── question_002_result.json
│   └── ...
└── intelligence_analysis_queries.md  # Source questions
```

## Usage Examples

### Starting a New Analysis

```bash
# This will extract all questions and start processing
python run_batch_analysis.py
```

### Checking Progress

```bash
# View current status and recent results
python check_batch_status.py
```

### Resuming After Break

```bash
# The system will automatically skip completed questions
python run_batch_analysis.py
```

## Status Viewer Options

When running `check_batch_status.py`, you can:

1. **Show all question details** - Complete information for all questions
2. **Show only completed questions** - Review successful results
3. **Show only error questions** - Identify and troubleshoot issues
4. **Exit** - Return to command line

## Error Handling

- **Individual Question Failures**: The system continues processing other questions
- **Tool Selection Errors**: Falls back to business intelligence analysis
- **File I/O Errors**: Graceful handling with error messages
- **MCP Tool Errors**: Captured and stored in question status

## Customization

### Adding New Question Types

To add support for new question types, modify the `get_mcp_tool_for_question()` method in `batch_intelligence_analysis.py`:

```python
# Add new condition
elif any(keyword in question_lower for keyword in ['your_keywords']):
    return "your_mcp_tool", {
        "param1": "value1",
        "param2": "value2"
    }
```

### Modifying Tool Parameters

Adjust the tool parameters in the same method to customize analysis behavior.

### Changing Status File Location

Modify the `status_file` attribute in the `IntelligenceAnalysisBatchProcessor` class.

## Troubleshooting

### Common Issues

1. **"No questions found"**
   - Ensure `intelligence_analysis_queries.md` exists and contains questions in the correct format
   - Check that questions start with `- **` and end with `**`

2. **"Virtual environment not found"**
   - Ensure `.venv/Scripts/python.exe` exists
   - Run `python -m venv .venv` if needed

3. **"MCP tool import errors"**
   - Ensure all MCP tools are properly installed
   - Check that the virtual environment is activated

4. **"Permission denied" errors**
   - Ensure write permissions in the current directory
   - Check that `batch_analysis_results/` directory can be created

### Debug Mode

To see detailed processing information, the system provides verbose output during execution.

## Performance Considerations

- **Processing Time**: Each question typically takes 10-30 seconds depending on complexity
- **Memory Usage**: Results are stored in JSON format to minimize memory footprint
- **Resume Capability**: The system can handle interruptions and resume efficiently
- **Parallel Processing**: Currently sequential for reliability, but can be modified for parallel processing

## Best Practices

1. **Regular Status Checks**: Use `check_batch_status.py` to monitor progress
2. **Backup Results**: Copy `batch_analysis_results/` directory for safekeeping
3. **Review Errors**: Check error questions to understand and fix issues
4. **Incremental Processing**: Process questions in batches for better control

## Support

For issues or questions about the batch analysis system:

1. Check the troubleshooting section above
2. Review the status file for error details
3. Examine individual result files for specific issues
4. Check the console output for detailed error messages

---

**Note**: This system is designed for intelligence analysis workflows and provides comprehensive tracking and error handling for long-running batch processes.
