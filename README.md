# Canadian News Media Attitudes Analysis with LLM API

This repository contains an analysis of Canadian news media's attitudes toward economic and social issues using Large Language Models (LLMs). The study evaluates whether Canadian news media has shifted over time toward favoring government intervention in the economy and examines attitudes toward free speech. Results are presented as time-series graphs and include an exploration of model temperature sensitivity and article relevance for economists.

## Project Overview
This project aims to address the claim by high-profile figures that Canada is shifting toward communism. By leveraging LLM APIs, we quantitatively analyze Canadian news articles for attitudes on:
1. Markets vs. government intervention.
2. Free speech vs. censorship.

### Tasks Performed
1. **Economic Attitude Analysis**:
   - Articles scored from 0 (pro-markets) to 1 (pro-government intervention).
   - Time-series graph generated to track changes over time.
2. **Free Speech Attitude Analysis**:
   - Articles scored from 0 (friendly to censorship) to 1 (friendly to free speech).
   - Time-series graph generated to visualize trends.
3. **Temperature Sensitivity Analysis**:
   - Examined how model "temperature" affects economic attitude scores.
4. **Article Relevance Analysis**:
   - Scored 20 articles from a Canadian student newspaper to identify the most and least relevant for economists.

### Key Components
- **Python Script**: Automates API calls, processes data, and generates results.
- **Results PDF**: Includes time-series graphs, temperature sensitivity analysis, and article relevance scores.

### Files in Repository
1. `analysis_script.py`: Python script for API calls and data processing.
2. `results.pdf`: Contains detailed results, graphs, and insights.
3. `README.md`: Overview of the project and instructions for usage.

### How to Run the Analysis
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Canadian_News_Media_LLM_Analysis.git
