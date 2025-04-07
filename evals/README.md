# YuTalk Evaluation System

This directory contains evaluation tools for measuring the performance of the YuTalk Chinese language learning chatbot.

TODO(2025-04-07): Make the test cases more difficult. Currently they give 100% accuracy. I need to make it more discriminatory between good/bad AI systems. Perhaps from some of my own conversations online?


## Overview

The evaluation system is designed to assess three key aspects of YuTalk:

1. **Speech Recognition Quality**: How accurately the system transcribes Chinese speech.
2. **Grammar Correction Accuracy**: How well the system identifies and corrects grammar errors.
3. **Conversation Quality**: How natural, engaging, and pedagogically valuable the conversation is.

## Running Evaluations

### Basic Usage

To run all evaluations with default settings:

```bash
python run_evals.py
```

### Creating a Baseline

Before running regular evaluations, it's recommended to create a baseline:

```bash
python run_evals.py --create-baseline
```

This will run all evaluations and store the results as a reference point for future comparisons.

### Running Specific Evaluations

To run only specific types of evaluations:

```bash
python run_evals.py --speech-only    # Speech recognition only
python run_evals.py --grammar-only   # Grammar correction only
python run_evals.py --conversation-only  # Conversation quality only
```

### Additional Options

```bash
python run_evals.py --help  # View all available options
```

Common options include:
- `--config PATH`: Use a custom configuration file
- `--output-dir PATH`: Specify where to save evaluation results
- `--run-name NAME`: Give this evaluation run a specific name
- `--no-compare`: Don't compare results with baseline
- `--force`: Force overwrite of existing baseline

## Configuration

The `config.yaml` file controls which evaluations are run and their parameters. You can modify this file to:
- Enable/disable specific evaluation types
- Change model parameters
- Adjust which metrics are calculated

## Test Data

### Audio Test Files

Place audio test files in the `data/audio/` directory. The files should contain spoken Chinese for evaluation.

### Reference Transcripts

Reference transcripts are defined in YAML files in the `data/audio_transcripts.yaml` file. This maps audio filenames to their correct transcriptions.

### Grammar Test Cases

Grammar test cases are defined in `data/grammar_cases.yaml`. Each test case includes a Chinese sentence and the expected grammar checking result.

### Conversation Test Cases

Conversation test cases are defined in `data/conversation_cases.yaml`. These provide sample user messages that are sent to the chatbot for evaluation.

## Adding New Test Cases

### Adding Audio Test Files

1. Add the audio file to `data/audio/`
2. Add the reference transcript to `data/audio_transcripts.yaml`

### Adding Grammar Test Cases

Add new entries to `data/grammar_cases.yaml` following the existing format:

```yaml
- category: word_order
  text: "我每天吃饭三次。"
  expected_result:
    has_errors: true
    errors:
      - type: word_order
        description: "Frequency should be placed before the object"
```

### Adding Conversation Test Cases

Add new entries to `data/conversation_cases.yaml` following the existing format:

```yaml
- user_message: "我喜欢打篮球，你呢？"
  skill_level: beginner
  description: "Sharing a hobby and asking the assistant"
```

## Results and Analysis

Evaluation results are saved to the `results/` directory. Each run creates a new subdirectory with:

- `results.json`: The complete evaluation metrics
- `comparison.json`: Comparison with the baseline (if available)

## Interpreting Results

### Speech Recognition Metrics

- **CER (Character Error Rate)**: Lower is better. Measures percentage of incorrect characters.

### Grammar Correction Metrics

- **Precision**: Higher is better. Percentage of correctly identified errors out of all detected errors.
- **Recall**: Higher is better. Percentage of correctly identified errors out of all actual errors.
- **F1 Score**: Higher is better. Harmonic mean of precision and recall.
- **Accuracy**: Higher is better. Overall percentage of correct assessments.

### Conversation Quality Metrics

All conversation metrics are rated on a scale of 1-5, with higher being better:

- **Coherence**: How well responses follow from user input.
- **Relevance**: How relevant responses are to user messages.
- **Engagement**: How well responses encourage continued conversation.
- **Language Level**: How well responses match the user's proficiency level.
- **Pedagogical Value**: How valuable responses are for language learning.