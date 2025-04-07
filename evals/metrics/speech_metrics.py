"""
Speech recognition evaluation metrics for YuTalk.
"""

import logging
from pathlib import Path
from typing import Dict

import openai
import yaml
from dotenv import load_dotenv

from yutalk.utils import setup_logger

logger = setup_logger(name="speech_metrics", level=logging.INFO)

load_dotenv()
DEFAULT_SPEECH_MODEL = "whisper-1"

def character_error_rate(reference: str, hypothesis: str) -> float:
    """
    Calculate Character Error Rate (CER) between reference and hypothesis.
    
    CER = (S + D + I) / N
    Where:
    S = number of substitutions
    D = number of deletions
    I = number of insertions
    N = number of characters in reference
    """
    if not reference:
        return 1.0 if hypothesis else 0.0
    
    # Convert to list of characters
    r = list(reference)
    h = list(hypothesis)
    
    # Create a matrix to store the edit distances
    d = [[0 for _ in range(len(h) + 1)] for _ in range(len(r) + 1)]
    
    # Initialize the first row and column
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    
    # Compute the edit distance
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion = d[i][j-1] + 1
                deletion = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)
    
    # The last element contains the edit distance
    edit_distance = d[len(r)][len(h)]
    
    # Calculate CER
    cer = edit_distance / len(r)
    return cer


def transcribe_audio(audio_path: str, model: str = DEFAULT_SPEECH_MODEL) -> str:
    """
    Transcribe an audio file using OpenAI's Whisper API. Uses Chinese as the
    language argument.
    
    Args:
        audio_path: Path to the audio file
        model: Model name to use (defaults to whisper-1)
        
    Returns:
        Transcribed text
    """
    try:
        with open(audio_path, "rb") as audio_file:
            response = openai.audio.transcriptions.create(
                model=model,
                file=audio_file,
                language="zh"
            )
        return response.text
    except Exception as e:
        logger.error(f"Error transcribing audio {audio_path}: {str(e)}")
        return ""

def load_reference_transcripts(reference_file: str) -> Dict[str, str]:
    """
    Load reference transcripts from a YAML file.
    
    Args:
        reference_file: Path to the YAML file with reference transcripts
        
    Returns:
        Dictionary mapping audio filenames to reference transcripts
    """
    with open(reference_file, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    transcripts = {}
    for item in data.get('test_cases', []):
        filename = item.get('audio_file')
        transcript = item.get('transcript')
        if filename and transcript:
            transcripts[filename] = transcript
    
    return transcripts


def evaluate_speech_recognition(
    audio_dir: str, 
    reference_file: str, 
    model: str = DEFAULT_SPEECH_MODEL
) -> Dict:
    """
    Evaluate speech recognition on test audio files.
    
    Args:
        audio_dir: Directory containing audio test files
        reference_file: Path to the YAML file with reference transcripts
        model: Whisper model to use
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Load reference transcripts
    reference_transcripts = load_reference_transcripts(reference_file)
    
    results = {
        "overall": {
            "wer": 0.0,
            "cer": 0.0,
            "transcription_count": 0
        },
        "details": []
    }
    
    total_cer = 0.0
    total_count = 0
    
    audio_dir_path = Path(audio_dir)
    audio_files = list(audio_dir_path.glob("**/*.m4a"))
    
    for audio_file in audio_files:
        filename = audio_file.name
        
        # Skip if no reference transcript
        if filename not in reference_transcripts:
            logger.warning(f"No reference transcript for {filename}")
            continue
        
        reference = reference_transcripts[filename]
        
        # Transcribe audio and calculate metrics
        hypothesis = transcribe_audio(str(audio_file), model)

        current_cer = character_error_rate(reference, hypothesis)
        total_cer += current_cer
        total_count += 1
        
        file_result = {
            "filename": filename,
            "reference": reference,
            "hypothesis": hypothesis,
            "cer": current_cer
        }
        results["details"].append(file_result)
        
        logger.info(f"File: {filename}, CER: {current_cer:.4f}")
        logger.info(f"  Reference: {reference}")
        logger.info(f"  Hypothesis: {hypothesis}")
    
    if total_count > 0:
        results["overall"]["cer"] = total_cer / total_count
        results["overall"]["transcription_count"] = total_count
    
    return results


if __name__ == "__main__":
    # Example usage
    ref = "你好，我的名字是李明。"
    hyp = "你好，我的明子是李名。"
    
    cer = character_error_rate(ref, hyp)
    
    print(f"Reference: {ref}")
    print(f"Hypothesis: {hyp}")
    print(f"CER: {cer:.4f}")

    results = evaluate_speech_recognition(
        "evals/data/audio",
        "evals/data/audio_transcripts.yaml"
    )
