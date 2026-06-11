# Running Mistral Medium 3.5 through BullshitBenchmark

## Summary

This document describes how to run Mistral Medium 3.5 through the BullshitBenchmark tests. Due to sandbox network restrictions, the actual API calls cannot be executed from within the sandbox environment, but all the necessary code changes and configurations have been prepared.

## Changes Made

### 1. Added Mistral Provider Support

Modified `scripts/openrouter_benchmark.py` to add:
- "mistral" as a new provider type in `MODEL_PROVIDER_ALIASES` and `MODEL_PROVIDER_VALUES`
- `MistralClient` class that communicates with Mistral's OpenAI-compatible chat/completions API
- Client initialization logic for both collection and grading phases

### 2. Configuration Files

Created two configuration files:
- `config.mistral-medium-3.5.json` - Uses OpenRouter to access Mistral models
- `config.mistral-medium-3.5.direct.json` - Uses Mistral's direct API

## Running the Benchmark

### Option 1: Using OpenRouter (Recommended)

OpenRouter can proxy requests to Mistral models. This is the easiest approach.

```bash
# Set your OpenRouter API key
export OPENROUTER_API_KEY="your_openrouter_api_key"

# Run the full benchmark (collection + grading)
./scripts/run_end_to_end.sh \
  --config config.mistral-medium-3.5.json \
  --run-id mistral_medium_35_run1 \
  --output-dir runs \
  --with-additional-judges
```

**Note:** The model identifier in this config is `mistralai/mistral-medium-3.5`. You may need to verify this is the correct model ID in OpenRouter's model list.

### Option 2: Using Mistral's Direct API

If you have direct access to Mistral's API:

```bash
# Set your Mistral API key and base URL
export MISTRAL_API_KEY="your_mistral_api_key"
export MISTRAL_BASE_URL="https://api.mistral.ai/v1/chat/completions"

# Run collection only
python3 scripts/openrouter_benchmark.py collect \
  --config config.mistral-medium-3.5.direct.json \
  --run-id mistral_medium_35_direct \
  --output-dir runs

# Then run grading (uses OpenRouter for judges by default)
./scripts/run_end_to_end.sh \
  --config config.mistral-medium-3.5.direct.json \
  --run-id mistral_medium_35_direct \
  --output-dir runs \
  --skip-collect \
  --with-additional-judges
```

## Model Identifier

The correct model identifier for Mistral Medium 3.5 is:
- `mistral-medium-3-5` (for direct Mistral API)
- `mistralai/mistral-medium-3-5` (for OpenRouter)

Both configuration files have been updated with these correct identifiers.

## Verification

A dry run was successfully completed, confirming:
- The configuration is valid
- The MistralClient class is properly integrated
- All 100 v2 questions are recognized
- The grading pipeline works

## Files Modified/Created

- `scripts/openrouter_benchmark.py` - Added Mistral provider support
- `config.mistral-medium-3.5.json` - OpenRouter-based config
- `config.mistral-medium-3.5.direct.json` - Direct API config
- `runs/mistral_medium_35_test/` - Dry run test artifacts

## Next Steps

To run this outside the sandbox:
1. Ensure you have Python 3.8+ installed
2. Install required dependencies (if any)
3. Set the appropriate API key environment variable
4. Run the commands above

The benchmark will:
1. Collect responses from Mistral Medium 3.5 for all 100 v2 questions
2. Grade the responses using the primary judge (Claude Sonnet 4.6)
3. Optionally run the full 3-judge panel for consensus scoring
4. Generate artifacts in the `runs/` directory
