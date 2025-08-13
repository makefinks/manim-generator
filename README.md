# AI - Manim Video Generator
Automatic video generation using an agentic LLM flow in combination with the [manim](https://www.manim.community/) python library.

## 🌟 Inspiration
Following the release of Deepseek-R1, users found it was particularly adept at generating manim code for mathematical animations.
A good example of this ability can be seen in this [X post](https://x.com/christiancooper/status/1881366695178367069) and in the repository related to that post: [
Math-To-Manim](https://github.com/HarleyCoops/Math-To-Manim)

## Goal
The Goal of this project is to experiment with the automatic creation of manim based videos. Based on any input data / information you want to visualize.

As of now the project consist of an (agentic) flow with one `Code Reviewer` and one `Code Writer` that interact in a feedback loop. 

Manim code generated in a one shot fashion often contains errors or has visual inconsistency. The approach in this project tries to reduce the error rate and improve the quality of the videos through better prompting, feedback loops and the use VLMs (Image Capable LLMs).

The use of [LiteLLM](https://github.com/BerriAI/litellm) in the project ensures that future models can also be easily benchmarked and compared on this task.

Benchmarking future model releases on this task will also be interesting, since it requires good spatial understanding which most LLMs struggle with.

> ❗ It's important to note that creating a coherent video that compiles and has no issues is still pushing the current SOTA LLMs to their limits

## Current Flow (subject to change)
![Creation flow](images/flow.png)

## 🚀 Installation

### 1. Clone the repository:
```bash
git clone https://github.com/makefinks/manim-generator.git
cd manim-generator
```
### 2. Install the requirements
With [uv](https://github.com/astral-sh/uv) (Recommended)
```bash
uv sync
```
> Note: `uv` offers faster installation times and better dependency resolution

Or using the requirements.txt directly
```bash
pip install -r requirements.txt
```

### You also need to install ffmpeg and for some cases a LaTeX distribution

Windows (using Chocolatey):
```bash
choco install ffmpeg
choco install miktex
```

macOS (using Homebrew):
```bash
brew install ffmpeg
brew install --cask mactex
```

Linux (Debian/Ubuntu):
```bash
sudo apt-get update
sudo apt-get install texlive texlive-latex-extra texlive-fonts-extra texlive-science
```

### 3. Setup Environment Variables
Either by creating a `.env` file or by defining the API key of your preferred provider system wide.
> Recommendation: use [openrouter](https://openrouter.ai/) to access all llms 

## ⚡ Usage
### 1. Execute the script
```bash
python generate.py
```

### 2. CLI Arguments
The script supports the following command-line arguments:

#### Video Data Input
| Argument | Description | Default |
|---------|-------------|---------|
| `--video_data` | Description of the video to generate (text string) | - |
| `--video_data_file` | Path to file containing video description | "video_data.txt" |

#### Model Configuration
| Argument | Description | Default |
|---------|-------------|---------|
| `--manim_model` | Model to use for generating Manim code | "openrouter/anthropic/claude-sonnet-4" |
| `--review_model` | Model to use for reviewing code | "openrouter/anthropic/claude-sonnet-4" |
| `--streaming` | Enable streaming responses from the model | False |
| `--temperature` | Temperature for the LLM Model | 0.4 |
| `--force_vision` | Adds images to the review process, regardless if LiteLLM reports vision is not supported | - |
| `--provider` | Specific provider to use for OpenRouter requests (e.g., 'anthropic', 'openai') | - |
#### Process Configuration
| Argument | Description | Default |
|---------|-------------|---------|
| `--review_cycles` | Number of review cycles to perform | 5 |
| `--manim_logs` | Show Manim execution logs | False |
| `--success_threshold` | Percentage of scenes that must render successfully to trigger enhanced visual review mode | 100 |
| `--frame_extraction_mode` | Frame extraction mode: highest_density (single best frame) or fixed_count (multiple frames) | "fixed_count" |
| `--frame_count` | Number of frames to extract when using fixed_count mode | 3 |

#### Reasoning Tokens Configuration
| Argument | Description | Default |
|---------|-------------|---------|
| `--reasoning_effort` | Reasoning effort level for OpenAI-style models (choices: "minimal", "low", "medium", "high") | - |
| `--reasoning_max_tokens` | Maximum tokens for reasoning (Anthropic-style) | - |
| `--reasoning_exclude` | Exclude reasoning tokens from response (model still uses reasoning internally) | - |

> Note: You cannot use both `--reasoning_effort` and `--reasoning_max_tokens` at the same time.

### Example with specific models and video data:
```bash
python generate.py --video_data "Explain the concept of neural networks with visual examples" --manim_model "openrouter/anthropic/claude-sonnet-4" --review_model "openrouter/anthropic/claude-sonnet-4" --review_cycles 3
```

### 3. Tweak the config

Images will only be used if the reviewer model supports image inputs. 
####  Check compatibility:
- https://openrouter.ai/models?modality=text+image-%3Etext
- https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json (can be tweaked manually if necessary)


## 🤝 Contributing
Contributions are welcome! 
### Areas for Contribution
- Improving the prompt engineering
- Enhancing the review feedback loop -> extending it with more agentic patterns
- Improving code quality  
- Adding / Proposing new features or optimizations

### Known Issues:
- **Streaming**: current streaming implementation is broken, contains lots of flickering and leaves artifacts in scroll history.
- **Prompting / Environment Setup**: There may be a mismatch of what version the llm chooses to use and what is currently installed on the system. 
