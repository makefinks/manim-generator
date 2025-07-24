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
### 1. Input any data you want to have visualised as a video into `video_data.txt`
### 2. Execute the script
```bash
python manim_generator.py
```

### Example with specific models and video data:
```bash
python manim_generator.py --video_data "Explain the concept of neural networks with visual examples" --manim_model "openrouter/anthropic/claude-3.5-sonnet" --review_model "openrouter/anthropic/claude-3.5-sonnet" --review_cycles 3
```


### 3. Tweak the config
Manually configure what llm the coder and reviewer should use inside `manim_generator.py`.

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
- **Streaming**: current implementation performs no syntax highlighting and leaves artifacts in the console history when scrolling
- **Prompting / Environment Setup**: There may be a mismatch of what version the llm chooses to use and what is currently installed on the system. 

## Observations:

### LLMs
- Claude 3.7 Sonnet creates unique and stable scenes, allocating extra thinking budget increases the complexity of the visualisations but introduces visual errors or invalid use of the manim api
- o3-mini seems to be the most reliable coder but does not create the most impressive scenes - more testing is neeeded (also no image inputs yet)
- Claude 3.5 Sonnet V2 performs very well as the reviewer and coder - to no ones suprise ;)
- Deepseek R1 sometimes creates more impressive compositions but code often has syntax errors
- Gemini 2.0 Flash struggles creating runnable scripts without additional prompting

### Flow
- In the original flow the llm was instructed to only output one class (Manim Scene) so that it was easier to render the video with one command. Now each scene is its own class. This was done so I could extract the `--last-frame` for each scene and pass it to the review model. There might be a better way to do this!

## 🎬 Examples
You can find example videos in the [examples](examples/) directory.
