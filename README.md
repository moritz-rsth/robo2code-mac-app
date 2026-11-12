# Prompt2Robo model duel (Mac)

Type a request. Two language models each write a `run(robot)` program, and both programs run side by side in a live MuJoCo simulation. Held-out examples are scored automatically.

![Untuned model against the fine-tuned model building a tower](docs/duel-tower.gif)

## Start

You need an Apple-Silicon Mac (16 GB RAM or more is best), Python 3.10 or newer and about 10 GB of free disk.

1. Download this repo.
2. Double-click `run_mac.command`, or run `sh run_mac.sh` in Terminal.
3. The first start takes a few minutes, then http://localhost:8766 opens. Models download when you first use them.

`sh run_mac.sh --mock` tries the page with fake models and no download. `P2R_PORT=9000 sh run_mac.sh` uses another port.

## Examples

These are real programs written by the models, run in the simulator. They are hand-picked successes: the same fine-tuned model solved 218 of 306 held-out examples, and most of the untuned model's programs fail.

**Untuned Qwen3-8B against the fine-tuned model**

![Pyramid duel](docs/duel-pyramid.gif)
![2x2 square duel](docs/duel-square.gif)

**Plain requests, fine-tuned model**

| | |
|---|---|
| ![Pyramid](docs/pyramid.gif) | ![2x2 square](docs/square.gif) |
| ![Remove a cube from the stack](docs/remove-from-stack.gif) | ![Top cube next to the stack](docs/top-next-to-stack.gif) |
| ![Line](docs/line.gif) | ![Tower](docs/tower.gif) |

## The page

![The app after a run](docs/app.png)

- **Models:** each side has a picker with base models, full models and adapters. Selecting one downloads it if needed and frees the side's old model.
- **Manual code:** the last picker entry runs a `def run(robot):` program you type into the box under the simulation, where the other sides show the model's output. No model needed. The box and the model's programs are syntax highlighted (highlight.js, needs internet; offline the text stays plain).
- **Scene:** make a random scene with up to 48 cubes, or load a test example. Drag the 3D layout to look around.
- **Scene generator:** the pencil next to **New random scene** adds towers (2 to 10 high), pyramids (2 to 5 layers), flat squares (2x2 to 4x4), lines and L, T and cross shapes, each at a random free spot, in one colour or mixed. The cube count cannot go below what they need, and any extra cubes are placed as single cubes.
- **Run:** run the left side, the right side or both, then **Replay** the recording at any speed.
- **Download video:** under each simulation, once its run has ended. Opens a small export overlay for that side: tick **Original prompt** (the request, typed into a message over the blurred scene) and **Model and reasoning** (the request flies into a model card that slides away, then the reasoning streams in under cycling "Baking…", "Marinating…" words, then the answer with syntax highlighting; the model's name stays as a footnote), set the thinking pace (25 words/s by default; the request bubble follows it), the code pace (50 words/s, twice as fast) and the simulation speed (0.5x to 4x, 2x by default), and press **Export**. The server renders a 1920x1080 MP4 (H.264, 30 fps, plays in QuickTime, iMessage and WhatsApp) in the background, named after the request, the model and the time. The intro opens on the sharp start scene for 1.5 s. With both boxes off it is just the simulation. Without ffmpeg the button saves a GIF of the simulation instead.
- **Music:** the export has a soundtrack (the **Music** switch in the overlay), composed by `music.py` from the run itself with numpy only: the request text sets key, tempo and chords, every cube is a note, the typed words and code lines play along, and the simulation is scored to what the picture shows: `sim_worker.py` records when the gripper closes on a cube and opens again and when a cube comes to rest (on the table, on another cube with the stack's height, or dropped from above), in simulated seconds on the frames' clock, and the exporter places each on the video's timeline exactly as its frame, so a grab, a release, a landing (the bell climbs with the stack), a clunk and the fanfare for the finished stack are heard as they happen, lightly quantised to the beat and thinned at high speeds. If the music cannot be made, the silent video is delivered.
- **Manual model use:** the button next to **Run both** locks the request and scene controls and shows the system and user message a side would send, to edit by hand (**Prefill for left / right** fills them in). **Run left** / **Run right** send the edited texts to that side's model; its prompt format, thinking and token limit stay as in a normal run. A scene edited in the Scene_Info JSON is the one simulated; if the JSON does not read, the current scene is used and a warning says so. Manual runs are not checked against a dataset row. **Exit manual mode** brings the normal controls back.
- **Chain prompt:** under each simulation. The end state of that side becomes the new scene and the request box is cleared, so the next request continues from there. A cube still in the gripper is let go.
- **Result:** a run always plays to the end. The panel shows whether the task was solved and what the stricter dataset checker would have flagged.
- **Prompt format:** *Auto* picks what the model was trained with, or choose one yourself (see the table below).
- **Reasoning:** a thinking model (data_posv3) first reasons. A compact **Thinking…** field with a live dot shows that it is at work; click it to read the reasoning as it streams. Once the model closes its reasoning (`</think>`), the field reads **Thought** and its answer (code, refusal or one sentence) follows below it as its own block. Only that answer is run. If the model stops while still reasoning (token limit: 2600 new tokens for thinking models, 2048 otherwise) or writes nothing after its reasoning, the panel says so and nothing runs. Dataset rows of data_posv3 show their two stored reasonings under the request, and their source, family and start state in the row info.
- **Scene questions:** models with `"answers": true` in `p2r_meta.json` reply to a question about the scene in one sentence. The arm does not move, and dataset rows show correct or wrong. `"max_cubes"` there sets when the page warns about scene size.

Prompt formats, set per model with `"prompt_format"` in `p2r_meta.json`. Without that key the format comes from the older keys, as before: `"answers": true` is `answers_posv1`, `"scene_info": true` is `scene_v5`, neither is `objects`. An explicit `"enable_thinking"` always wins.

| `prompt_format` | Trained on | User message | System prompt | Thinking |
|---|---|---|---|---|
| `objects` | data_v0 to data_v4 | `Objects:` line, then the request | `system_prompt.md` | off |
| `scene_v5` | data_v5, data_posv0 | request, then `Scene_Info:` JSON | Scene_Info prompt | off |
| `answers_posv1` | data_posv1, data_posv2 | as `scene_v5` | `system_prompt_answers.md` | off |
| `posv3_think` | data_posv3 with reasoning | as `scene_v5` | `system_prompt_posv3.md` | on |
| `posv3_nothink` | data_posv3 without reasoning | as `scene_v5` | `system_prompt_posv3.md` without its "Before the code, think briefly" line | off |

**ID** finds any example by id, in any dataset version and split:

<img src="docs/id-search.png" alt="ID search with dataset and split pickers" width="480">

## Models

Public models work without a token. For private ones, open **Settings** and paste a Hugging Face token that can read them, or set `HF_TOKEN`. **Other…** in the picker takes a folder path or a Hugging Face link.

| Name ends in | What it is |
|---|---|
| `-MLX-4bit` | Quantized for MLX, the fast choice |
| `-full` | bf16 weights, loaded by MLX |
| `-MLX-adapter` | LoRA adapter in MLX format, runs on a 4-bit base |
| `-adapter` | PEFT adapter. Needs `.venv/bin/pip install torch peft`, a 16 GB base and a lot of RAM, so prefer the MLX version |

The best public model to start with is `Prompt2Robo v3-full-4b, Qwen3-4B · MLX 4-bit` (2.3 GB, 131 of 163 solved).

## Good to know

- If memory runs out, the panel says so. Choose a 4-bit model instead.
- Programs run in a sandbox: no imports, only the robot API.
- The arm in the picture is the NEURA LARA 5 model by [Stack Lab](https://github.com/carlo2004/Simulation-Neura-LARA-5). It is drawn only: physics and checks use the simulator's generic arm. Its meshes download on first start (see `lara5/NOTICE.md`), and `P2R_ARM=generic` turns it off.
- A shared copy of the app can contain `hf_readonly_token.txt`. Treat such a copy like a password. The file is ignored by git.

## Files

| Path | Purpose |
|---|---|
| `server.py`, `static/index.html` | Web server and page |
| `scene_gen.py` | Scene generator for towers, pyramids, squares and shapes |
| `models_scan.py`, `llm_backends.py` | Model discovery and MLX/Transformers backends |
| `sim_worker.py`, `floor_view.py`, `lara_view.py`, `lara_assets.py` | Simulation runs and picture; `sim_worker.py` also records the gripper and landing events the music follows |
| `recorder.py` | Records each run into a video file (ffmpeg from `imageio-ffmpeg`) |
| `exporter.py`, `fonts/` | Renders the shareable 1920x1080 export (Pillow, Pygments); Geist and Geist Mono fonts under the SIL Open Font License (`fonts/OFL.txt`) |
| `music.py` | Composes and synthesises the export's soundtrack from the run (numpy) |
| `sim/` | Scene, robot API and checkers from the dataset repo (`check_posv1.py` for data_posv1/2, `check_posv3.py` for data_posv3) |
| `datasets/`, `examples.jsonl` | Dataset versions for the ID search, held-out examples |
