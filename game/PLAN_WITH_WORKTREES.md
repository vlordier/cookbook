# Parallel Implementation Plan with Git Worktrees

How to implement the game using multiple Claude Code agents working in parallel, each in its own git worktree.

## Core Idea

Git worktrees let you check out multiple branches simultaneously into separate directories. Each Claude Code session opens in one worktree directory and has no idea the others exist. They work in parallel without conflicts.

---

## Dependency Graph

```
[Step 1: Spike] --> [Step 2: Scaffold]
                          |
              +-----------+-----------+
              |           |           |
       [Step 3:       [Step 4:    [Step 6:
        Inference]     Webcam]     Game Engine]
              |           |           |
              +-----+-----+           |
                    |                 |
              [Step 5: Wire        (done)
               inference+webcam]     |
                    |                 |
                    +---------+-------+
                              |
                        [Step 7: Connect
                         VLM to game]
                              |
                        [Step 8: Polish]
```

Three agents can run in parallel after the scaffold is ready: inference module, webcam module, and game engine.

---

## Phase 1: Sequential Prerequisites (do manually)

Steps 1 and 2 are blocking for everything else. Complete these first in the main worktree before launching any agents.

- **Step 1 (spike):** Validate the model loads and classifies in a plain HTML file. This is exploratory, do it yourself.
- **Step 2 (scaffold):** `npm create vite@latest`, install deps, create empty module files, confirm `npm run dev` works.

Commit the scaffold to `pau/example/game`. This becomes the base branch for all parallel agents.

---

## Phase 2: Create Worktrees

Run from the repo root after the scaffold commit:

```bash
git worktree add ../game-inference -b pau/example/game/inference pau/example/game
git worktree add ../game-webcam    -b pau/example/game/webcam    pau/example/game
git worktree add ../game-engine    -b pau/example/game/engine    pau/example/game
```

Result: three directories, each on their own branch, all starting from the same scaffold commit.

```
cookbook/game/          <- pau/example/game            (your main session)
../game-inference/      <- pau/example/game/inference
../game-webcam/         <- pau/example/game/webcam
../game-engine/         <- pau/example/game/engine
```

---

## Phase 3: Launch Parallel Agents

Open a Claude Code session in each worktree in three separate terminals:

```bash
cd ../game-inference && claude
cd ../game-webcam    && claude
cd ../game-engine    && claude
```

Give each agent a focused prompt based on IMPLEMENTATION.md. Keep file ownership strict to prevent merge conflicts.

**Agent 1 - Inference module (Step 3):**
> Read IMPLEMENTATION.md Step 3. Implement `src/inference/model.ts`, `src/inference/prompt.ts`, and `src/inference/steering.ts` exactly as described. The scaffold already exists. Do not touch any files outside `src/inference/`.

**Agent 2 - Webcam module (Step 4):**
> Read IMPLEMENTATION.md Step 4. Implement `src/webcam/capture.ts` and `src/webcam/sampler.ts` exactly as described. The scaffold already exists. Do not touch any files outside `src/webcam/`.

**Agent 3 - Game engine (Step 6):**
> Read IMPLEMENTATION.md Step 6. Implement `src/game/car.ts`, `src/game/road.ts`, `src/game/renderer.ts`, and `src/game/engine.ts` with keyboard input. The scaffold already exists. Do not touch any files outside `src/game/`.

The tight file ownership boundaries are important. They prevent merge conflicts because each agent only edits its own directory.

---

## Phase 4: Integration Waves

### Wave 1 - Wire inference to webcam (Step 5)

Once inference and webcam branches are done, merge them and launch a new agent:

```bash
git merge pau/example/game/inference
git merge pau/example/game/webcam
git worktree add ../game-wiring -b pau/example/game/wiring pau/example/game
cd ../game-wiring && claude
```

Agent prompt:
> Read IMPLEMENTATION.md Step 5. Wire the inference and webcam modules together in `src/main.ts`. Both modules are already implemented. Log the live steering direction to the console.

### Wave 2 - Connect VLM to game (Step 7)

Once the wiring branch and game engine branch are both done, merge them and launch another agent:

```bash
git merge pau/example/game/wiring
git merge pau/example/game/engine
git worktree add ../game-connect -b pau/example/game/connect pau/example/game
cd ../game-connect && claude
```

Agent prompt:
> Read IMPLEMENTATION.md Step 7. Replace the keyboard input in `src/game/engine.ts` with the live `currentDirection` from the inference pipeline. Wire everything together in `src/main.ts`.

### Wave 3 - Polish (Step 8)

Work directly on `pau/example/game` after merging Step 7:

```bash
git merge pau/example/game/connect
```

Then implement the loading screen, browser check, score display, and game over screen as described in IMPLEMENTATION.md Step 8.

---

## Phase 5: Clean Up Worktrees

```bash
git worktree remove ../game-inference
git worktree remove ../game-webcam
git worktree remove ../game-engine
git worktree remove ../game-wiring
git worktree remove ../game-connect
```

---

## Summary Table

| Phase | Agents | Branches | Blocks on |
|---|---|---|---|
| Spike + Scaffold | 1 (you) | `pau/example/game` | nothing |
| Parallel build | 3 | `/inference`, `/webcam`, `/engine` | scaffold |
| Wiring (Step 5) | 1 | `/wiring` | inference + webcam merged |
| VLM connect (Step 7) | 1 | `/connect` | wiring + engine merged |
| Polish (Step 8) | 1 | `pau/example/game` | Step 7 merged |

---

## Key Rules for Agents

- Each agent owns exactly one subdirectory under `src/`. It must not edit files outside it.
- `src/main.ts` is off-limits for all parallel agents. It is only touched during integration steps (5 and 7).
- Agents should read IMPLEMENTATION.md and ARCHITECTURE.md at the start of their session for full context.
- If an agent is uncertain about an interface boundary (e.g. what type to export), it should define the type explicitly and document it in a comment so the integration agent can see it clearly.
